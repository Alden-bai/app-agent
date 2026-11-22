"""Run the AWM CLI with a shared rate limit for OpenAI GPT-6 Luna Judge calls."""
import asyncio
from contextlib import closing
from email.utils import parsedate_to_datetime
from functools import lru_cache
import math
import os
from pathlib import Path
import random
import sqlite3
import time
from urllib.parse import urlsplit

from harness_awm.backend import BudgetExceeded
from harness_awm.io import write_json
from harness_awm.protocol import estimate_tokens


ROOT = Path(__file__).resolve().parents[1]
JUDGE_STAGES = {"judge", "judge_v2", "judge_v2_review"}
WINDOW_SECONDS = 60
MAX_ATTEMPTS = 3


class RateLimit:
    def __init__(self, path, rpm, tpm):
        self.path, self.rpm, self.tpm = Path(path), rpm, tpm
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS sends (id INTEGER PRIMARY KEY AUTOINCREMENT, started REAL, tokens INTEGER)")
            db.execute("CREATE TABLE IF NOT EXISTS cooldown (id INTEGER PRIMARY KEY, until REAL)")
            db.execute("INSERT OR IGNORE INTO cooldown VALUES (1, 0)")

    def _reserve(self, tokens):
        # Serialize read, check and insert even when two CLI processes dispatch together.
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            now = time.time()
            db.execute("DELETE FROM sends WHERE started <= ?", (now - WINDOW_SECONDS,))
            until = db.execute("SELECT until FROM cooldown WHERE id = 1").fetchone()[0]
            rows = db.execute("SELECT started, tokens FROM sends ORDER BY started, id").fetchall()
            if rows:
                started, previous_tokens = rows[-1]
                until = max(until, started + WINDOW_SECONDS * max(1 / self.rpm, previous_tokens / self.tpm))
                if len(rows) >= self.rpm or sum(row[1] for row in rows) + tokens > self.tpm:
                    until = max(until, rows[0][0] + WINDOW_SECONDS)
            if until > now:
                return None, until - now
            return db.execute("INSERT INTO sends(started, tokens) VALUES (?, ?)", (now, tokens)).lastrowid, 0

    async def acquire(self, tokens):
        if tokens > self.tpm:
            raise ValueError(f"judge_request_exceeds_tpm: estimated {tokens}, configured {self.tpm}")
        while True:
            reservation, delay = self._reserve(tokens)
            if reservation is not None:
                return reservation
            await asyncio.sleep(delay)

    def refund(self, reservation):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("DELETE FROM sends WHERE id = ?", (reservation,))

    def defer(self, seconds):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("UPDATE cooldown SET until = max(until, ?) WHERE id = 1", (time.time() + seconds,))


@lru_cache(maxsize=1)
def luna_encoding():
    try:
        import tiktoken
        return tiktoken.get_encoding("o200k_base")
    except (ImportError, OSError):
        return None


def estimated_tokens(request):
    messages = request["messages"]
    output = request.get("max_completion_tokens", request.get("max_tokens"))
    if not isinstance(output, int) or output < 1:
        raise ValueError("rate_limited_judge_requires_max_output_tokens")
    encoding = luna_encoding()
    if encoding is None or any(not isinstance(message.get("content"), str) for message in messages):
        return estimate_tokens(messages) + output + 512
    prompt = sum(len(encoding.encode_ordinary(message["content"])) + 8 for message in messages) + 16
    return math.ceil(prompt * 1.10) + output + 512


def retry_delay(error, attempt):
    response = getattr(error, "response", None)
    retry_after = response.headers.get("retry-after") if response is not None else None
    seconds = 0
    if retry_after:
        try:
            seconds = float(retry_after)
        except ValueError:
            try:
                seconds = parsedate_to_datetime(retry_after).timestamp() - time.time()
            except (TypeError, ValueError, OverflowError):
                pass
    return max(5 * 2 ** attempt, seconds if math.isfinite(seconds) else 0) + random.uniform(0, 1)


def paced_transport(api_request, limit):
    async def request(config, kwargs, budget, key, stage, folder, *, consume_hold=None):
        if (stage not in JUDGE_STAGES or kwargs["model"] != "gpt-6-luna"
                or urlsplit(config.base_url).hostname != "api.openai.com"):
            return await api_request(config, kwargs, budget, key, stage, folder, consume_hold=consume_hold)

        tokens = estimated_tokens(kwargs)
        folder = Path(folder)
        for attempt in range(MAX_ATTEMPTS):
            reservation = await limit.acquire(tokens)
            suffix = f"rate_retry_{attempt:02d}"
            call_key, call_folder = (f"{key}/{suffix}", folder / suffix) if attempt else (key, folder)
            try:
                data = await api_request(config, kwargs, budget, call_key, stage, call_folder, consume_hold=consume_hold)
            except BudgetExceeded:
                limit.refund(reservation)
                raise
            except Exception as error:
                if (getattr(error, "status_code", None) != 429
                        or getattr(error, "code", None) not in {None, "rate_limit_exceeded", "slow_down"}):
                    raise
                limit.defer(retry_delay(error, attempt))
                if attempt + 1 == MAX_ATTEMPTS:
                    raise
            else:
                if attempt:
                    write_json(folder / "response.json", data)
                return data

    return request


def main():
    from dotenv import load_dotenv
    from harness_awm import backend

    load_dotenv(ROOT / ".env", override=False)
    try:
        rpm, tpm = (int(os.environ[name]) for name in ("JUDGE_OPENAI_RPM", "JUDGE_OPENAI_TPM"))
        if min(rpm, tpm) < 1:
            raise ValueError
    except (KeyError, ValueError):
        raise SystemExit("Set JUDGE_OPENAI_RPM and JUDGE_OPENAI_TPM to positive integers in .env or the shell.") from None

    limit = RateLimit(ROOT / "runs" / "luna_judge_rate.sqlite", rpm, tpm)
    # Install before the CLI imports the Judge modules. The regular CLI and
    # existing runs keep their original transport and frozen source files.
    backend.api_request = paced_transport(backend.api_request, limit)
    from harness_awm.cli import main as run_cli

    run_cli()


if __name__ == "__main__":
    main()
