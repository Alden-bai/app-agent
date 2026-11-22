"""Run the AWM CLI with a shared rate limit for OpenAI GPT-6 Luna Judge calls."""
import asyncio
from contextlib import closing
from functools import lru_cache
import math
from pathlib import Path
import sqlite3
import time
from urllib.parse import urlsplit

from harness_awm.protocol import estimate_tokens
from .judge_recovery import JUDGE_STAGES, request_with_retries, retry_delay


ROOT = Path(__file__).resolve().parents[1]
WINDOW_SECONDS = 60


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


def paced_transport(api_request, limit):
    async def request(config, kwargs, budget, key, stage, folder, *, consume_hold=None):
        if (stage not in JUDGE_STAGES or kwargs["model"] != "gpt-6-luna"
                or urlsplit(config.base_url).hostname != "api.openai.com"):
            return await api_request(config, kwargs, budget, key, stage, folder, consume_hold=consume_hold)

        tokens = estimated_tokens(kwargs)
        folder = Path(folder)
        send = getattr(api_request, "_single_request", api_request)
        return await request_with_retries(send, config, kwargs, budget, key, stage, folder,
                                          consume_hold=consume_hold, limit=limit, tokens=tokens)

    return request



def install(config, default_db):
    """Install before or after Judge import, once per process and rate database."""
    if config.judge_rpm is None:
        return
    from . import judge_runtime, rubric_judge, backend, trajectory_quality
    path = config.judge_rate_db or str(default_db)
    original = getattr(backend.api_request, '_unpaced_transport', backend.api_request)
    wrapped = paced_transport(original, RateLimit(path, config.judge_rpm, config.judge_tpm))
    wrapped._unpaced_transport = original
    backend.api_request = wrapped
    judge_runtime.api_request = wrapped
    rubric_judge.api_request = wrapped
    trajectory_quality.api_request = wrapped
