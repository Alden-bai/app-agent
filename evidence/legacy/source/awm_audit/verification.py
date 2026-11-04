from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from .assets import load_asset, normalize
from .common import AWM, ROOT, backup, now, read_json, save_json


def verifier_entry(run_dir, task, mode):
    filename = "gen_verifier.pure_code.jsonl" if mode == "code" else "gen_verifier.jsonl"
    entries = [e for e in load_asset(run_dir, filename)
               if normalize(e["scenario"]) == normalize(task["scenario"]) and e.get("task_idx") == task["task_idx"]]
    if len(entries) != 1:
        raise ValueError(f"Official verifier record count={len(entries)}; no arbitrary duplicate selection")
    if entries[0].get("task") != task["task"]:
        raise ValueError("Official verifier task text mismatch")
    return entries[0]


def execute_verifier(entry, mode, folder, *, timeout=30):
    work = Path(folder) / ("verifier_" + mode)
    work.mkdir(parents=True, exist_ok=True)
    for name in ("initial", "final"):
        backup(Path(folder) / (name + ".db"), work / (name + ".db"))
    before = {name: (work / (name + ".db")).read_bytes() for name in ("initial", "final")}
    request = {"entry": entry, "mode": mode, "initial": str(work / "initial.db"),
               "final": str(work / "final.db"), "answer": read_json(Path(folder) / "trajectory.json")["final_answer"]}
    save_json(work / "request.json", request)
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "LC_ALL", "TZ")}
    env["PYTHONPATH"] = str(ROOT)
    try:
        result = subprocess.run([sys.executable, "-m", "awm_audit.worker", str(work / "request.json"), str(work / "result.json")],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
        (work / "stdout.log").write_text(result.stdout)
        (work / "stderr.log").write_text(result.stderr)
        if result.returncode or not (work / "result.json").exists():
            return {"status": "verifier_exception", "accepted": None, "error": "Worker failed; inspect verifier stderr.log"}
        verdict = read_json(work / "result.json")
        if any((work / (n + ".db")).read_bytes() != b for n, b in before.items()):
            return {"status": "verifier_exception", "accepted": None, "error": "Verifier modified its database copy", **verdict}
        if verdict["verdict"] == "judge_error":
            return {"status": "verifier_exception", "accepted": None, **verdict}
        return {"status": "valid", "accepted": verdict["verdict"] == "complete" if mode == "code" else None, **verdict}
    except subprocess.TimeoutExpired:
        return {"status": "verifier_timeout", "accepted": None, "timeout_seconds": timeout}


class LoggedClient:
    """Keep the official prompt, while controlling transport retries and logging."""
    def __init__(self, config):
        from openai import AsyncOpenAI
        self.config = config
        self.client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key,
                                  timeout=config.timeout, max_retries=0)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.log = []

    async def create(self, **kwargs):
        from openai import APIConnectionError, APIStatusError
        for attempt in range(self.config.retries + 1):
            record = {"attempt": attempt + 1, "started_at": now(), "request": kwargs}
            self.log.append(record)
            try:
                response = await self.client.chat.completions.create(**kwargs)
                record.update({"finished_at": now(), "response": response.model_dump(mode="json")})
                return response
            except Exception as e:
                record.update({"finished_at": now(), "error_type": type(e).__name__, "error": self.config.redact(e)})
                if isinstance(e, APIStatusError):
                    record["status_code"] = e.status_code
                transient = isinstance(e, APIConnectionError) or (isinstance(e, APIStatusError) and
                              (e.status_code == 429 or e.status_code >= 500))
                if not transient or attempt == self.config.retries:
                    raise
                await asyncio.sleep(2 ** attempt)


async def judge_sql(config, task, extraction, trajectory, folder, *, circuit=None):
    if circuit and circuit.get("blocked"):
        return {"status": "llm_unavailable", "accepted": None, "reason": circuit["blocked"]}
    sys.path.insert(0, str(AWM)) if str(AWM) not in sys.path else None
    from awm.core import verify
    # Only transport is wrapped; official prompt, temperature and verdict parser
    # are unchanged. Independent labels/rules are never sent to this Judge.
    client = LoggedClient(config)
    original = verify.AsyncOpenAI
    verify.AsyncOpenAI = lambda **kwargs: client
    try:
        classification, result = await verify.run_llm_judge(
            task["task"], extraction, config.base_url, config.api_key, config.model, trajectory)
    finally:
        verify.AsyncOpenAI = original
        await client.client.close()
        save_json(Path(folder) / "llm_requests.json", client.log)
    status = "valid"
    if classification == "judge_error":
        status = "llm_call_failure" if client.log and "error" in client.log[-1] else "llm_invalid_output"
    if circuit is not None and client.log and client.log[-1].get("status_code") in (401, 402, 403):
        circuit["blocked"] = {"status_code": client.log[-1]["status_code"], "at": now()}
    return {"status": status, "classification": classification,
            "accepted": classification == "complete" if status == "valid" else None, "details": result}
