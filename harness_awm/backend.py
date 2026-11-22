import asyncio
import os
import sqlite3
import time
from pathlib import Path
from .io import canonical, now, write_json
from .protocol import estimate_tokens


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, path, max_requests, max_tokens):
        self.path = Path(path)
        self.max_requests, self.max_tokens = max_requests, max_tokens
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS requests (id INTEGER PRIMARY KEY, request_key TEXT UNIQUE, stage TEXT, reserved INTEGER, status TEXT, usage TEXT, started TEXT, finished TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS holds (hold_key TEXT PRIMARY KEY, reserved INTEGER NOT NULL)")

    def hold(self, key, tokens):
        """Reserve one future physical request atomically across concurrent episodes."""
        with sqlite3.connect(self.path, timeout=30, isolation_level=None) as db:
            db.execute("BEGIN IMMEDIATE")
            count, total = db.execute("SELECT count(*),coalesce(sum(reserved),0) FROM requests").fetchone()
            hc, ht = db.execute("SELECT count(*),coalesce(sum(reserved),0) FROM holds WHERE hold_key!=?", (key,)).fetchone()
            if ((self.max_requests is not None and count + hc + 1 > self.max_requests)
                    or (self.max_tokens is not None and total + ht + tokens > self.max_tokens)):
                raise BudgetExceeded("global_budget_exhausted")
            db.execute("INSERT OR REPLACE INTO holds VALUES(?,?)", (key, tokens))
            db.commit()

    def release(self, key):
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute("DELETE FROM holds WHERE hold_key=?", (key,))

    def request_record(self, key):
        with sqlite3.connect(self.path, timeout=30) as db:
            row = db.execute("SELECT reserved,status FROM requests WHERE request_key=?", (key,)).fetchone()
        return {"reserved": row[0], "status": row[1]} if row else None

    def reserve(self, key, stage, messages, output_tokens, consume_hold=None):
        upper = estimate_tokens(messages) + output_tokens + 512
        with sqlite3.connect(self.path, timeout=30, isolation_level=None) as db:
            db.execute("BEGIN IMMEDIATE")
            count, total = db.execute("SELECT count(*),coalesce(sum(reserved),0) FROM requests").fetchone()
            hc, ht = db.execute("SELECT count(*),coalesce(sum(reserved),0) FROM holds WHERE hold_key!=?", (consume_hold or "",)).fetchone()
            if consume_hold and not db.execute("SELECT 1 FROM holds WHERE hold_key=?", (consume_hold,)).fetchone():
                raise BudgetExceeded("finalization_hold_missing")
            if ((self.max_requests is not None and count + hc >= self.max_requests)
                    or (self.max_tokens is not None and total + ht + upper > self.max_tokens)):
                raise BudgetExceeded("global_budget_exhausted")
            cursor = db.execute("INSERT INTO requests(request_key,stage,reserved,status,started) VALUES(?,?,?,?,?)",
                                (key, stage, upper, "in_flight", now()))
            if consume_hold:
                db.execute("DELETE FROM holds WHERE hold_key=?", (consume_hold,))
            db.commit()
            return cursor.lastrowid

    def finish(self, rid, usage=None):
        with sqlite3.connect(self.path, timeout=30) as db:
            if usage and isinstance(usage.get("total_tokens"), int) and usage["total_tokens"] >= 0:
                db.execute("UPDATE requests SET reserved=?,usage=?,status='response',finished=? WHERE id=?",
                           (usage["total_tokens"], canonical(usage), now(), rid))
            else:
                db.execute("UPDATE requests SET status='usage_unknown',finished=? WHERE id=?", (now(), rid))

    def summary(self):
        with sqlite3.connect(self.path) as db:
            count, total = db.execute("SELECT count(*),coalesce(sum(reserved),0) FROM requests").fetchone()
            unknown = db.execute("SELECT count(*) FROM requests WHERE status!='response'").fetchone()[0]
            hc, ht = db.execute("SELECT count(*),coalesce(sum(reserved),0) FROM holds").fetchone()
        return {"physical_requests": count, "tokens_accounted_or_reserved": total,
                "unknown_usage_requests": unknown, "held_requests": hc, "held_tokens": ht}


async def _api_request_once(config, kwargs, budget, key, stage, folder, *, consume_hold=None):
    from openai import AsyncOpenAI
    if not os.environ.get(config.api_key_env, "").strip():
        raise ValueError("missing_api_key_environment_variable: " + config.api_key_env)
    wire = dict(kwargs)
    if config.reasoning_effort is not None:
        wire['reasoning_effort'] = config.reasoning_effort
    if config.max_tokens_alias and "max_completion_tokens" in wire:
        wire["max_tokens"] = wire.pop("max_completion_tokens")
    if config.enable_thinking is not None:
        extra = dict(wire.get("extra_body") or {})
        extra.pop("thinking", None)  # DeepSeek's switch is not a Qwen parameter.
        extra["enable_thinking"] = config.enable_thinking
        wire["extra_body"] = extra
    if config.context_limit is not None:
        output_tokens = wire.get("max_completion_tokens", wire.get("max_tokens", 0))
        from .protocol_v2 import fits_context
        if not fits_context(wire["messages"], output_tokens, config.context_limit):
            raise ValueError("api_context_limit_exceeded")
    write_json(Path(folder) / "request.json", wire)
    rid = budget.reserve(key, stage, wire["messages"], wire.get("max_completion_tokens", wire.get("max_tokens", 0)), consume_hold)
    try:
        async with AsyncOpenAI(base_url=config.base_url, api_key=os.environ[config.api_key_env],
                               max_retries=0, timeout=config.timeout) as client:
            response = await client.chat.completions.create(**wire)
        data = response.model_dump(mode="json")
        write_json(Path(folder) / "response.json", data)
        budget.finish(rid, data.get("usage"))
        return data
    except BaseException as exc:
        budget.finish(rid)
        # Preserve provider failure evidence without persisting Authorization headers.
        error = {"type": type(exc).__name__, "status_code": getattr(exc, "status_code", None),
                 "body": getattr(exc, "body", None), "usage_status": "unknown_reserved"}
        causes, current = [], exc
        while current is not None and len(causes) < 5:
            causes.append({"type": type(current).__name__, "message": str(current)[:2000]})
            current = current.__cause__ or current.__context__
        error["causes"] = causes
        secret = os.environ.get(config.api_key_env, "")
        def redact(value):
            if isinstance(value, str):
                return value.replace(secret, "[REDACTED]") if secret else value
            if isinstance(value, dict):
                return {str(k): redact(v) for k, v in value.items()}
            if isinstance(value, list):
                return [redact(v) for v in value]
            if value is None or isinstance(value, (bool, int, float)):
                return value
            return redact(str(value))
        write_json(Path(folder) / "error.json", redact(error))
        raise


async def api_request(config, kwargs, budget, key, stage, folder, *, consume_hold=None):
    from .judge_recovery import request_with_retries
    return await request_with_retries(_api_request_once, config, kwargs, budget, key, stage, folder,
                                      consume_hold=consume_hold)


# The rate limiter wraps each physical attempt, without multiplying retry loops.
api_request._single_request = _api_request_once


class ScriptedBackend:
    """Actions supplied by the operator; no access to private runtime evidence."""
    def __init__(self, actions):
        self.actions = list(actions)
        self.index = 0

    async def generate(self, messages, folder, turn, **options):
        if self.index >= len(self.actions):
            raise RuntimeError("script_exhausted")
        item = self.actions[self.index]
        self.index += 1
        if isinstance(item, dict) and "raw_output" in item:
            content = item["raw_output"]
            reason = item.get("finish_reason", "stop")
        else:
            content, reason = (item if isinstance(item, str) else canonical(item)), "stop"
        return {"content": content, "finish_reason": reason, "test_only": True,
                "model_version": "scripted-v1", "usage": None}


class APIBackend:
    def __init__(self, config, budget, semaphore, rollout_id):
        self.config, self.budget, self.semaphore, self.rollout_id = config, budget, semaphore, rollout_id

    async def generate(self, messages, folder, turn, *, output_tokens=None, final_hold=None):
        c = self.config
        kwargs = {"model": c.actor_api.model, "messages": messages,
                  "temperature": c.actor_api.temperature, "max_completion_tokens": output_tokens or c.max_output_tokens}
        queued = time.monotonic()
        self.phase = "model_queue"
        async with self.semaphore:
            queue_seconds = time.monotonic() - queued
            self.phase = "model_request"
            started = time.monotonic()
            data = await api_request(c.actor_api, kwargs, self.budget, f"{self.rollout_id}/actor/{turn}", "actor", folder,
                                     consume_hold=final_hold)
        choice = data["choices"][0]
        return {"content": choice["message"].get("content") or "", "finish_reason": choice["finish_reason"],
                "test_only": False, "usage": data.get("usage"), "model_version": data.get("model"),
                "system_fingerprint": data.get("system_fingerprint"), "queue_seconds": queue_seconds,
                "model_seconds": time.monotonic() - started}


# Kept at the existing import location; torch is loaded only on actual use.
from .training.actor import TrainingActorBackend


class ProcessEvaluator:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("P5 process rubric calibration is not implemented")

