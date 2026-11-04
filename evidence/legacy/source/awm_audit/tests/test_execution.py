import asyncio
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from awm_audit import runner, verification
from awm_audit.common import FILES, backup, digest, read_json, save_json
from awm_audit.config import LLMConfig
from awm_audit.oracle import TaskSpec, call, mutate, row_rule


def test_resume_and_model_rule_asset_cache_invalidation(tmp_path, monkeypatch):
    root = tmp_path / "run"
    task = {"scenario": "fixture", "task_idx": 0, "task": "Set quantity to two"}
    manifest = {"seed": 20261021, "revision": "a" * 40, "dataset": "fixture", "scenarios": ["fixture"],
                "tasks": [task], "assets_complete": True, "files": {}}
    for filename in FILES:
        data = [dict(task)] if "verifier" in filename else [{"scenario": "fixture"}]
        p = root / "assets" / (filename + ".json")
        save_json(p, data)
        manifest["files"][filename] = {"selected_sha256": digest(p)}
    save_json(root / "manifest.json", manifest)
    initial = root / "initials/fixture.db"
    initial.parent.mkdir()
    with sqlite3.connect(initial) as db:
        db.executescript("CREATE TABLE items(id INTEGER PRIMARY KEY,quantity INTEGER);INSERT INTO items VALUES(1,1)")
    save_json(initial.with_suffix(".initialization.json"), {"errors": [], "foreign_key_check": [], "integrity_check": [["ok"]], "database_sha256": digest(initial)})
    spec = TaskSpec([call("set", body={"quantity": 2})], [row_rule("items", {"id": 1, "quantity": 2})],
                    [mutate(0, "body.quantity", 0), mutate(0, "body.quantity", 3)])
    monkeypatch.setattr(runner, "load_spec", lambda *a: spec)
    calls = []
    async def execute(run_dir, task, case, source, folder):
        calls.append(case["name"])
        folder.mkdir()
        backup(source, folder / "initial.db")
        backup(source, folder / "final.db")
        with sqlite3.connect(folder / "final.db") as db:
            for step in case["calls"]:
                db.execute("UPDATE items SET quantity=?", (step["arguments"]["body"]["quantity"],))
        payload = {"failure": None, "final_answer": case["answer"], "trajectory": []}
        save_json(folder / "trajectory.json", payload)
        return payload
    monkeypatch.setattr(runner, "execute_case", execute)
    monkeypatch.setattr(runner, "execute_verifier", lambda *a: {"status": "valid", "accepted": False, "details": {}})
    config = LLMConfig("https://example.invalid", "m1", "fake")
    options = dict(judge_enabled=False)
    asyncio.run(runner.run(root, config, **options))
    assert len(calls) == 4
    state = read_json(root / "task_states.json")["fixture:0"]
    first = state["attempt"]
    assert state["status"] == "completed_by_tools"
    asyncio.run(runner.run(root, config, **options))
    assert len(calls) == 4
    asyncio.run(runner.run(root, LLMConfig(config.base_url, "m2", "fake"), **options))
    assert len(calls) == 8
    assert read_json(root / "task_states.json")["fixture:0"]["attempt"] != first
    spec.notes = "A changed frozen rule"
    asyncio.run(runner.run(root, config, **options))
    assert len(calls) == 12
    changed = root / "assets/gen_scenario.jsonl.json"
    save_json(changed, [{"scenario": "fixture", "description": "changed"}])
    manifest["files"]["gen_scenario.jsonl"]["selected_sha256"] = digest(changed)
    save_json(root / "manifest.json", manifest)
    asyncio.run(runner.run(root, config, **options))
    assert len(calls) == 16
    with sqlite3.connect(initial) as db:
        db.execute("UPDATE items SET quantity=999")
    asyncio.run(runner.run(root, config, **options))
    assert len(calls) == 16
    assert read_json(root / "task_states.json")["fixture:0"]["status"] == "runtime_blocked"


def test_official_judge_invalid_output_is_not_false_negative(tmp_path, monkeypatch):
    requests = []
    class Client:
        def __init__(self, config):
            self.log = [{"request": "recorded"}]
            self.client = SimpleNamespace(close=self.close)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        async def close(self):
            pass
        async def create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))])
    monkeypatch.setattr(verification, "LoggedClient", Client)
    config = LLMConfig("https://example.invalid", "fixture", "fake-key")
    result = asyncio.run(verification.judge_sql(config, {"task": "A real task"}, {"db_fact": 3}, [], tmp_path))
    assert result["status"] == "llm_invalid_output"
    assert result["accepted"] is None
    assert len(requests) == 1  # invalid JSON must not trigger favorable retries
    assert "A real task" in requests[0]["messages"][1]["content"]
    assert "oracle" not in requests[0]["messages"][1]["content"]


def test_transport_retries_but_not_malformed_success():
    from openai import APIConnectionError
    import httpx
    attempts = []
    config = LLMConfig("https://example.invalid", "fixture", "fake-key", retries=2)
    client = verification.LoggedClient(config)
    actual = client.client
    async def create(**kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
        return SimpleNamespace(model_dump=lambda **kw: {"choices": [{"message": {"content": "invalid JSON"}}]})
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    async def run():
        try:
            return await client.create(model="fixture", messages=[])
        finally:
            await actual.close()
    response = asyncio.run(run())
    assert len(attempts) == 3
    assert len(client.log) == 3
    assert response is not None
