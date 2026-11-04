import asyncio
import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

from awm_audit.assets import sample_names, validate_assets, prepare
from awm_audit.common import FILES, backup, digest, read_json, save_json
from awm_audit.config import load_config, validate_config, LLMConfig
from awm_audit.oracle import Rule, TaskSpec, evaluate, row_rule, equivalent, resolve
from awm_audit.report import metrics, generate
from awm_audit.verification import execute_verifier, verifier_entry


@pytest.fixture
def dbs(tmp_path):
    initial = tmp_path / "initial.db"
    with sqlite3.connect(initial) as db:
        db.executescript("CREATE TABLE items(id INTEGER PRIMARY KEY,name TEXT,quantity INTEGER); INSERT INTO items VALUES(1,'old',1);")
    final = tmp_path / "final.db"
    backup(initial, final)
    return initial, final


def test_configuration_precedence_and_no_execution(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('OPENAI_BASE_URL="https://file.example/v1"\nDEEPSEEK_MODEL="file-model"\nDEEPSEEK_API_KEY="not-a-real-key-123"\n')
    for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL", "AWM_SYN_OVERRIDE_MODEL", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://process.example/v1")
    config = load_config(env, model="cli-model")
    assert config.base_url == "https://process.example/v1"
    assert config.model == "cli-model"
    assert config.api_key not in json.dumps(config.public())
    assert config.api_key not in config.redact("server echoes " + config.api_key)
    env.write_text('DEEPSEEK_API_KEY="$(touch /tmp/should-not-run)"\n')
    assert load_config(env).api_key == "$(touch /tmp/should-not-run)"
    with pytest.raises(ValueError):
        validate_config(LLMConfig("https://key@host/v1", "model", "key"))


def test_fixed_sampling_order_independent():
    frame = [{"scenario": f"s{i:04}"} for i in range(1000)]
    chosen = sample_names(frame)
    assert chosen == sample_names(list(reversed(frame)))
    assert len(set(chosen)) == 10
    with pytest.raises(ValueError):
        sample_names(frame + [frame[0]])


def test_sqlite_backup_includes_wal_and_prevents_same_path(dbs):
    initial, target = dbs
    writer = sqlite3.connect(initial)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("INSERT INTO items VALUES(2,'wal',3)")
    writer.commit()
    backup(initial, target)
    with sqlite3.connect(target) as db:
        assert db.execute("SELECT name FROM items WHERE id=2").fetchone() == ("wal",)
    writer.close()
    with pytest.raises(ValueError):
        backup(initial, initial)


def test_independent_rule_same_row_and_duplicate_writes(dbs):
    initial, final = dbs
    spec = TaskSpec([], [row_rule("items", {"name": "target", "quantity": 2}, new=True)])
    with sqlite3.connect(final) as db:
        db.executescript("INSERT INTO items VALUES(2,'target',1); INSERT INTO items VALUES(3,'wrong',2);")
    assert evaluate(spec, initial, final, "")["correct"] is False
    with sqlite3.connect(final) as db:
        db.execute("UPDATE items SET quantity=2 WHERE id=2")
    assert evaluate(spec, initial, final, "")["correct"] is True
    with sqlite3.connect(final) as db:
        db.execute("INSERT INTO items VALUES(4,'target',2)")
    assert evaluate(spec, initial, final, "")["correct"] is False


def test_answer_order_and_required_api_evidence(dbs):
    initial, final = dbs
    expected = [{"id": 2}, {"id": 1}]
    spec = TaskSpec([], [], answer_expected=expected, answer_ordered=False)
    assert evaluate(spec, initial, final, '[{"id":1},{"id":2}]')["correct"] is True
    assert evaluate(spec, initial, final, "[]")["correct"] is False
    spec.required_tools = ["list_items"]
    assert evaluate(spec, initial, final, json.dumps(expected))["correct"] is False
    trace = [{"tool_calls": [{"arguments": {"tool_name": "list_items"}}]}]
    assert evaluate(spec, initial, final, json.dumps(expected), trace)["correct"] is True
    assert not equivalent(True, 1)
    assert equivalent(0.1 + 0.2, 0.3)


def test_oracle_exceptions_and_empty_predicates_are_unknown(dbs):
    initial, final = dbs
    assert evaluate(TaskSpec([], []), initial, final, "")["status"] == "unknown"
    assert evaluate(TaskSpec([], [Rule("bad", "SELECT nonexistent FROM items")]), initial, final, "")["correct"] is None
    assert evaluate(TaskSpec([], [Rule("empty", "SELECT 1 WHERE 0")]), initial, final, "")["status"] == "unknown"


def test_recipe_dedup_and_answer_formatting(dbs):
    initial, final = dbs
    spec = TaskSpec([], [Rule("yes", "SELECT 1")], [{"kind": "omit", "index": 0, "name": "duplicate"}])
    assert len(spec.cases()) == 1  # state-only tasks cannot inflate coverage with closing prose
    formatted = resolve({"$result": -1, "$sort": {"path": ["items"], "by": "price", "reverse": True}}, final,
                        [{"items": [{"price": 1}, {"price": 3}]}])
    assert formatted["items"][0]["price"] == 3


def test_official_worker_and_timeout_separation(dbs, tmp_path, monkeypatch):
    initial, final = dbs
    save_json(tmp_path / "trajectory.json", {"final_answer": "done"})
    entry = {"verification": {"code": 'def verify_task_completion(initial_db_path, final_db_path, final_answer=None):\n return {"result":"complete"}\n'}}
    result = execute_verifier(entry, "code", tmp_path)
    assert result["status"] == "valid" and result["accepted"] is True
    bad = {"verification": {"code": 'def verify_task_completion(initial_db_path, final_db_path):\n raise RuntimeError("broken")\n'}}
    result = execute_verifier(bad, "code", tmp_path)
    assert result["status"] == "verifier_exception" and result["accepted"] is None
    def timeout(*a, **kw):
        raise subprocess.TimeoutExpired(a[0], 30)
    monkeypatch.setattr(subprocess, "run", timeout)
    result = execute_verifier(entry, "code", tmp_path)
    assert result["status"] == "verifier_timeout" and result["accepted"] is None


def fixture_case(task, name, label, accepted=None, status="valid"):
    return {"task": task, "case": name, "status": "valid", "oracle": {"correct": label},
            "modes": {"code": {"status": status, "accepted": accepted}}, "path": name}


def test_metrics_fixed_coverage_and_errors(tmp_path):
    cases = [fixture_case("s:0", "p", True, True)] + [fixture_case("s:0", str(i), False, i == 0) for i in range(3)]
    cases.append(fixture_case("s:1", "error", True, None, "verifier_timeout"))
    m = metrics(cases, "code")
    assert m["positive_pass_rate"]["rate"] == 1
    assert m["negative_false_accept_rate"]["numerator"] == 1
    assert m["case_accuracy"]["denominator"] == 4
    assert m["covered_tasks"] == ["s:0"] and m["passed_tasks"] == []
    assert m["errors"]["verifier_timeout"] == 1
    save_json(tmp_path / "manifest.json", {"dataset": "fixture", "revision": "a" * 40, "seed": 20261021,
              "scenarios": ["s"], "tasks": [{"scenario": "s", "task_idx": i, "task": str(i)} for i in range(100)]})
    report = generate(tmp_path)
    assert report["fixed_task_denominator"] == 100
    assert report["task_states"] == {"pending": 100}


def test_frozen_asset_hash_and_duplicate_verifier(tmp_path):
    tasks = [{"scenario": f"s{i}", "tasks": [f"task {i}-{j}" for j in range(10)]} for i in range(1000)]
    source = tmp_path / "source";source.mkdir()
    for filename in FILES:
        if filename == "gen_tasks.jsonl":
            data = tasks
        elif "verifier" in filename:
            data = [{"scenario": r["scenario"], "task_idx": j, "task": task} for r in tasks for j, task in enumerate(r["tasks"])]
        else:
            data = [{"scenario": r["scenario"]} for r in tasks]
        (source / filename).write_text("".join(json.dumps(r) + "\n" for r in data))
    root = tmp_path / "run"
    manifest = prepare(root, source=source, revision="a" * 40)
    assert len(manifest["tasks"]) == 100
    assert validate_assets(root, manifest) == []
    p = root / "assets/gen_verifier.jsonl.json"
    entries = read_json(p);entries.append(entries[0]);save_json(p, entries)
    t = next(t for t in manifest["tasks"] if t["scenario"] == entries[0]["scenario"] and t["task_idx"] == entries[0]["task_idx"])
    with pytest.raises(ValueError, match="Frozen asset"):
        validate_assets(root, manifest)
    manifest["files"]["gen_verifier.jsonl"]["selected_sha256"] = digest(p)
    problems = validate_assets(root, manifest)
    assert any(x["error"] == "record_count=2" for x in problems)
    with pytest.raises(ValueError, match="record count=2"):
        verifier_entry(root, t, "sql")


def test_ecommerce_headphone_regression(tmp_path):
    """Outside the random sample: prove the known budget-order bug again."""
    from awm_audit.common import ROOT, rows
    from awm_audit.runtime import initialize
    source = ROOT / "examples/awm_e_commerce_33"
    fixture = tmp_path / "fixture"
    for name in ("gen_db.jsonl", "gen_sample.jsonl"):
        save_json(fixture / "assets" / (name + ".json"), rows(source / name))
    folder = tmp_path / "probe";folder.mkdir()
    initialize(fixture, "e_commerce_33", folder / "initial.db")
    backup(folder / "initial.db", folder / "final.db")
    with sqlite3.connect(folder / "final.db") as db:
        db.execute("INSERT INTO cart_items(cart_id,product_offer_id,quantity) VALUES(1,1,1)")
    save_json(folder / "trajectory.json", {"final_answer": "Added Sony WH-CH710N, quantity 1, price $149.99."})
    entry = rows(source / "gen_verifier.pure_code.jsonl")[0]
    result = execute_verifier(entry, "code", folder)
    assert result["status"] == "valid" and result["accepted"] is False
    # This test is explicitly a DB diagnostic, not an MCP or SQL+LLM success.


def test_date_only_and_three_valued_required_conditions(dbs):
    initial, final = dbs
    with sqlite3.connect(initial) as db:
        db.executescript("CREATE TABLE schedule(id INTEGER PRIMARY KEY,open_date TEXT,notes TEXT);INSERT INTO schedule VALUES(1,'2026-01-05 00:00:00','2026-01-05');")
    backup(initial, final)
    spec = TaskSpec([], [row_rule('schedule', {'id': 1, 'open_date': '2026-01-05', 'notes': '2026-01-05'})])
    assert evaluate(spec, initial, final, '')['correct'] is True
    with sqlite3.connect(final) as db:
        db.execute("UPDATE schedule SET open_date='2026-01-06 00:00:00'")
    assert evaluate(spec, initial, final, '')['correct'] is False
    conjunction = TaskSpec([], [Rule('missing metadata', 'SELECT NULL'), Rule('known violation', 'SELECT 0')])
    assert evaluate(conjunction, initial, final, '')['correct'] is False
    assert evaluate(TaskSpec([], [Rule('unknown', 'SELECT NULL'), Rule('yes', 'SELECT 1')]), initial, final, '')['status'] == 'unknown'


def test_reference_assembly_uses_actual_tool_fields_only(dbs):
    from awm_audit.reference_answers import compute
    results = [{'parts': [{'id': 4, 'part_number': 'A', 'short_description': 'real', 'list_price': 12},
                           {'id': 5, 'part_number': 'B', 'short_description': 'wrong position', 'list_price': 9}]},
               {'fitment_applications': [{'part_id': 4, 'position_id': 2}, {'part_id': 5, 'position_id': 3}]},
               {'stock_records': [{'part_id': 4, 'on_hand': 2}, {'part_id': 4, 'on_hand': 7}, {'part_id': 5, 'on_hand': 99}]}]
    answer = compute({'$compute': 'parts_stock', 'position_id': 2}, results)
    assert answer['parts'] == [{'part_number': 'A', 'short_description': 'real', 'list_price': 12, 'total_on_hand': 9}]
