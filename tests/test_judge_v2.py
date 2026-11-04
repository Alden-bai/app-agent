import asyncio
import copy
import json
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from harness_awm.backend import Budget
from harness_awm.contracts import RunConfig
from harness_awm.evaluation import apply_evaluation, evaluate_episode
from harness_awm.io import canonical, digest, file_hash, read_json, seal, write_json
from harness_awm.judge_evidence import EvidenceStore, model_view
from harness_awm.judge_inputs import prepare_input, strict_json, verify_input
from harness_awm.judge_report import summarize
from harness_awm.judge_runtime import evaluate_input
from harness_awm.judge_v2 import aggregate, normalize_rubric, validate_assessment
from harness_support import fixture as task_fixture


def rubric_fixture():
    task = "Set value to 80 then report it."
    original = {"task_id": "harness_fixture:0", "task": task, "sources": [{"id": "task", "kind": "task", "text": task}]}
    rubric = {"task_id": original["task_id"], "rubric": [
        {"id": "R1", "requirement_level": "required", "criterion": "要求：Set value to 80\nG：Read value\nP：value=80\nF：value!=80",
         "sources": [{"source_id": "task", "quote": "Set value to 80"}]},
        {"id": "Q1", "requirement_level": "quality", "criterion": "要求：report it\nG：Read answer\nP：clear answer",
         "sources": [{"source_id": "task", "quote": "report it"}]}], "coverage": [], "unresolved": []}
    return original, rubric


def assessment(states=("pass", "pass"), *, refs=None, kind="semantic_ambiguity", sample="valid"):
    refs = ["fact"] if refs is None else refs
    def proof(flag):
        return {"established": flag, "evidence_ids": refs if flag else [], "reason": "Specific source-bound evidence."}
    return {"action": "finish", "task_id": "harness_fixture:0", "items": [
        {"id": rid, "satisfied": proof(state == "pass"), "violated": proof(state == "fail"),
         "unknown": {"kind": kind, "missing_fact": "Target year", "impact": "Different years change date correctness",
                     "checked_evidence_ids": refs, "next_action": "External clarification", "can_resolve_with_tools": False} if state == "unknown" else None}
        for rid, state in zip(("R1", "Q1"), states)],
        "sample_validity": {"status": sample, "evidence_ids": refs, "reason": "Recorded execution premises checked"},
        "rubric_issues": [], "unresolved_checks": []}


def validated(value, rubric=None, evidence=None):
    original, default = rubric_fixture()
    return validate_assessment(value, normalize_rubric(original, rubric or default),
                               evidence or {"fact": {"kind": "query", "valid": True, "content": {"value": 80}}})


@pytest.mark.parametrize("soft", ["pass", "fail", "unknown"])
def test_soft_status_never_blocks_hard_completion(soft):
    result = aggregate(validated(assessment(("pass", soft))))
    assert result["task_status"] == "confirmed_pass"
    assert result["quality_counts"][soft] == 1


@pytest.mark.parametrize("kind,status", [("semantic_ambiguity", "valid"), ("evidence_gap", "valid"),
    ("evidence_conflict", "valid"), ("evidence_error", "evaluation_error"),
    ("budget_exhausted", "evaluation_error"), ("not_investigated", "evaluation_error")])
def test_unknown_kinds_are_not_equivalent(kind, status):
    result = aggregate(validated(assessment(("unknown", "pass"), kind=kind)))
    assert result["evaluation_status"] == status
    assert result["task_status"] == ("provisional_pass" if status == "valid" else "undetermined")
    assert result["business_label"] == "unknown"


def test_independent_failure_dominates_other_required_unknown():
    _, rubric = rubric_fixture()
    rubric["rubric"][1]["requirement_level"] = "required"
    result = aggregate(validated(assessment(("fail", "unknown"), kind="evidence_error"), rubric))
    assert result["task_status"] == "failed" and result["evaluation_valid"]


@pytest.mark.parametrize("sample,status", [("invalid", "invalid_sample"), ("unknown", "evaluation_error")])
def test_sample_validity_is_distinct_from_business(sample, status):
    result = aggregate(validated(assessment(("fail", "pass"), sample=sample)))
    assert result["business_label"] == "fail" and result["evaluation_status"] == status


def test_rubric_issue_scope_cannot_demote_a_required_issue():
    value = assessment()
    value["rubric_issues"] = [{"scope": "quality", "affected_items": ["Q1"], "reason": "Unsupported quality constraint", "source_ids": ["task"]}]
    assert aggregate(validated(value))["evaluation_status"] == "valid"
    value["rubric_issues"][0]["affected_items"] = ["R1"]
    assert aggregate(validated(value))["evaluation_status"] == "rubric_error"


@pytest.mark.parametrize("fault", ["duplicate", "missing", "unknown_ref", "no_ref", "contradiction", "not_bool", "empty_reason", "unknown_missing", "unexpected_total"])
def test_invalid_assessments_are_rejected(fault):
    value = assessment()
    if fault == "duplicate": value["items"][1] = copy.deepcopy(value["items"][0])
    if fault == "missing": value["items"].pop()
    if fault == "unknown_ref": value["items"][0]["satisfied"]["evidence_ids"] = ["invented"]
    if fault == "no_ref": value["items"][0]["satisfied"]["evidence_ids"] = []
    if fault == "contradiction": value["items"][0]["violated"] = copy.deepcopy(value["items"][0]["satisfied"])
    if fault == "not_bool": value["items"][0]["satisfied"]["established"] = 1
    if fault == "empty_reason": value["items"][0]["satisfied"]["reason"] = " "
    if fault == "unknown_missing": value["items"][0]["satisfied"]["established"] = False
    if fault == "unexpected_total": value["reward"] = 1
    with pytest.raises(ValueError): validated(value)


def test_failed_queries_and_task_text_cannot_prove_success():
    for ev in [{"kind": "query", "valid": False, "content": None, "error": "bad column"},
               {"kind": "task", "valid": True, "content": "expected result"}]:
        with pytest.raises(ValueError): validated(assessment(), evidence={"fact": ev})
    assert validated(assessment(), evidence={"fact": {"kind": "query", "valid": True, "truncated": True, "content": [[80]]}})


def test_legacy_adapter_preserves_predicates_and_sources():
    original, rubric = rubric_fixture()
    before = copy.deepcopy(rubric)
    normalized = normalize_rubric(original, rubric)
    assert normalized["items"][1]["success_condition"] == "clear answer"
    assert normalized["items"][1]["failure_origin"] == "independent_counterexample_to_P"
    assert rubric == before and normalized["rubric_hash"] == digest(rubric)
    rubric["rubric"][0]["sources"][0]["quote"] = "fabricated"
    with pytest.raises(ValueError): normalize_rubric(original, rubric)


def test_no_required_and_duplicate_sources_rejected():
    original, rubric = rubric_fixture()
    rubric["rubric"][0]["requirement_level"] = "quality"
    with pytest.raises(ValueError): normalize_rubric(original, rubric)
    original["sources"].append(original["sources"][0])
    with pytest.raises(ValueError): normalize_rubric(original, rubric_fixture()[1])


def test_unresolved_must_be_accounted_for_without_changing_scope():
    original, rubric = rubric_fixture()
    rubric["unresolved"] = [{"issue": "year", "affected_items": ["R1"], "needed": "year interpretation"}]
    normal = normalize_rubric(original, rubric)
    evidence = {"fact": {"kind": "query", "valid": True, "content": 1}}
    value = assessment()
    with pytest.raises(ValueError): validate_assessment(value, normal, evidence)
    value["unresolved_checks"] = [{"id": "U001", "status": "not_triggered", "affected_items": ["R1"], "evidence_ids": ["fact"], "reason": "Year does not affect this execution"}]
    assert aggregate(validate_assessment(value, normal, evidence))["task_status"] == "confirmed_pass"
    value["unresolved_checks"][0]["affected_items"] = ["Q1"]
    with pytest.raises(ValueError): validate_assessment(value, normal, evidence)


@pytest.fixture
def snapshots(tmp_path):
    paths = []
    for name, value in (("initial", 10), ("final", 80)):
        path = tmp_path / f"{name}.db"
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE items(id INTEGER PRIMARY KEY, value INTEGER, text TEXT)')
            db.executemany('INSERT INTO items VALUES(?,?,?)', [(i, value, "x" * 200) for i in range(110)])
            db.execute('CREATE TABLE unkeyed(value INTEGER)')
            db.execute('CREATE TABLE composite(a TEXT, b TEXT, UNIQUE(a,b))')
            db.execute("INSERT INTO composite VALUES('A','B')")
        paths.append(path)
    return paths


def evidence_call(store, sql, snapshot="final"):
    return store.call("query_snapshot", {"snapshot": snapshot, "sql": sql}, objective="object", affected_items=["R1"])


@pytest.mark.parametrize("sql", ["DELETE FROM items", "UPDATE items SET value=1", "ATTACH DATABASE ':memory:' AS other", "PRAGMA user_version=2", "SELECT load_extension('oops')", "SELECT 1; SELECT 2", "VACUUM INTO '/tmp/judge_escape.db'"])
def test_sql_read_only(snapshots, sql):
    store = EvidenceStore(*snapshots)
    assert not evidence_call(store, sql)["valid"]
    store.verify()


def test_query_states_params_and_real_schema(snapshots):
    store = EvidenceStore(*snapshots)
    initial = evidence_call(store, "SELECT value FROM items WHERE id=0", "initial")
    final = store.call("query_snapshot", {"snapshot": "final", "sql": "SELECT value FROM items WHERE id=?", "params": [0]}, objective="x", affected_items=["R1"])
    assert initial["content"]["rows"] == [[10]] and final["content"]["rows"] == [[80]]
    missing = evidence_call(store, "SELECT * FROM items WHERE id=-1")
    error = evidence_call(store, "SELECT no_column FROM items")
    assert missing["valid"] and missing["content"]["rows"] == [] and not missing["truncated"]
    assert not error["valid"] and error["content"] is None
    full = evidence_call(store, "SELECT * FROM items")
    assert full["truncated"] and len(full["content"]["rows"]) == 100
    schema, truncated = store.inspect_schema("final", "items")
    assert schema["columns"][0]["pk"] == 1 and not truncated


def test_bounded_bytes_timeout_and_object_comparison(snapshots):
    store = EvidenceStore(*snapshots, max_bytes=1000, seconds=0.01)
    row = evidence_call(store, "SELECT * FROM items")
    assert row["truncated"] and len(canonical(row["content"]).encode()) <= 1000
    row = evidence_call(store, "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n) SELECT sum(x) FROM n")
    assert not row["valid"] and "interrupted" in row["error"]
    assert not evidence_call(store, "SELECT zeroblob(2000000)")["valid"]
    diff, truncated = store.compare_records("items", {"id": 0})
    assert diff["changed"] and not truncated
    assert not store.compare_records("composite", {"a": "A", "b": "B"})[0]["changed"]
    with pytest.raises(ValueError): store.compare_records("unkeyed", {"value": 1})
    with pytest.raises(ValueError): store.compare_records("composite", {"a": "A"})


def test_full_evidence_is_recoverable_after_preview_truncation(snapshots):
    store = EvidenceStore(*snapshots)
    value = {"body": "中文" * 6000, "sentinel": "end"}
    row = store.add("tool_result", value)
    view = model_view(store.records)
    assert view[row["id"]]["view_truncated"]
    offset, chunks = 0, []
    while offset is not None:
        page, truncated = store.read_evidence(row["id"], offset)
        chunks.append(page["text"])
        offset = page["next_offset"]
        assert truncated
    assert json.loads("".join(chunks)) == value
    bad = store.add("query_error", None, valid=False, error="SQL failed")
    with pytest.raises(ValueError): store.read_evidence(bad["id"])


def test_material_unresolved_and_all_pass_is_rejected():
    original, rubric = rubric_fixture()
    rubric["unresolved"] = [{"issue": "year", "affected_items": ["R1"], "needed": "target year"}]
    answer = assessment()
    answer["unresolved_checks"] = [{"id": "U001", "status": "material", "affected_items": ["R1"], "evidence_ids": ["fact"], "reason": "Year changes correctness"}]
    with pytest.raises(ValueError): validated(answer, rubric)


@pytest.fixture
def bundle(tmp_path):
    task = task_fixture(tmp_path / "task")
    episode = tmp_path / "episode"
    episode.mkdir()
    for name, value in (("initial", 10), ("final", 80)):
        with sqlite3.connect(episode / f"{name}.db") as db:
            db.execute("CREATE TABLE items(id INTEGER PRIMARY KEY, value INTEGER)")
            db.execute("INSERT INTO items VALUES(1,?)", (value,))
    write_json(episode / "task.json", task.model_dump())
    write_json(episode / "trajectory.json", {"task": task.instruction, "finished": True, "final_answer": "80",
               "trajectory": [{"iteration": 1, "is_final": True, "content": "80"}]})
    write_json(episode / "episode_result.json", {"task_instance_id": task.task_instance_id, "rollout_id": "fixture",
        "attempt_id": "a0", "execution_integrity": "valid", "terminal_state_status": "trusted", "termination_reason": "finish", "test_only": True})
    (episode / "events.jsonl").write_text(canonical({"event_type": "execution_closed", "time": "2026-01-01T00:00:00Z"}) + "\n")
    seal(episode)
    original, rubric = rubric_fixture()
    return prepare_input(episode, original, rubric)


def test_input_binding_and_seal(bundle):
    assert verify_input(bundle) == bundle
    bad = copy.deepcopy(bundle)
    bad["hashes"]["runtime"] = "tampered"
    with pytest.raises(ValueError): verify_input(bad)
    write_json(Path(bundle["episode"]) / "evidence_seal.json", {})
    with pytest.raises(ValueError): verify_input(bundle)


def test_duplicate_json_and_non_finite_numbers_rejected():
    for value in ['{"a":1,"a":2}', '{"a":NaN}']:
        with pytest.raises(ValueError): strict_json(value)


class Responder:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = 0

    async def __call__(self, config, request, budget, key, stage, folder):
        rid = budget.reserve(key, stage, request["messages"], request["max_completion_tokens"])
        self.calls += 1
        answer = next(self.answers)
        if isinstance(answer, Exception):
            budget.finish(rid)
            raise answer
        if callable(answer): answer = answer(request)
        raw = {"choices": [{"finish_reason": "stop", "message": {"content": answer if isinstance(answer, str) else canonical(answer)}}], "usage": {"total_tokens": 100}}
        budget.finish(rid, raw["usage"])
        write_json(Path(folder) / "request.json", request)
        write_json(Path(folder) / "response.json", raw)
        return raw


def run_judge(bundle, tmp_path, monkeypatch, answers, *, config=None, budget_requests=40):
    responder = Responder(answers)
    monkeypatch.setattr("harness_awm.judge_runtime.api_request", responder)
    config = config or RunConfig(judge="api", judge_protocol="rubric_v2", judge_v2={"review_all_decisions": False})
    budget = Budget(tmp_path / "budget.sqlite", budget_requests, 2_000_000)
    result = asyncio.run(evaluate_input(bundle, config, budget, asyncio.Semaphore(1), tmp_path / "results"))
    return result, responder


def query(sql="SELECT value FROM items WHERE id=1", objective="value"):
    return {"action": "query", "queries": [{"tool": "query_snapshot", "objective": objective, "affected_items": ["R1"],
            "arguments": {"snapshot": "final", "sql": sql}}]}


def test_frozen_replay_caches_first_valid_and_never_calculates_reward(bundle, tmp_path, monkeypatch):
    before = {str(p): file_hash(p) for p in Path(bundle["episode"]).rglob("*") if p.is_file()}
    result, responder = run_judge(bundle, tmp_path, monkeypatch, [assessment(refs=["E00002"])])
    assert result["task_status"] == "confirmed_pass", result
    assert result["reward"] is None and result["reward_valid"] is False
    assert responder.calls == 1
    again, responder = run_judge(bundle, tmp_path, monkeypatch, [])
    assert again == result and responder.calls == 0
    assert before == {str(p): file_hash(p) for p in Path(bundle["episode"]).rglob("*") if p.is_file()}


def test_unknown_review_is_mandatory_and_preserves_true_uncertainty(bundle, tmp_path, monkeypatch):
    answer = assessment(("unknown", "pass"), refs=["E00002"])
    result, responder = run_judge(bundle, tmp_path, monkeypatch, [answer, answer])
    assert result["task_status"] == "provisional_pass" and result["business_label"] == "unknown", result
    assert responder.calls == 2 and (Path(result["attempt"]) / "review_assessment.json").exists()


def test_query_recovery_and_failed_evidence_history(bundle, tmp_path, monkeypatch):
    result, _ = run_judge(bundle, tmp_path, monkeypatch, [query("SELECT bad FROM items"), query(), assessment(refs=["E00008"])])
    assert result["task_status"] == "confirmed_pass", result
    records = read_json(Path(result["attempt"]) / "evidence.json")
    assert not records["E00007"]["valid"] and records["E00008"]["valid"]
    assert records["E00007"]["arguments"]["sql"] == "SELECT bad FROM items"


def test_unrelated_query_failure_does_not_block_sufficient_evidence(bundle, tmp_path, monkeypatch):
    result, _ = run_judge(bundle, tmp_path, monkeypatch, [query("SELECT bad FROM items"), assessment(refs=["E00002"])])
    assert result["evaluation_status"] == "valid", result


def test_still_failed_query_is_not_discountable_uncertainty(bundle, tmp_path, monkeypatch):
    answer = assessment(("unknown", "pass"), refs=["E00002"])
    answer["items"][0]["unknown"]["checked_evidence_ids"] = ["E00007"]
    result, _ = run_judge(bundle, tmp_path, monkeypatch, [query("SELECT bad FROM items"), answer, answer])
    assert result["evaluation_status"] == "evaluation_error", result


def test_unknown_can_be_resolved_in_review(bundle, tmp_path, monkeypatch):
    answer = assessment(("unknown", "pass"), refs=["E00002"])
    answer["items"][0]["unknown"]["can_resolve_with_tools"] = True
    result, responder = run_judge(bundle, tmp_path, monkeypatch, [answer, query(), assessment(refs=["E00007"])])
    assert result["task_status"] == "confirmed_pass" and responder.calls == 3, result


@pytest.mark.parametrize("answers,valid", [(['bad json', assessment(refs=["E00002"])], True), (['bad json', 'bad json'], False)])
def test_only_one_output_repair(bundle, tmp_path, monkeypatch, answers, valid):
    result, responder = run_judge(bundle, tmp_path, monkeypatch, answers)
    assert result["evaluation_valid"] == valid and responder.calls == 2, result


def test_review_failure_and_budget_exhaustion_remain_errors(bundle, tmp_path, monkeypatch):
    answer = assessment(("unknown", "pass"), refs=["E00002"])
    result, responder = run_judge(bundle, tmp_path, monkeypatch, [answer], budget_requests=1)
    assert result["evaluation_status"] == "evaluation_error" and result["business_label"] == "unknown", result
    assert responder.calls == 1 and "BudgetExceeded" in result["errors"][0]


def test_query_round_exhaustion_cannot_become_provisional(bundle, tmp_path, monkeypatch):
    config = RunConfig(judge="api", judge_protocol="rubric_v2", judge_v2={"query_rounds": 0})
    answer = assessment(("unknown", "pass"), refs=["E00002"])
    result, _ = run_judge(bundle, tmp_path, monkeypatch, [query(), answer], config=config)
    assert result["evaluation_status"] == "evaluation_error", result
    assert "query_round_budget_exhausted" in result["errors"]


def test_failed_objective_cannot_retry_forever(bundle, tmp_path, monkeypatch):
    answer = assessment(("unknown", "pass"), refs=["E00002"], kind="evidence_error")
    result, _ = run_judge(bundle, tmp_path, monkeypatch, [query("SELECT bad FROM items")] * 4 + [answer, answer])
    assert result["evaluation_status"] == "evaluation_error"
    records = read_json(Path(result["attempt"]) / "evidence.json")
    assert records["E00010"]["error"] == "objective_query_repair_limit"


def test_valid_failure_stops_before_unknown_review(bundle, tmp_path, monkeypatch):
    bundle = copy.deepcopy(bundle)
    bundle["rubric"]["rubric"][1]["requirement_level"] = "required"
    bundle = prepare_input(bundle["episode"], bundle["original"], bundle["rubric"])
    result, responder = run_judge(bundle, tmp_path, monkeypatch, [assessment(("fail", "unknown"), refs=["E00002"], kind="not_investigated")])
    assert result["task_status"] == "failed" and responder.calls == 1


def test_invalid_sample_is_cached_without_selecting_a_new_judgment(bundle, tmp_path, monkeypatch):
    result, _ = run_judge(bundle, tmp_path, monkeypatch, [assessment(refs=["E00002"], sample="invalid")])
    assert result["evaluation_status"] == "invalid_sample"
    again, responder = run_judge(bundle, tmp_path, monkeypatch, [])
    assert again == result and responder.calls == 0


def test_v2_dispatch_bypasses_official_verifier_and_updates_episode(bundle, tmp_path, monkeypatch):
    task = task_fixture(tmp_path / "task2")
    for name in ("original", "rubric"):
        write_json(tmp_path / f"{name}.json", bundle[name])
    manifest = {"schema_version": "awm-judge-input-manifest-v2", "tasks": {task.task_instance_id: {
        name: {"path": f"{name}.json", "sha256": file_hash(tmp_path / f"{name}.json")} for name in ("original", "rubric")}}}
    write_json(tmp_path / "inputs.json", manifest)
    config = RunConfig(judge="api", judge_protocol="rubric_v2", judge_inputs_manifest=str(tmp_path / "inputs.json"))
    monkeypatch.setattr("harness_awm.evaluation.worker", AsyncMock(side_effect=AssertionError("official verifier must not run")))
    responder = Responder([assessment(refs=["E00002"]), assessment(refs=["E00002"])])
    monkeypatch.setattr("harness_awm.judge_runtime.api_request", responder)
    result = asyncio.run(evaluate_episode(bundle["episode"], task, config, Budget(tmp_path / "b.db", 5, 100000), asyncio.Semaphore(1)))
    assert result["task_status"] == "confirmed_pass", result
    applied = apply_evaluation(bundle["episode"], result)
    assert applied["judge_protocol"] == "rubric_v2" and applied["reward"] is None
    assert verify_input(bundle) == bundle  # mutable evaluation fields aren't execution evidence


def test_report_separates_provisional_from_confirmed(bundle, tmp_path, monkeypatch):
    answer = assessment(("unknown", "pass"), refs=["E00002"])
    result, _ = run_judge(bundle, tmp_path, monkeypatch, [answer, answer])
    report = summarize([result], {result["task_id"]: "unknown"})
    assert report["confirmed_passes"] == 0 and report["provisional_passes"] == 1
    assert report["independent_accuracy_measured"] is False


def test_v2_metrics_do_not_use_reward_for_success(tmp_path):
    from harness_awm.stages import metrics
    candidates = [{"path": f"episodes/{i}", "task_instance_id": str(i), "scenario": "fixture"} for i in range(2)]
    write_json(tmp_path / "manifest.json", {"candidates": candidates})
    for candidate, status in zip(candidates, ("confirmed_pass", "provisional_pass")):
        write_json(tmp_path / candidate["path"] / "episode_result.json", {
            "judge_protocol": "rubric_v2", "evaluation_valid": True, "evaluation_status": "valid", "task_status": status,
            "reward": None, "reward_valid": False, "termination_reason": "finish", "lifecycle": "SETTLED"})
    result = metrics(tmp_path)
    assert result["valid_settlements"] == 2 and result["valid_successes"] == 1
    assert result["judge_v2"]["task_statuses"]["provisional_pass"] == 1


def test_cli_prepares_and_reports_without_model_calls(bundle, tmp_path):
    import subprocess
    import sys
    original, rubric = rubric_fixture()
    for name, value in (("original", original), ("rubric", rubric)):
        write_json(tmp_path / f"{name}.json", value)
    output = tmp_path / "cli_input.json"
    command = [sys.executable, "-m", "harness_awm.cli", "judge-prepare", "--episode", bundle["episode"],
               "--original", str(tmp_path / "original.json"), "--rubric", str(tmp_path / "rubric.json"), "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True)
    assert read_json(output) == bundle
    assert subprocess.run(command, capture_output=True).returncode != 0


def test_default_proof_review_can_reject_false_failure(bundle, tmp_path, monkeypatch):
    config = RunConfig(judge='api', judge_protocol='rubric_v2')
    result, responder = run_judge(bundle, tmp_path, monkeypatch,
        [assessment(('fail', 'pass'), refs=['E00002']), assessment(refs=['E00002'])], config=config)
    assert result['task_status'] == 'confirmed_pass' and responder.calls == 2
    assert read_json(Path(result['attempt']) / 'assessment.json')['items'][0]['status'] == 'fail'
    assert read_json(Path(result['attempt']) / 'review_assessment.json')['items'][0]['status'] == 'pass'


def test_spent_review_batch_cannot_hide_unfinished_evidence(bundle, tmp_path, monkeypatch):
    answer = assessment(('unknown', 'pass'), refs=['E00002'], kind='evidence_gap')
    result, _ = run_judge(bundle, tmp_path, monkeypatch, [answer, query(), answer])
    assert result['task_status'] == 'undetermined'
    assert result['items'][0]['unknown']['kind'] == 'budget_exhausted'
    assert 'query_round_budget_exhausted' in result['errors']


def test_exhaustion_does_not_override_genuine_year_ambiguity(bundle, tmp_path, monkeypatch):
    answer = assessment(('unknown', 'pass'), refs=['E00002'], kind='semantic_ambiguity')
    result, _ = run_judge(bundle, tmp_path, monkeypatch, [answer, query(), answer])
    assert result['task_status'] == 'provisional_pass'
