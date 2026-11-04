"""D1 regressions from the development20 review; no semantic accuracy claims."""
import copy
import itertools
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rubric_candidates_100 as legacy
import rubric_d1 as d1
from harness_awm.rubric_judge import decide_item, validate_and_aggregate


def fixture():
    task = "Create a note titled Alpha. Preferably add a friendly summary."
    inp = {"task_id": "d1:0", "task": task, "quality_policy": None, "sources": [
        {"id": "task", "kind": "task", "text": task},
        {"id": "contract:notes", "kind": "public_contract", "text": json.dumps({"description": "Create a new note."})},
    ]}
    _, pack = legacy.prepare_input(inp)
    out = {"task_id": inp["task_id"], "selection": [], "items": [], "unresolved": []}
    for n, c in enumerate(pack["candidates"], 1):
        rid, cid = f"R{n:02d}", c["candidate_id"]
        out["selection"].append({"candidate_id": cid, "action": "keep", "item_ids": [rid], "reason": "题面要求或明确偏好"})
        out["items"].append({"id": rid, "candidate_ids": [cid], "contract_ref_ids": ["K0001"] if n == 1 else [],
            "requirement_level": "required" if n == 1 else "quality", "requirement": c["quote"],
            "gate": "可由动作及状态确认创建，或由已知目标标题证明错误；不要求所有属性齐备。",
            "predicate": "本次创建同一笔记并持久化，title=Alpha。" if n == 1 else "交付友好的摘要。",
            "failure_predicate": "完整动作与状态证实未创建；或本次目标笔记的已知标题不是Alpha。" if n == 1 else "完整最终答复没有摘要。",
            "evidence": [{"source_id": None, "description": "完整动作、返回身份、笔记初末态及最终答复。"}], "method": "hybrid"})
    rubric = d1.compile_rubric(out, inp, pack)
    evidence = {"facts": {"valid": True, "content": "test-only facts"},
                "failed_query": {"valid": False, "error": "missing table", "content": None}}
    return inp, pack, out, rubric, evidence


def response(rubric, states, sample="valid"):
    def proof(flag):
        return {"established": flag, "evidence_ids": ["facts"] if flag else [], "reason": "具体证明或未证明原因"}
    return {"task_id": rubric["task_id"], "items": [
        {"id": row["id"], "satisfied": proof(state == "pass"), "violated": proof(state == "fail"),
         "unknown_reason": "目标年份缺失，其他证据不能决定。" if state == "unknown" else ""}
        for row, state in zip(rubric["rubric"], states)],
        "sample_validity": {"status": sample, "evidence_ids": ["facts"], "reason": "已核对执行前提"}, "rubric_issues": []}


@pytest.mark.parametrize("satisfied,violated,expected", [(False, False, "unknown"), (True, False, "pass"), (False, True, "fail")])
def test_item_truth_table(satisfied, violated, expected):
    assert decide_item(satisfied=satisfied, violated=violated) == expected


@pytest.mark.parametrize("states", list(itertools.product(["pass", "fail", "unknown"], repeat=2)))
def test_required_failure_dominates_other_unknown(states):
    _, _, _, rubric, ev = fixture()
    rubric["rubric"][1]["requirement_level"] = "required"
    result = validate_and_aggregate(response(rubric, states), rubric, ev)
    expected = "fail" if "fail" in states else "pass" if states == ("pass", "pass") else "unknown"
    assert result["label"] == expected
    assert result["reward"] == ({"pass": 1, "fail": 0, "unknown": None}[expected])


@pytest.mark.parametrize("quality", ["pass", "fail", "unknown"])
def test_quality_never_blocks_required_pass(quality):
    _, _, _, rubric, ev = fixture()
    result = validate_and_aggregate(response(rubric, ["pass", quality]), rubric, ev)
    assert result["label"] == "pass"
    assert result["quality_statuses"] == {"R02": quality}


def test_no_creation_with_unknown_timezone_can_prove_same_item_failure():
    _, _, _, rubric, ev = fixture()
    value = response(rubric, ["fail", "unknown"])
    value["items"][0]["violated"]["reason"] = "完整动作及初末态证明根本没有创建，时区歧义不能改变这一违例。"
    result = validate_and_aggregate(value, rubric, ev)
    assert result["label"] == "fail"


@pytest.mark.parametrize("sample,label", [("invalid", "invalid"), ("unknown", "unknown")])
def test_bad_sample_never_becomes_training_reward(sample, label):
    _, _, _, rubric, ev = fixture()
    result = validate_and_aggregate(response(rubric, ["fail", "pass"], sample), rubric, ev)
    assert result["business_label"] == "fail"
    assert result["label"] == label
    assert not result["reward_valid"] and result["reward"] is None


@pytest.mark.parametrize("fault", ["duplicate", "omit", "change_level", "total", "fake_evidence", "failed_evidence",
                                   "mixed_failed_evidence", "missing_evidence", "both_true", "boolean_number", "unknown_without_gap"])
def test_malformed_or_unsupported_proof_is_rejected(fault):
    _, _, _, rubric, ev = fixture()
    value = response(rubric, ["pass", "unknown"])
    row = value["items"][0]
    if fault == "duplicate": value["items"][1] = copy.deepcopy(row)
    if fault == "omit": value["items"].pop()
    if fault == "change_level": row["requirement_level"] = "quality"
    if fault == "total": value["task_label"] = "pass"
    if fault == "fake_evidence": row["satisfied"]["evidence_ids"] = ["invented"]
    if fault == "failed_evidence": row["satisfied"]["evidence_ids"] = ["failed_query"]
    if fault == "mixed_failed_evidence": row["satisfied"]["evidence_ids"] = ["facts", "failed_query"]
    if fault == "missing_evidence": row["satisfied"]["evidence_ids"] = []
    if fault == "both_true": row["violated"] = copy.deepcopy(row["satisfied"])
    if fault == "boolean_number": row["satisfied"]["established"] = 1
    if fault == "unknown_without_gap": value["items"][1]["unknown_reason"] = " "
    with pytest.raises(ValueError): validate_and_aggregate(value, rubric, ev)


def test_partial_evidence_can_prove_present_fact_but_query_error_cannot():
    _, _, _, rubric, ev = fixture()
    ev["facts"]["truncated"] = True
    # Completeness depends on the claim, not a blanket all-evidence gate.
    assert validate_and_aggregate(response(rubric, ["fail", "unknown"]), rubric, ev)["label"] == "fail"
    ev["facts"]["error"] = "query failed"
    with pytest.raises(ValueError): validate_and_aggregate(response(rubric, ["fail", "unknown"]), rubric, ev)


def test_rubric_defect_is_separate_from_business_label():
    _, _, _, rubric, ev = fixture()
    value = response(rubric, ["pass", "pass"])
    value["rubric_issues"] = ["原题义务遗漏，需要按来源修订。"]
    result = validate_and_aggregate(value, rubric, ev)
    assert result["business_label"] == "pass" and result["label"] == "invalid"
    assert result["reward"] is None


def test_no_required_items_is_not_vacuous_pass():
    _, _, _, rubric, ev = fixture()
    for row in rubric["rubric"]: row["requirement_level"] = "quality"
    assert validate_and_aggregate(response(rubric, ["pass", "pass"]), rubric, ev)["label"] == "unknown"


def test_compiler_retains_sources_and_old_compiler_semantics():
    inp, pack, out, rubric, _ = fixture()
    original = copy.deepcopy(out)
    compiled = d1.compile_rubric(out, inp, pack)
    assert out == original
    old = legacy.compile_rubric(d1.legacy_projection(out), inp, pack)
    assert old["rubric"][0]["decision"]["unknown_when"] == "本条G不成立"
    for before, after in zip(old["rubric"], compiled["rubric"]):
        assert before["sources"] == after["sources"]
        assert set(before) == set(after)
        assert after["criterion"].endswith(out["items"][int(after["id"][1:]) - 1]["failure_predicate"])
    with pytest.raises(ValueError): validate_and_aggregate(response(old, ["pass", "pass"]), old, {"e": {"valid": True, "content": 1}})


@pytest.mark.parametrize("fault", ["missing_failure", "blank_failure", "missing_trigger", "missing_resolution", "bad_mapping", "tampered_quote"])
def test_compilation_requires_failure_route_and_scoped_unresolved(fault):
    inp, pack, out, _, _ = fixture()
    if fault == "missing_failure": out["items"][0].pop("failure_predicate")
    if fault == "blank_failure": out["items"][0]["failure_predicate"] = " "
    if fault in {"missing_trigger", "missing_resolution"}:
        u = {"issue": "同名候选", "affected_items": ["R01"], "needed": "候选范围", "trigger": "实际多名且结果不同", "resolution": "查询范围内所有候选"}
        u.pop(fault.removeprefix("missing_"))
        out["unresolved"] = [u]
    if fault == "bad_mapping": out["selection"][0]["item_ids"] = ["R02"]
    if fault == "tampered_quote": pack["reference_registry"]["C01"]["start"] = 1
    with pytest.raises(ValueError): d1.compile_rubric(out, inp, pack)


def test_requests_include_rubric_levels_policy_and_original_task():
    inp, _, out, rubric, ev = fixture()
    bundle = {"original": inp, "rubric": rubric, "trajectory": [], "final_answer": "", "evidence": ev}
    built = d1.runtime_request(bundle)
    payload = json.loads(built["messages"][1]["content"])
    assert payload["rubric"] == rubric and payload["original"]["task"] == inp["task"]
    assert "D1" in built["messages"][0]["content"]
    generated = json.loads(d1.generation_request(inp)["messages"][1]["content"])
    assert "failure_predicate" in generated["output_schema"]["properties"]["items"]["items"]["required"]
    assert json.loads(d1.review_request(inp, out)["messages"][1]["content"])["rubric"] == rubric


def test_cli_round_trip_and_exclusive_output(tmp_path):
    inp, _, out, rubric, ev = fixture()
    files = {"input": inp, "selection": out, "response": response(rubric, ["pass", "unknown"]),
             "bundle": {"original": inp, "rubric": rubric, "trajectory": [], "final_answer": "", "evidence": ev}}
    for key, value in files.items(): (tmp_path / f"{key}.json").write_text(json.dumps(value))
    for command, args in [("prepare", ["--input", "input.json"]),
                          ("compile", ["--input", "input.json", "--selection", "selection.json"]),
                          ("judge-request", ["--bundle", "bundle.json"]),
                          ("judge-validate", ["--bundle", "bundle.json", "--response", "response.json"])]:
        cmd = [sys.executable, str(ROOT / "scripts/rubric_d1.py"), command, *args, "--output", command + ".json"]
        subprocess.run(cmd, cwd=tmp_path, check=True, capture_output=True)
    result = json.loads((tmp_path / "judge-validate.json").read_text())
    assert result["label"] == "pass" and result["reward"] == 1
    before = (tmp_path / "judge-validate.json").read_bytes()
    assert subprocess.run(cmd, cwd=tmp_path, capture_output=True).returncode != 0
    assert (tmp_path / "judge-validate.json").read_bytes() == before


def test_duplicate_json_keys_are_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"established":false,"established":true}')
    with pytest.raises(ValueError): d1.load(path)
