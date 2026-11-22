"""CPU regressions for real Judge failure modes, without model substitutes as scores."""
import asyncio
import copy
import json

import pytest

from harness_awm import trajectory_quality as runtime
from harness_awm.backend import Budget
from harness_awm.contracts import RunConfig
from harness_awm.io import canonical, read_json, write_json
from harness_awm.protocol_v2 import fits_context


FIRST = {"business_label": "pass", "task_status": "confirmed_pass"}


def assessment(turn="0001", mid="M1", grade=0):
    return {"status": "valid", "missing_reason": None, "dimensions": [
        {"dimension": name, "grade": grade if name == "information" else 0,
         "reason": "Checked the actual visible evidence.",
         "defect_id": "repeated-known-result" if grade and name == "information" else None,
         "action_quotes": [{"turn": turn, "quote": "repeat"}] if grade and name == "information" else [],
         "observations": [{"turn": turn, "message_id": mid, "quote": "Already returned"}]
             if grade and name == "information" else []}
        for name in runtime.DIMENSIONS]}


def context():
    return {"messages": {"M1": {"role": "user", "content": "Already returned"},
                         "M2": {"role": "user", "content": "future observation"}},
            "turns": [{"turn": "0001", "input_message_ids": ["M1"], "output": "repeat query"}]}


def test_nested_json_quotes_are_resolved_to_actual_substrings_only():
    ctx = context()
    inner = '{"file_name": "Transcript_UofT.pdf"}'
    ctx["messages"]["M1"]["content"] = canonical({"content": {"text": inner}})
    value = assessment(grade=.5)
    value["dimensions"][0]["observations"][0]["quote"] = '"file_name": "Transcript_UofT.pdf"'
    before = copy.deepcopy(value)
    result = runtime.validate_quality(value, ctx)
    actual = result["dimensions"][0]["observations"][0]["quote"]
    assert actual in ctx["messages"]["M1"]["content"]
    assert value == before
    assert runtime.validate_quality(result, ctx) == result
    for wrong in ('"file_name": "Other.pdf"', '"file_name":"Transcript_UofT.pdf"'):
        value["dimensions"][0]["observations"][0]["quote"] = wrong
        with pytest.raises(ValueError, match="invalid_quality_observation_quote"):
            runtime.validate_quality(value, ctx)


def test_all_visibility_errors_reported_and_not_relocated():
    ctx, value = context(), assessment(grade=1)
    observations = value["dimensions"][0]["observations"]
    observations[0].update(message_id="M2", quote="future observation")
    observations.append({"turn": "0001", "message_id": "M99", "quote": "invented"})
    with pytest.raises(runtime.QualityCitationError) as exc:
        runtime.validate_quality(value, ctx)
    assert len(exc.value.issues) == 2
    assert all(i["allowed_observation_ids"] == ["M1"] for i in exc.value.issues)
    assert observations[0]["message_id"] == "M2"


def test_ledger_and_system_rules_do_not_make_schema_recovery_a_defect():
    ctx = context()
    ctx["messages"]["M1"]["content"] = canonical({"execution_ledger": [{"action": "get_tool_schema", "tool": "example"}]})
    ctx["messages"]["M2"] = {"role": "system", "content": "Repeating discovery returns the same information."}
    ctx["turns"][0].update(input_message_ids=["M1", "M2"], output='{"type":"get_tool_schema","name":"example"}')
    value = assessment(grade=.5)
    value["dimensions"][0]["action_quotes"][0]["quote"] = "get_tool_schema"
    value["dimensions"][0]["observations"] = [
        {"turn": "0001", "message_id": "M1", "quote": "get_tool_schema"},
        {"turn": "0001", "message_id": "M2", "quote": "Repeating discovery"}]
    with pytest.raises(ValueError, match="rediscovery_requires_visible_content"):
        runtime.validate_quality(value, ctx)
    ctx["messages"]["M3"] = {"role": "user", "content": "example schema: required fields..."}
    ctx["turns"][0]["input_message_ids"].append("M3")
    value["dimensions"][0]["observations"].append({"turn": "0001", "message_id": "M3", "quote": "required fields"})
    assert runtime.validate_quality(value, ctx)["dimensions"][0]["grade"] == .5


def test_ledger_compaction_is_lossless_including_unicode_and_other_fields():
    ctx = context()
    rows = [{"turn": 1, "action": "search", "observation_ref": "obs_中文"},
            {"turn": 2, "action": "read", "business_success": "not_inferred"}]
    for i in (1, 2):
        ctx["messages"][f"L{i}"] = {"role": "user", "content": canonical({
            "execution_ledger": rows[:i], "note": "引用不是观察全文", "complete": False})}
    ctx["turns"][0]["input_message_ids"].append("L1")
    ctx["turns"].append({"turn": "0002", "input_message_ids": ["M1", "M2", "L2"], "output": "finish"})
    original = copy.deepcopy(ctx)
    payload = runtime.compact_quality_payload(ctx, "task", FIRST)
    assert len(payload["ledger_rows"]) == 2
    for mid, original_message in ctx["messages"].items():
        wire = payload["messages"][mid]
        if wire.get("content_encoding"):
            content = wire["content"]
            reconstructed = canonical({**content["other_fields"], "execution_ledger": [
                payload["ledger_rows"][rid] for rid in content["ledger_row_ids"]]})
        else:
            reconstructed = wire["content"]
        assert reconstructed == original_message["content"]
    assert payload["messages"]["M2"]["visible_at_turns"] == ["0002"]
    assert "M2" not in payload["turns"][0]["allowed_observation_ids"]
    assert ctx == original


def large_context():
    ctx = {"messages": {}, "turns": []}
    for i in range(6):
        mid, turn = f"M{i}", f"{i + 1:04d}"
        ctx["messages"][mid] = {"role": "user", "content": "Already returned " + str(i) * 7000}
        ctx["turns"].append({"turn": turn, "input_message_ids": [mid], "output": "repeat query"})
    return ctx


def window_config():
    return RunConfig(judge="api", judge_api={"context_limit": 50000})


def test_windows_cover_every_action_with_exact_visibility_and_reserved_review_space():
    ctx, config = large_context(), window_config()
    windows = runtime.quality_windows(ctx, "task", FIRST, config)
    assert len(windows) > 1
    assert [t for w in windows for t in w["turns"]] == ctx["turns"]
    for window in windows:
        mids = {mid for t in window["turns"] for mid in t["input_message_ids"]}
        assert window["messages"] == {mid: ctx["messages"][mid] for mid in mids}
        payload = runtime.compact_quality_payload(window, "task", FIRST)
        payload.update(scope="action_window", target_turns=[t["turn"] for t in window["turns"]])
        assert fits_context(runtime.request_messages(payload), config.judge_v2.max_output_tokens + runtime.FOLLOWUP_RESERVE, 50000)
    config = config.model_copy(update={"judge_api": config.judge_api.model_copy(update={"context_limit": 1000})})
    with pytest.raises(ValueError, match="single_action_context_too_large"):
        runtime.quality_windows(ctx, "task", FIRST, config)


def test_all_windows_are_reviewed_and_zero_findings_are_not_unknown(tmp_path, monkeypatch):
    config, calls = window_config(), []
    async def api(api, request, budget, key, stage, folder):
        payload = json.loads(request["messages"][1]["content"])
        calls.append((payload["target_turns"], stage))
        return {"choices": [{"finish_reason": "stop", "message": {"content": canonical(assessment())}}]}
    monkeypatch.setattr(runtime, "api_request", api)
    result = asyncio.run(runtime.assess_context(large_context(), "task", FIRST, config,
        Budget(tmp_path / "budget.db", 30, 1000000), asyncio.Semaphore(1), "test/", tmp_path))
    assert result["status"] == "valid"
    assert all(d["grade"] == 0 for d in result["dimensions"])
    assert len(calls) >= 4
    assert [t for turns, stage in calls if stage == "trajectory_quality" for t in turns] == [t["turn"] for t in large_context()["turns"]]
    assert calls[::2] == [(turns, "trajectory_quality") for turns, stage in calls[1::2]]
    assert all(stage == "trajectory_quality_review" for _, stage in calls[1::2])


def test_complete_window_unknown_is_repaired_to_three_dimensions(tmp_path, monkeypatch):
    calls = []
    async def api(*args):
        calls.append(args[4])
        return {"choices": [{"finish_reason": "stop", "message": {"content": canonical({
            "status": "unknown", "dimensions": [], "missing_reason": "Tool evidence missing."}
            if len(calls) == 1 else assessment())}}]}
    monkeypatch.setattr(runtime, "api_request", api)
    result = asyncio.run(runtime.assess_context(large_context(), "task", FIRST, window_config(),
        Budget(tmp_path / "budget.db", 30, 1000000), asyncio.Semaphore(1), "test/", tmp_path))
    assert result["status"] == "valid" and len(result["dimensions"]) == 3
    assert calls[:3] == ["trajectory_quality", "trajectory_quality", "trajectory_quality_review"]


def test_completed_window_survives_later_window_failure_and_retry(tmp_path, monkeypatch):
    calls = []
    fail = [True]
    async def api(api, request, budget, key, stage, folder):
        payload = json.loads(request['messages'][1]['content'])
        turns = payload['target_turns']
        calls.append((turns, stage))
        if turns[0] != '0001' and fail[0]:
            raise TimeoutError('simulated later-window failure')
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': canonical(assessment())}}]}
    monkeypatch.setattr(runtime, 'api_request', api)
    budget = Budget(tmp_path / 'budget.db', 30, 1000000)
    ctx, config = large_context(), window_config()
    with pytest.raises(TimeoutError):
        asyncio.run(runtime.assess_context(ctx, 'task', FIRST, config, budget,
            asyncio.Semaphore(1), 'first/', tmp_path / 'attempt-0000'))
    fail[0] = False
    result = asyncio.run(runtime.assess_context(ctx, 'task', FIRST, config, budget,
        asyncio.Semaphore(1), 'retry/', tmp_path / 'attempt-0001'))
    assert result['status'] == 'valid'
    assert len([c for c in calls if c[0][0] == '0001']) == 2  # original + review, no retry


def test_window_defects_are_consolidated_not_summed(tmp_path, monkeypatch):
    ctx, calls = large_context(), []
    async def api(api, request, budget, key, stage, folder):
        payload = json.loads(request["messages"][1]["content"])
        calls.append((payload["scope"], stage))
        # Only the first action is defective, so consolidation stays small.
        turn = payload["turns"][0]
        grade = .5 if turn["turn"] == "0001" else 0
        result = assessment(turn["turn"], turn["input_message_ids"][0], grade)
        if payload["scope"] == "consolidation":
            assert len(payload["window_assessments"]) > 1
            assert [t["turn"] for t in payload["turns"]] == ["0001"]
        return {"choices": [{"finish_reason": "stop", "message": {"content": canonical(result)}}]}
    monkeypatch.setattr(runtime, "api_request", api)
    result = asyncio.run(runtime.assess_context(ctx, "task", FIRST, window_config(),
        Budget(tmp_path / "budget.db", 30, 1000000), asyncio.Semaphore(1), "test/", tmp_path))
    assert result["dimensions"][0]["grade"] == .5
    assert calls[-2:] == [("consolidation", "trajectory_quality"), ("consolidation", "trajectory_quality_review")]
    assert runtime.validate_quality(result, ctx) == result


@pytest.mark.parametrize("recover", [True, False])
def test_repairs_are_bounded_without_defaulting_invalid_citations_to_full_marks(tmp_path, monkeypatch, recover):
    calls, payloads = [], []
    valid = assessment(grade=.5)
    invalid = copy.deepcopy(valid)
    invalid["dimensions"][0]["observations"][0]["message_id"] = "M2"
    responses = [invalid, invalid, valid, invalid, invalid, valid] if recover else [invalid] * 3
    async def api(api, request, budget, key, stage, folder):
        calls.append(stage)
        payloads.append(json.loads(request["messages"][1]["content"]))
        return {"choices": [{"finish_reason": "stop", "message": {"content": canonical(responses.pop(0))}}]}
    monkeypatch.setattr(runtime, "api_request", api)
    ctx, config = context(), RunConfig(judge="api")
    coroutine = runtime.assess_payload(runtime.compact_quality_payload(ctx, "task", FIRST), ctx, config,
        Budget(tmp_path / "budget.db", 6, 1000000), asyncio.Semaphore(1), "test/", tmp_path)
    if recover:
        assert asyncio.run(coroutine)["dimensions"][0]["grade"] == .5
        assert calls == ["trajectory_quality"] * 3 + ["trajectory_quality_review"] * 3
    else:
        with pytest.raises(ValueError, match="quality_evidence_not_visible_at_action"):
            asyncio.run(coroutine)
        assert len(calls) == 3
    assert "allowed_observation_ids" in read_json(tmp_path / "repair-00.json")["validation_error"]
    assert payloads[1]["scope"] == "citation_repair"
    assert payloads[1]["messages"] == {"M1": {"role": "user", "content": "Already returned", "visible_at_turns": ["0001"]}}
    if recover:
        assert "M2" in payloads[3]["messages"]  # mandatory review returns to full context


def test_replay_uses_the_original_business_judge_not_another_cached_verdict(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).parents[1] / "scripts/replay_trajectory_judge.py"
    spec = importlib.util.spec_from_file_location("replay_trajectory_judge_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "verify_input", lambda bundle: bundle)
    write_json(tmp_path / "episode_result.json", {"evaluation_id": "original", "business_label": "pass"})
    for eid in ("other", "original"):
        folder = tmp_path / "evaluations_v2" / eid / "attempt_0000"
        write_json(folder / "result.json", {"evaluation_id": eid, "evaluation_valid": True,
                                            "evaluation_status": "valid", "business_label": "pass"})
        write_json(folder / "input.json", {"source": eid})
    bundle, first, source = module.replay_inputs(tmp_path)
    assert bundle["source"] == first["evaluation_id"] == "original"
    source.unlink()
    with pytest.raises(ValueError, match="missing_valid_first_judge"):
        module.replay_inputs(tmp_path)
