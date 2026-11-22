"""C/S/T priority, evidence visibility, cache, and GRPO boundary regressions."""
import asyncio
import copy
import json

import pytest

from harness_awm.composite_reward import components_for, reward_from_components
from harness_awm.contracts import EpisodeResult, RunConfig
from harness_awm.io import digest, read_json, write_json
from harness_awm.judge_runtime import config_hash
from harness_awm.judge_v2 import JudgeResult
from harness_awm.reward_values import COMPOSITE_VERSION
from harness_awm.training.rewards import InvalidRewardGroup, admitted_group, population_advantages
from harness_awm.trajectory_quality import actor_context, evaluate_trajectory, validate_quality
from test_training_grpo import result_for, engine_fixture, setup_sources


def quality(grades=(0, 0, 0)):
    return {"status": "valid", "missing_reason": None, "dimensions": [
        {"dimension": name, "grade": grade, "reason": "Recorded evidence checked.",
         "defect_id": name if grade else None,
         "action_quotes": [{"turn": "0002", "quote": "repeat"}] if grade else [],
         "observations": [{"turn": "0002", "message_id": name, "quote": "Already returned"}] if grade else []}
        for name, grade in zip(("information", "feedback", "exploration"), grades)]}


def composed(states=("pass", "pass"), grades=(0, 0, 0), **kwargs):
    result = result_for(states, **kwargs)
    result.update(trajectory_quality={"status": "valid", "assessment": quality(grades)},
                  reward_policy=COMPOSITE_VERSION, reward_admission_id="synthetic-admission")
    parts = components_for(result)
    result.update(reward_components=parts, reward=reward_from_components(parts, result["business_label"], result["task_status"]),
                  reward_valid=True)
    return result


@pytest.mark.parametrize("states,grades,expected", [
    (("pass", "pass"), (0, 0, 0), 1),
    (("pass", "pass"), (1, 0, 0), .95),
    (("pass", "fail"), (1, 1, 1), .8),
    (("unknown", "pass"), (0, 0, 0), .3),
    (("unknown", "fail"), (1, 1, 1), .24),
    (("fail", "pass"), (0, 0, 0), 0),
    (("pass", "unknown"), (0, 0, 0), .965),
])
def test_design_formula_and_serialization(states, grades, expected):
    result = composed(states, grades)
    assert result["reward"] == pytest.approx(expected)
    assert JudgeResult.model_validate(result).reward == pytest.approx(expected)
    episode = EpisodeResult(rollout_id="r", task_instance_id="t", termination_reason="finish",
                            reward=expected, reward_policy=COMPOSITE_VERSION)
    assert EpisodeResult.model_validate_json(episode.model_dump_json()).reward == expected


def test_success_always_above_unknown_and_failure_cannot_be_rescued():
    assert composed(("pass", "fail"), (1, 1, 1))["reward"] > composed(("unknown", "pass"))["reward"]
    result = result_for(("fail", "unknown"), kind="evidence_error")
    result["trajectory_quality"] = {"status": "error"}
    assert reward_from_components(components_for(result), "fail", "failed") == 0


def test_no_soft_items_is_neutral_and_unsettled_quality_does_not_get_default_full_marks():
    result = composed()
    result["items"] = [r for r in result["items"] if r["requirement_level"] == "required"]
    assert components_for(result)["S"] == 1
    for status in ("unknown", "error"):
        with pytest.raises(ValueError, match="quality_unsettled"):
            components_for({**result, "trajectory_quality": {"status": status}})
    for kind in ("evidence_error", "budget_exhausted", "not_investigated"):
        with pytest.raises(ValueError, match="soft_evidence_unsettled"):
            composed(("pass", "unknown"), kind=kind)


def test_all_success_group_can_now_have_nonzero_advantages():
    flags = {"execution_integrity": "valid", "terminal_state_status": "trusted"}
    results = [{**composed(grades=g), **flags} for g in [(0, 0, 0), (1, 0, 0), (.5, 0, 0), (0, 0, 0)]]
    rewards = admitted_group(results)
    assert rewards == [1, .95, .975, 1]
    assert population_advantages(rewards)[0] > 0 > population_advantages(rewards)[1]
    results[1]["reward"] = .99
    with pytest.raises(InvalidRewardGroup):
        admitted_group(results)


def test_changed_components_and_mixed_policy_groups_are_rejected():
    result = composed()
    result["reward_components"]["lambda_t"] = .8
    with pytest.raises(ValueError):
        JudgeResult.model_validate(result)
    a = {**composed(), "execution_integrity": "valid", "terminal_state_status": "trusted"}
    with pytest.raises(InvalidRewardGroup, match="mixed_reward_policies"):
        admitted_group([a, {**a, "reward_policy": "judge-v2-terminal-discount-v1"}], 2)


def visible_context():
    return {"messages": {name: {"role": "user", "content": "Already returned; use the existing result."}
                         for name in ("information", "feedback", "exploration")},
            "turns": [{"turn": "0002", "input_message_ids": ["information", "feedback", "exploration"], "output": "repeat query"}]}


def test_penalties_require_verbatim_prior_actor_visible_evidence():
    context = visible_context()
    value = quality((1, 0, 0))
    assert validate_quality(value, context)["dimensions"][0]["grade"] == 1
    for change in ("future", "invented_quote", "self_claim", "no_observation"):
        bad, ctx = copy.deepcopy(value), copy.deepcopy(context)
        if change == "future": ctx["turns"][0]["input_message_ids"] = []
        if change == "invented_quote": bad["dimensions"][0]["observations"][0]["quote"] = "not in evidence"
        if change == "self_claim": ctx["messages"]["information"]["role"] = "assistant"
        if change == "no_observation": bad["dimensions"][0]["observations"] = []
        with pytest.raises(ValueError): validate_quality(bad, ctx)


def test_actor_context_expands_observation_before_each_action_without_future_leakage(tmp_path, monkeypatch):
    from harness_awm import trajectory_quality as runtime
    first = {"role": "user", "content": "Update driver 45."}
    schema = {"role": "user", "content": "schema: driver_id is required; license_class is a string"}
    action = {"role": "assistant", "content": "get_tool_schema"}
    sealed = {}
    for index, messages in enumerate(([first], [first, action, schema]), 1):
        turn = tmp_path / "model_turns" / f"{index:04d}"
        write_json(turn / "input.json", {"messages": messages})
        write_json(turn / "output.json", {"content": "get_tool_schema"})
        for name in ("input.json", "output.json"):
            sealed[str((turn / name).relative_to(tmp_path))] = "fixture"
    monkeypatch.setattr(runtime, "verify_seal", lambda root: sealed)
    context = actor_context(tmp_path)
    assert context["turns"][0]["latest_observation"]["content"] == first["content"]
    assert context["turns"][1]["latest_observation"]["content"] == schema["content"]
    schema_id = context["turns"][1]["latest_observation"]["message_id"]
    assert schema_id.startswith("M") and len(schema_id) == 6
    assert schema_id not in context["turns"][0]["input_message_ids"]
    assert context["messages"][schema_id] == schema
    assert all(mid.startswith("A") for mid, message in context["messages"].items() if message["role"] == "assistant")


def test_same_defect_cannot_be_counted_twice_under_different_names():
    value = quality((1, 1, 0))
    value["dimensions"][1]["observations"] = copy.deepcopy(value["dimensions"][0]["observations"])
    with pytest.raises(ValueError, match="duplicate_trajectory_defect"):
        validate_quality(value, visible_context())


def test_composite_cannot_reuse_legacy_admission(tmp_path):
    from harness_awm.reward_policy import admission_for
    task, config, original, rubric = setup_sources(tmp_path)
    config = config.model_copy(update={"reward_policy": COMPOSITE_VERSION})
    with pytest.raises(ValueError, match="policy_mismatch"):
        admission_for(task, config, original, rubric)


def test_landlock_composite_requires_both_quality_controls(tmp_path, monkeypatch):
    from harness_awm.io import file_hash
    from harness_awm.reward_policy import admission_for
    task, config, original, rubric = setup_sources(tmp_path)
    config = config.model_copy(update={"reward_policy": COMPOSITE_VERSION, "isolation": "landlock"})
    manifest = read_json(config.reward_admissions)
    entry = manifest["tasks"][task.task_instance_id]
    proof_path = tmp_path / "proof.json"
    proof = {"passed": True, "task_instance_id": task.task_instance_id, "isolation": "landlock",
             "judge_config_hash": config_hash(config),
             "controls": [{"matches": True, "test_only": True, "path": str(tmp_path)} for _ in range(4)]}
    entry.update(policy_version=COMPOSITE_VERSION, judge_config_hash=config_hash(config), purpose="development",
                 isolation="landlock", calibration_ref=str(proof_path))
    manifest["schema_version"] = "awm-reward-admissions-v2"
    monkeypatch.setattr("harness_awm.reward_policy.verify_seal", lambda path: {})
    def save():
        write_json(proof_path, proof)
        entry["calibration_sha256"] = file_hash(proof_path)
        write_json(config.reward_admissions, manifest)
    save()
    with pytest.raises(ValueError, match="trajectory_reward_calibration"):
        admission_for(task, config, original, rubric)
    proof["trajectory_controls"] = [{"kind": kind, "matches": True, "test_only": True, "path": str(tmp_path)}
                                     for kind in ("clean", "redundant")]
    save()
    assert admission_for(task, config, original, rubric).policy_version == COMPOSITE_VERSION
    proof["trajectory_controls"][1]["matches"] = False
    save()
    with pytest.raises(ValueError, match="trajectory_reward_calibration"):
        admission_for(task, config, original, rubric)


def test_second_judge_uses_shared_rate_limiter(tmp_path):
    from harness_awm import backend, judge_runtime, judge_rate_limit, rubric_judge, trajectory_quality
    missing = object()
    saved = {m: getattr(m, "api_request", missing) for m in (backend, judge_runtime, rubric_judge, trajectory_quality)}
    try:
        config = RunConfig(judge_rpm=5000, judge_tpm=2000000)
        judge_rate_limit.install(config, tmp_path / "rate.db")
        assert trajectory_quality.api_request is judge_runtime.api_request
        assert {"trajectory_quality", "trajectory_quality_review"} <= judge_rate_limit.JUDGE_STAGES
    finally:
        for module, original in saved.items():
            if original is missing:
                delattr(module, "api_request")
            else:
                module.api_request = original


def test_repeated_unknown_with_complete_context_is_bounded_error_not_invented_quality(tmp_path, monkeypatch):
    from harness_awm import trajectory_quality as runtime
    from harness_awm.backend import Budget
    monkeypatch.setattr(runtime, "verify_input", lambda bundle: bundle)
    monkeypatch.setattr(runtime, "actor_context", lambda episode: visible_context())
    calls = []
    unknown = {"status": "unknown", "dimensions": [], "missing_reason": "External tool observation is unavailable."}
    async def respond(*args, **kwargs):
        calls.append(args[4])
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(unknown)}}]}
    monkeypatch.setattr(runtime, "api_request", respond)
    bundle = {"episode": str(tmp_path), "hashes": {"sealed": "test"}, "original": {"task": "test"}}
    config = RunConfig(judge="api", judge_protocol="rubric_v2", reward_policy=COMPOSITE_VERSION)
    budget = Budget(tmp_path / "budget.db", 4, 100000)
    args = (bundle, {"business_label": "pass", "task_status": "confirmed_pass"}, config, budget, asyncio.Semaphore(1), tmp_path / "quality")
    result = asyncio.run(evaluate_trajectory(*args))
    assert result["status"] == "error" and result["assessment"] is None and len(calls) == 3
    result = asyncio.run(evaluate_trajectory(*args))
    assert result["status"] == "error" and len(calls) == 6
    result = asyncio.run(evaluate_trajectory(*args))
    assert result["error"] == "trajectory_quality_persistent_attempt_limit" and len(calls) == 6


def test_primary_and_review_each_allow_one_quote_repair(tmp_path, monkeypatch):
    from harness_awm import trajectory_quality as runtime
    from harness_awm.backend import Budget
    monkeypatch.setattr(runtime, "verify_input", lambda bundle: bundle)
    monkeypatch.setattr(runtime, "actor_context", lambda episode: visible_context())
    valid = quality((.5, 0, 0))
    invalid = copy.deepcopy(valid)
    invalid["dimensions"][0]["observations"][0]["quote"] = "not an actual observation"
    responses, calls = [invalid, valid, invalid, valid], []
    async def respond(*args, **kwargs):
        calls.append(args[4])
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(responses.pop(0))}}]}
    monkeypatch.setattr(runtime, "api_request", respond)
    bundle = {"episode": str(tmp_path), "hashes": {"sealed": "test"}, "original": {"task": "test"}}
    config = RunConfig(judge="api", judge_protocol="rubric_v2", reward_policy=COMPOSITE_VERSION)
    result = asyncio.run(evaluate_trajectory(bundle, {"business_label": "pass", "task_status": "confirmed_pass"},
        config, Budget(tmp_path / "budget.db", 4, 100000), asyncio.Semaphore(1), tmp_path / "quality"))
    assert result["status"] == "valid" and result["reviewed"] and len(calls) == 4
    assert calls == ["trajectory_quality", "trajectory_quality", "trajectory_quality_review", "trajectory_quality_review"]


@pytest.mark.parametrize("status,expected_calls", [("error", 2), ("valid", 1), ("unknown", 1)])
def test_calibration_repairs_only_invalid_quality_not_valid_scores(tmp_path, monkeypatch, status, expected_calls):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("cst_calibration_test", Path(__file__).parents[1] / "scripts/calibrate_verl_grpo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    async def evaluate(*args):
        calls.append(args)
        return {"evaluation_valid": True, "business_label": "pass", "task_status": "confirmed_pass",
                "trajectory_quality": {"status": status if len(calls) == 1 else "valid"}}
    monkeypatch.setattr(module, "evaluate_episode", evaluate)
    asyncio.run(module.assess_control(tmp_path, None, None, None, None))
    assert len(calls) == expected_calls
    assert (tmp_path / "quality_retry.json").exists() == (status == "error")


def test_composite_reward_reaches_actual_optimizer_with_all_hard_passes(tmp_path, monkeypatch):
    import torch
    from harness_awm import trajectory_quality as runtime
    engine, task, _, _, _ = engine_fixture(tmp_path, monkeypatch, test_only=False, states=("pass", "pass"))
    config = engine.runner.config.model_copy(update={"reward_policy": COMPOSITE_VERSION})
    engine.runner.config = config
    manifest = read_json(config.reward_admissions)
    entry = manifest["tasks"][task.task_instance_id]
    entry.update(policy_version=COMPOSITE_VERSION, judge_config_hash=config_hash(config))
    write_json(config.reward_admissions, manifest)
    calls = [0]
    first_request = [None]
    async def respond(api, request, budget, key, stage, folder, **kwargs):
        payload = json.loads(request["messages"][1]["content"])
        value = quality()
        # Controlled transport fixture exercises arithmetic and plumbing, not Judge accuracy.
        prefix = key.rsplit("/", 1)[0]
        if first_request[0] is None:
            first_request[0] = prefix
        if prefix != first_request[0]:
            turn = payload["turns"][0]
            mid = next(i for i in turn["input_message_ids"] if payload["messages"][i]["role"] == "user")
            value["dimensions"][0].update(grade=1, defect_id="fixture",
                action_quotes=[{"turn": turn["turn"], "quote": (payload["messages"][turn["output_message_id"]]["content"]
                    if "output_message_id" in turn else turn["output"])}],
                observations=[{"turn": turn["turn"], "message_id": mid, "quote": payload["messages"][mid]["content"]}])
        calls[0] += 1
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}
    monkeypatch.setattr(runtime, "api_request", respond)
    before = {n: p.detach().clone() for n, p in engine.model.named_parameters()}
    group = asyncio.run(engine.collect(task))
    results = [read_json(p) for p in sorted(group.glob("candidate-*/episode_result.json"))]
    assert [r["business_label"] for r in results] == ["pass", "pass"]
    assert [r["reward"] for r in results] == [1, .95], results
    report = engine.update(group)
    assert report["status"] == "updated" and report["optimizer_step"] == 1
    assert any(not torch.equal(before[n], p) for n, p in engine.model.named_parameters())
