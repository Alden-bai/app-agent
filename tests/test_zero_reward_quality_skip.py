"""Exercise the real reward gates; only the external Judge is substituted."""
import asyncio
import copy
import shutil
from unittest.mock import AsyncMock

import pytest

from harness_awm.backend import Budget
from harness_awm.evaluation import apply_evaluation, evaluate_episode
from harness_awm.io import canonical, read_json, seal, write_json
from harness_awm.judge_inputs import prepare_input
from harness_awm.judge_runtime import config_hash
from harness_awm.judge_v2 import JudgeResult
from harness_awm.reward_policy import settle_reward
from harness_awm.reward_values import COMPOSITE_VERSION
from harness_awm.training.rewards import population_advantages
from test_composite_reward import quality
from test_training_grpo import result_for, setup_sources


@pytest.fixture
def episode(tmp_path):
    task, config, original, rubric = setup_sources(tmp_path / "sources")
    config = config.model_copy(update={"reward_policy": COMPOSITE_VERSION})
    manifest = read_json(config.reward_admissions)
    manifest["tasks"][task.task_instance_id].update(
        policy_version=COMPOSITE_VERSION, judge_config_hash=config_hash(config))
    write_json(config.reward_admissions, manifest)
    folder = tmp_path / "episode"
    folder.mkdir()
    for name in ("initial.db", "final.db"):
        shutil.copyfile(task.initial_state_ref, folder / name)
    write_json(folder / "task.json", task.model_dump())
    write_json(folder / "trajectory.json", {
        "task": task.instruction, "finished": False, "final_answer": "", "trajectory": []})
    write_json(folder / "episode_result.json", {
        "task_instance_id": task.task_instance_id, "rollout_id": "fixture", "attempt_id": "a0",
        "execution_integrity": "valid", "terminal_state_status": "trusted",
        "termination_reason": "finish", "test_only": False})
    (folder / "events.jsonl").write_text(canonical({
        "event_type": "environment_ready", "isolation": config.isolation,
        "tools_hash": task.tool_schema_hash}) + "\n")
    seal(folder)
    return folder, task, config, original, rubric


def replay(episode, monkeypatch, *, state="fail", gate=None, quality_result=None, evaluation_only=False):
    folder, task, config, original, rubric = episode
    if gate in {"revoked", "missing_admission"}:
        manifest = read_json(config.reward_admissions)
        if gate == "revoked":
            manifest["tasks"][task.task_instance_id]["status"] = "revoked"
        else:
            manifest["tasks"].clear()
        write_json(config.reward_admissions, manifest)
    if gate in {"test_only", "untrusted", "runtime_error"}:
        runtime = read_json(folder / "episode_result.json")
        runtime.update({"test_only": True} if gate == "test_only" else
                       {"terminal_state_status": "untrusted"} if gate == "untrusted" else
                       {"execution_integrity": "runtime_error"})
        write_json(folder / "episode_result.json", runtime)
    if gate == "evaluation_only" or evaluation_only:
        config = config.model_copy(update={"purpose": "evaluation"})
    bundle = prepare_input(folder, original, rubric)
    assessment = result_for((state, "pass"))
    assessment["versions"] = {"inputs": bundle["hashes"], "configuration": config_hash(config)}
    changes = {
        "invalid_evaluation": {"evaluation_valid": False},
        "evaluation_error": {"evaluation_status": "evaluation_error"},
        "rubric_error": {"evaluation_status": "rubric_error", "task_status": "undetermined"},
        "invalid_sample_status": {"evaluation_status": "invalid_sample"},
        "unsettled_tier": {"task_status": "undetermined"},
        "assessment_test_only": {"test_only": True},
        "judge_error": {"errors": ["unsettled evidence"]},
        "invalid_sample": {"sample_validity": {"status": "invalid"}},
        "required_rubric_issue": {"rubric_issues": [{"scope": "required"}]},
        "wrong_configuration": {"versions": {**assessment["versions"], "configuration": "wrong"}},
    }
    assessment.update(changes.get(gate, {}))
    if quality_result is None:
        quality_result = {"status": "valid", "assessment": quality((1, 0, 0))}
    # This is the previous orchestration: always attach quality before settling.
    baseline = settle_reward(dict(assessment, trajectory_quality=quality_result), bundle, task, config)
    judge = AsyncMock(return_value=copy.deepcopy(assessment))
    second_judge = AsyncMock(return_value=copy.deepcopy(quality_result))
    monkeypatch.setattr("harness_awm.judge_runtime.evaluate_input", judge)
    monkeypatch.setattr("harness_awm.trajectory_quality.evaluate_trajectory", second_judge)
    actual = asyncio.run(evaluate_episode(folder, task, config,
        Budget(folder / "budget.sqlite", 0, 0), asyncio.Semaphore(1)))
    assert {k: v for k, v in actual.items() if k != "trajectory_quality"} == {
        k: v for k, v in baseline.items() if k != "trajectory_quality"}
    JudgeResult.model_validate(actual)
    assert actual in [read_json(p) for p in (folder / "rewards_v2").glob("*.json")]
    return actual, second_judge


@pytest.mark.parametrize("quality_result", [
    {"status": "valid", "assessment": quality((1, 1, 1))},
    {"status": "error", "error": "timeout"}, {"status": "unknown"},
])
def test_admitted_failure_skips_quality_without_changing_reward(episode, monkeypatch, quality_result):
    result, judge = replay(episode, monkeypatch, quality_result=quality_result)
    judge.assert_not_awaited()
    assert result["reward_valid"] and result["reward"] == 0
    assert result["trajectory_quality"]["reason"] == "confirmed_zero_reward"
    assert result["trajectory_quality"]["usage"]["physical_requests"] == 0
    applied = apply_evaluation(episode[0], result)
    assert applied["process_status"] == "skipped" and applied["reward"] == 0


@pytest.mark.parametrize("gate", ["revoked", "missing_admission", "test_only", "untrusted",
    "runtime_error", "evaluation_only", "wrong_configuration"])
def test_failure_label_does_not_bypass_reward_gates(episode, monkeypatch, gate):
    result, judge = replay(episode, monkeypatch, gate=gate)
    judge.assert_awaited_once()
    assert result["reward"] is None and not result["reward_valid"]


@pytest.mark.parametrize("state", ["pass", "unknown", "fail"])
@pytest.mark.parametrize("gate", ["invalid_evaluation", "evaluation_error", "rubric_error",
    "invalid_sample_status", "unsettled_tier", "assessment_test_only", "judge_error",
    "invalid_sample", "required_rubric_issue"])
@pytest.mark.parametrize("quality_result", [
    {"status": "valid", "assessment": quality((1, 0, 0))},
    {"status": "error", "error": "timeout"},
])
def test_invalid_primary_skips_quality_with_identical_settlement(
        episode, monkeypatch, state, gate, quality_result):
    result, judge = replay(episode, monkeypatch, state=state, gate=gate, quality_result=quality_result)
    judge.assert_not_awaited()
    assert result["reward"] is None and not result["reward_valid"]
    assert result["trajectory_quality"]["reason"] == "invalid_primary_assessment"
    assert result["trajectory_quality"]["usage"]["physical_requests"] == 0


def test_evaluation_only_keeps_quality_for_invalid_primary(episode, monkeypatch):
    result, judge = replay(episode, monkeypatch, state="pass", gate="rubric_error", evaluation_only=True)
    judge.assert_awaited_once()
    assert result["reward_exclusion"] == "evaluation_only_no_training_reward"


@pytest.mark.parametrize("state,expected", [("pass", .95), ("unknown", .285)])
def test_nonzero_tiers_still_require_quality(episode, monkeypatch, state, expected):
    result, judge = replay(episode, monkeypatch, state=state)
    judge.assert_awaited_once()
    assert result["reward"] == expected


@pytest.mark.parametrize("state", ["pass", "unknown"])
def test_unsettled_quality_is_not_converted_to_zero(episode, monkeypatch, state):
    result, judge = replay(episode, monkeypatch, state=state,
                           quality_result={"status": "error", "error": "timeout"})
    judge.assert_awaited_once()
    assert result["reward"] is None and not result["reward_valid"]


def test_group_rewards_and_advantages_unchanged(tmp_path, monkeypatch):
    old, new = [], []
    for index, state in enumerate(("fail", "pass", "unknown", "fail")):
        root = tmp_path / str(index)
        root.mkdir()
        fixture = episode.__wrapped__(root)
        result, _ = replay(fixture, monkeypatch, state=state)
        new.append(result["reward"])
        old.append({"fail": 0, "pass": .95, "unknown": .285}[state])
    assert new == old
    assert population_advantages(new, std_floor=.1) == population_advantages(old, std_floor=.1)


@pytest.mark.parametrize("name", ["task.json", "final.db"])
def test_tampered_execution_cannot_reach_skip_branch(episode, monkeypatch, name):
    folder, task, config, _, _ = episode
    with (folder / name).open("ab") as handle:
        handle.write(b"tampered")
    judge = AsyncMock(side_effect=AssertionError("untrusted evidence must not be judged"))
    monkeypatch.setattr("harness_awm.judge_runtime.evaluate_input", judge)
    monkeypatch.setattr("harness_awm.trajectory_quality.evaluate_trajectory", judge)
    result = asyncio.run(evaluate_episode(folder, task, config,
        Budget(folder / "budget.sqlite", 0, 0), asyncio.Semaphore(1)))
    assert result["reward"] is None and not result["reward_valid"]
    assert "evidence_hash_mismatch" in str(result["errors"])
    judge.assert_not_awaited()


@pytest.mark.parametrize("candidates,expected", [
    ([{"original_finish": 100, "primary_finish": 10, "skipped": True},
      {"original_finish": 90, "primary_finish": 20, "skipped": False}], 10),
    ([{"original_finish": 100, "primary_finish": 10, "skipped": True},
      {"original_finish": 90, "primary_finish": 20, "skipped": True}], 80),
    ([{"original_finish": 100, "primary_finish": 10, "skipped": False}], 0),
])
def test_benchmark_retains_concurrent_group_barrier(candidates, expected):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).parents[1] / "scripts/benchmark_zero_reward_quality_skip.py"
    spec = importlib.util.spec_from_file_location("zero_quality_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.projected_saving(candidates) == expected
