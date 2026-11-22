import asyncio
from argparse import Namespace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from harness_awm.io import digest, read_json, rows, write_json
from harness_awm.judge_runtime import config_hash
from harness_awm.training.verl.__main__ import resolve_prepare_defaults
from harness_awm.training.verl.coverage import selected_task_ids
from harness_awm.training.rewards import population_advantages
from harness_awm.training.verl.rollout import score_coverage_group
from test_composite_reward import quality
from test_training_grpo import result_for
from test_zero_reward_quality_skip import episode


def test_all_tasks_overrides_fixed_target_preset():
    args = resolve_prepare_defaults(Namespace(full_coverage=True, preset="a100x8-27b"))
    assert (args.groups, args.group_size, args.target_updates, args.save_every) == (2000, 4, None, 500)
    assert args.shuffle_tasks
    with pytest.raises(ValueError):
        resolve_prepare_defaults(Namespace(full_coverage=True, steps=2746))
    with pytest.raises(ValueError):
        resolve_prepare_defaults(Namespace(full_coverage=True, target_updates=600))
    with pytest.raises(ValueError):
        resolve_prepare_defaults(Namespace(full_coverage=True, save_every=200))


def test_frozen_selection_replays_without_replacement():
    runtime = Path(__file__).resolve().parents[1] / "data/training_runtime_20261106_v1"
    pool = [r["task_instance_id"] for r in rows(runtime / "grpo_train.jsonl")]
    selected = selected_task_ids(pool, 20261106)
    assert len(set(selected)) == len(selected) == 2000
    assert set(selected) < set(pool)
    assert digest(selected) == "6e7cc79e17af6a6d7d42e8519f91589e5f85f7c619cbe10610b3a956e967e522"
    with pytest.raises(ValueError):
        selected_task_ids(list(reversed(pool)), 20261106)


def test_delivery_accepts_four_task_milestones_even_with_repeated_optimizer_step(tmp_path):
    from delivery.nioflow.online import completed_checkpoints
    runtime = Path(__file__).resolve().parents[1] / "data/training_runtime_20261106_v1"
    pool = [r["task_instance_id"] for r in rows(runtime / "grpo_train.jsonl")]
    selected = selected_task_ids(pool, 20261106)
    groups = []
    updates = 0
    for count, key in enumerate(selected, 1):
        row = dict(task_instance_id=key, candidate_count=4, uniform_primary="failed", final_rewards=[0] * 4,
                   quality_scored=0, status="held", reason="all_failed")
        if count in {1100, 2000}:
            rewards = [1.0, .95, 1.0, 1.0] if count == 1100 else [1.0] * 4
            updated = len(set(rewards)) > 1
            updates += int(updated)
            row.update(uniform_primary="confirmed_pass", final_rewards=rewards, rewards=rewards,
                       quality_scored=4, status="updated" if updated else "zero_advantage",
                       reason=None if updated else "equal_final_rewards")
        groups.append(dict(row, optimizer_step=updates))
    expected = {500: 0, 1000: 0, 1500: 1, 2000: 1}
    plan = dict(tasks=[{"task_instance_id": key} for key in selected], run_config={"seed": 20261106},
                full_coverage=True, groups=2000, group_size=4, target_updates=None, save_unit="groups", save_every=500)
    summary = dict(failed=False, target_met=True, groups=groups, optimizer_steps=updates)
    coverage = dict(complete=True, missing_task_ids=[], completed_tasks=2000, trajectories=8000,
                    optimizer_steps=updates, final_update=updates, selected_task_ids_sha256=digest(selected),
                    checkpoints=[{"completed_tasks": n, "optimizer_steps": s} for n, s in expected.items()])
    for name, data in (("plan", plan), ("summary", summary), ("coverage", coverage)):
        write_json(tmp_path / (name + ".json"), data)
    assert completed_checkpoints(tmp_path, pool) == expected
    groups[-1]["status"] = "held"
    write_json(tmp_path / "summary.json", summary)
    with pytest.raises(ValueError, match="全成功"):
        completed_checkpoints(tmp_path, pool)


@pytest.mark.parametrize("states,grades,uniform,quality_calls", [
    (["pass"] * 4, [0, 0, 0, 0], "confirmed_pass", 4),
    (["pass"] * 4, [0, 1, 1, 0], "confirmed_pass", 4),
    (["fail"] * 4, [0] * 4, "failed", 0),
    (["pass", "fail", "pass", "fail"], [0] * 4, None, 2),
    (["pass", "pass", "pass", "evaluation_error"], [0] * 4, None, 3),
])
def test_group_judges_real_admission_before_process_score(episode, tmp_path, monkeypatch, states, grades, uniform, quality_calls):
    source, task, config, _, _ = episode
    episodes = []
    for index in range(4):
        path = tmp_path / f"candidate-{index}"
        shutil.copytree(source, path)
        episodes.append((path, read_json(path / "episode_result.json")))

    async def judge(bundle, config, *args):
        state = states[int(Path(bundle["episode"]).name.rsplit("-", 1)[1])]
        result = result_for(("pass" if state == "evaluation_error" else state, "pass"))
        if state == "evaluation_error":
            result.update(evaluation_status=state, evaluation_valid=False, errors=["test error"])
        result["versions"] = {"inputs": bundle["hashes"], "configuration": config_hash(config)}
        return result

    primary = AsyncMock(side_effect=judge)
    async def process(bundle, *args):
        assert primary.await_count == 4
        index = int(Path(bundle["episode"]).name.rsplit("-", 1)[1])
        return {"status": "valid", "assessment": quality((grades[index], 0, 0))}
    secondary = AsyncMock(side_effect=process)
    monkeypatch.setattr("harness_awm.judge_runtime.evaluate_input", primary)
    monkeypatch.setattr("harness_awm.trajectory_quality.evaluate_trajectory", secondary)
    runner = SimpleNamespace(config=config, budget=None, judge_semaphore=asyncio.Semaphore(4))
    results, actual_uniform = asyncio.run(score_coverage_group(runner, task, episodes))
    assert (actual_uniform, primary.await_count, secondary.await_count) == (uniform, 4, quality_calls)
    if uniform:
        assert all(r["reward_admission_id"] for r in results)
    if uniform == "failed":
        assert all(r["reward"] == 0 and r["trajectory_quality"]["reason"] == "all_failed" for r in results)
    if uniform == "confirmed_pass":
        rewards = [r["reward"] for r in results]
        assert rewards == pytest.approx([1 - 0.05 * g for g in grades])
        assert all(r["trajectory_quality"]["status"] == "valid" for r in results)
        assert any(population_advantages(rewards)) == (len(set(grades)) > 1)


@pytest.mark.parametrize("outcomes,saved,statuses", [
    (["failed"] * 4, [(2, 0), (4, 0)], ["held"] * 4),
    (["pass_equal", "pass_different", "mixed", "failed"], [(2, 1), (4, 2)],
     ["zero_advantage", "updated", "updated", "held"]),
    (["mixed", "pass_missing", "pass_different"], [(2, 1), (3, 2)], ["updated", "held", "updated"]),
])
def test_fit_saves_by_completed_tasks_including_skips(tmp_path, monkeypatch, outcomes, saved, statuses):
    from harness_awm.contracts import RunConfig
    from harness_awm.training import verl
    from harness_support import fixture

    # Only the driver is exercised. The Ray workers and parent trainer's
    # checkpoint writer are replaced so this test can run on a CPU Job.
    spec = spec_from_file_location(verl.__name__ + "._test_driver", Path(verl.__file__).with_name("runtime.py"))
    runtime = module_from_spec(spec)
    with monkeypatch.context() as stub:
        stub.setitem(sys.modules, "verl.trainer.main_ppo", SimpleNamespace(TaskRunner=object))
        stub.setitem(sys.modules, "verl.trainer.ppo.ray_trainer", SimpleNamespace(RayPPOTrainer=object))
        spec.loader.exec_module(runtime)
    sample = fixture(tmp_path / "assets")
    tasks = [sample.model_copy(update={"split": "train", "task_instance_id": f"task-{i}"})
             for i in range(len(outcomes))]
    plan = {"run_config": RunConfig().model_dump(), "tasks": [t.model_dump() for t in tasks],
            "output": str(tmp_path), "group_size": 4, "concurrency": 4, "groups": len(tasks),
            "full_coverage": True, "target_updates": None, "save_unit": "groups", "save_every": 2, "gpus": 1,
            "model": "fixture", "reward_policy": "fixture", "checkpoint": {"sha256": "base"}, "eos_ids": [1]}
    write_json(tmp_path / "plan.json", plan)
    trainer = runtime.AWMTrainer.__new__(runtime.AWMTrainer)
    trainer.config = SimpleNamespace(awm=SimpleNamespace(plan=str(tmp_path / "plan.json")))
    trainer.train_dataloader = [[t.task_instance_id] for t in tasks]
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.async_rollout_manager = SimpleNamespace(llm_client=None)
    trainer.actor_rollout_wg = SimpleNamespace(parameter_probe=lambda: [{"finite": True, "changed_samples": 1}])
    trainer.checkpoint_manager = SimpleNamespace(update_weights=lambda step: None, sleep_replicas=lambda: None,
                                                 abort_replicas=lambda: None)
    trainer._shutdown_dump_executor = lambda: None
    recorded = []
    trainer._save_group_checkpoint = lambda count, plan: recorded.append((count, trainer.global_steps))
    trainer._update_turns = lambda tensors: {}
    monkeypatch.setattr(runtime, "preflight", lambda *args: None)
    monkeypatch.setattr(runtime, "turn_batch", lambda *args: None)
    monkeypatch.setattr("harness_awm.tracking.SwanLabTracker", lambda *args, **kw: SimpleNamespace(
        mode="disabled", url=None, log=lambda *args, **kw: None, finish=lambda **kw: None))
    async def collect(runner, client, tokenizer, task, folder, **kwargs):
        folder.mkdir(parents=True)
        assert kwargs["full_coverage"]
        outcome = outcomes[int(task.task_instance_id[-1])]
        uniform = "confirmed_pass" if outcome.startswith("pass") else "failed" if outcome == "failed" else None
        return dict(candidate_count=4, uniform_primary=uniform, quality_scored=3 if outcome == "pass_missing" else 4,
                    quality_errors=0, unscored=0, evaluation_errors=0, unknown=0, null_rewards=int(outcome == "pass_missing"))
    monkeypatch.setattr(runtime, "collect_group", collect)
    def read(folder, task, *args, **kwargs):
        outcome = outcomes[int(task.task_instance_id[-1])]
        assert outcome in {"pass_equal", "pass_different", "mixed"}
        rewards = {"pass_equal": [1.0] * 4, "pass_different": [1, .95, 1, .95], "mixed": [1, 0, 1, 0]}[outcome]
        return [], rewards, population_advantages(rewards)
    monkeypatch.setattr(runtime, "read_group", read)
    trainer.fit()
    report = read_json(tmp_path / "coverage.json")
    assert report["complete"] and report["missing_task_ids"] == []
    assert report["trajectories"] == 4 * len(tasks)
    assert recorded == [(r["completed_tasks"], r["optimizer_steps"]) for r in report["checkpoints"]] == saved
    assert report["final_update"] == saved[-1][1]
    assert [r["status"] for r in read_json(tmp_path / "summary.json")["groups"]] == statuses
