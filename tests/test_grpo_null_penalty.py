"""Null penalties are training targets, not rewritten evaluation evidence."""
from types import SimpleNamespace

import pytest

from harness_awm.contracts import RunConfig
from harness_awm.io import digest, read_json, seal, write_json
from harness_awm.reward_values import VERSION
from harness_awm.training.rewards import admitted_samples, population_advantages
from harness_awm.training.verl.core import read_group


def result(reward, **changes):
    return {"reward": reward, "reward_valid": reward is not None, "test_only": False,
            "execution_integrity": "valid" if reward is not None else "runtime_error",
            "terminal_state_status": "trusted" if reward is not None else "unavailable",
            "reward_policy": VERSION, "termination_reason": "finish",
            "evaluation_status": "valid" if reward is not None else "evaluation_error", **changes}


@pytest.mark.parametrize("values,expected", [([1, None, 0, 1], [1, 0, 0, 1]),
    ([None] * 4, [0] * 4), ([1] * 4, [1] * 4), ([0] * 4, [0] * 4)])
def test_null_as_zero_and_relative_signal(values, expected):
    originals = [result(x) for x in values]
    indices, rewards, excluded = admitted_samples(originals, 4, null_reward_policy="zero")
    assert indices == [0, 1, 2, 3] and not excluded and rewards == expected
    assert [r["reward"] for r in originals] == values
    advantages = population_advantages(rewards, std_floor=.1)
    if values == [1, None, 0, 1]:
        assert advantages == pytest.approx([1, -1, -1, 1], abs=1e-5)
    else:
        assert advantages == [0] * 4


def test_exclusion_default_and_cancelled_or_test_samples_stay_excluded():
    samples = [result(1), result(None)]
    assert admitted_samples(samples, 2)[0] == [0]
    for change in ({"test_only": True}, {"termination_reason": "cancelled"}):
        samples[1] = result(None, **change)
        assert admitted_samples(samples, 2, null_reward_policy="zero")[0] == [0]


def test_identical_nonbinary_rewards_have_exact_zero_advantage():
    assert population_advantages([.95] * 3, std_floor=.1) == [0.0] * 3


def group_fixture(tmp_path, monkeypatch, *, missing_tokens=False):
    from harness_awm.training.verl import core
    config = RunConfig()
    task = SimpleNamespace(task_instance_id="t", model_dump=lambda: {"task_instance_id": "t"})
    admission = {"id": "test-admission"}
    monkeypatch.setattr(core, "admission_for", lambda *a, **kw: SimpleNamespace(model_dump=lambda: admission))
    monkeypatch.setattr(core, "sources_from_manifest", lambda *a: {})
    candidates = []
    for i, reward in enumerate([1, None, 0, 1]):
        folder = tmp_path / f"candidate-{i:03d}"
        record = result(reward, task_instance_id="t",
                        reward_admission_id=digest(admission) if reward is not None else None)
        write_json(folder / "episode_result.json", record)
        if not (missing_tokens and i == 1):
            turn = folder / "model_turns/0001"
            messages = [{"role": "user", "content": "task"}]
            write_json(turn / "input.json", {"messages": messages})
            write_json(turn / "output.json", {"content": "done"})
            write_json(turn / "policy_turn.json", {"prompt_ids": [2], "completion_ids": [3],
                "old_logprobs": [-1.0], "policy_version": "p/0", "test_only": False,
                "action_mask": [0, 1], "messages_hash": digest(messages),
                "sampling": {"temperature": 1.0, "top_k": 0, "top_p": 1.0}})
        seal(folder)
        candidates.append({"path": folder.name, "result_hash": digest(record)})
    write_json(tmp_path / "group.json", {"policy_version": "p/0", "config_hash": digest(config.model_dump()),
        "task": task.model_dump(), "candidates": candidates})
    return task, config, SimpleNamespace(decode=lambda *a, **kw: "done")


def test_verified_null_trajectory_enters_batch_and_preserves_evidence(tmp_path, monkeypatch):
    task, config, tokenizer = group_fixture(tmp_path, monkeypatch)
    path = tmp_path / "candidate-001/episode_result.json"
    original = path.read_bytes()
    trajectories, rewards, advantages = read_group(tmp_path, task, config, "p/0", 4, tokenizer,
                                                   null_reward_policy="zero")
    assert len(trajectories) == 4 and rewards == [1, 0, 0, 1] and advantages[1] < 0
    assert path.read_bytes() == original
    selection = read_json(tmp_path / "reward_selection.json")
    assert selection["null_penalties"] == [{"index": 1, "original_reward": None,
        "training_reward": 0.0, "reason": "evaluation_error"}]
    # Penalty mode cannot silently train from altered probability/context evidence.
    (tmp_path / "candidate-001/model_turns/0001/policy_turn.json").write_text('{}')
    with pytest.raises(ValueError):
        read_group(tmp_path, task, config, "p/0", 4, tokenizer, null_reward_policy="zero")


def test_no_generated_actions_cannot_create_fake_gradient(tmp_path, monkeypatch):
    task, config, tokenizer = group_fixture(tmp_path, monkeypatch, missing_tokens=True)
    trajectories, rewards, _ = read_group(tmp_path, task, config, "p/0", 4, tokenizer,
                                         null_reward_policy="zero")
    assert len(trajectories) == 3 and rewards == [1, 0, 1]
    assert read_json(tmp_path / "reward_selection.json")["excluded"] == [
        {"index": 1, "reason": "missing_policy_turns"}]


def test_800_preset_and_explicit_overrides():
    from harness_awm.training.verl.__main__ import resolve_prepare_defaults
    args = resolve_prepare_defaults(SimpleNamespace(preset="single-gpu-800"))
    assert (args.group_size, args.concurrency, args.target_updates, args.save_every) == (4, 4, 800, 200)
    assert args.groups == 3200 and args.null_reward_policy == "zero" and args.recover_tool_errors
    assert args.learning_rate == 1e-6 and args.rollout_correction == "token-is-v1"
    override = resolve_prepare_defaults(SimpleNamespace(preset="single-gpu-800", groups=1200, target_updates=200))
    assert (override.groups, override.target_updates) == (1200, 200)
    legacy = resolve_prepare_defaults(SimpleNamespace())
    assert legacy.groups == 1 and legacy.target_updates is None and legacy.null_reward_policy == "exclude"


@pytest.mark.parametrize("max_groups,target,expected_updates,expected_groups,expected_saves", [
    (3200, 800, 800, 1200, [200, 400, 600, 800]),
    (4, 3, 2, 4, [2]),
])
def test_effective_update_count_and_checkpoint_schedule(
        tmp_path, monkeypatch, max_groups, target, expected_updates, expected_groups, expected_saves):
    """Exercise the real control loop with fake rollout/optimizer/checkpoint workers; no training."""
    pytest.importorskip("verl")
    from harness_awm.training.verl import runtime
    from harness_awm.training.rewards import InvalidRewardGroup
    from harness_support import fixture
    from test_training_verl import turn

    task = fixture(tmp_path / "assets").model_copy(update={"split": "train"})
    plan = {"run_config": RunConfig().model_dump(), "tasks": [task.model_dump()],
            "output": str(tmp_path), "group_size": 4, "concurrency": 4,
            "groups": max_groups, "target_updates": target, "null_reward_policy": "zero",
            "gpus": 1, "save_every": 200, "clip": .2, "model": "fake",
            "reward_policy": VERSION, "checkpoint": {"sha256": "initial"}, "eos_ids": [1]}
    write_json(tmp_path / "plan.json", plan)
    trainer = runtime.AWMTrainer.__new__(runtime.AWMTrainer)
    trainer.config = SimpleNamespace(awm=SimpleNamespace(plan=str(tmp_path / "plan.json")))
    trainer.train_dataloader = [[task.task_instance_id]]
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.async_rollout_manager = SimpleNamespace(llm_client=None)
    trainer.actor_rollout_wg = SimpleNamespace(parameter_probe=lambda: [{"finite": True, "changed_samples": 1}])
    syncs, updates, saves = [], [], []
    trainer.checkpoint_manager = SimpleNamespace(update_weights=syncs.append,
        sleep_replicas=lambda: None, abort_replicas=lambda: None)
    trainer._save_checkpoint = lambda: saves.append(trainer.global_steps)
    trainer._shutdown_dump_executor = lambda: None
    def update(tensors):
        assert tensors["responses"].shape[0] == 4
        assert tensors["response_mask"].sum().item() == 4
        updates.append(trainer.global_steps)
        return {}
    trainer._update_turns = update
    monkeypatch.setattr(runtime, "preflight", lambda *a: None)

    class Tracker:
        mode, url = "disabled", None
        def __init__(self, *a, **kw): pass
        def log(self, *a, **kw): pass
        def finish(self, *a, **kw): pass
    monkeypatch.setattr("harness_awm.tracking.SwanLabTracker", Tracker)
    async def collect(runner, client, tokenizer, collected_task, folder, **kwargs):
        assert kwargs["group_size"] == 4 and kwargs["concurrency"] == 4
        assert collected_task.task_instance_id == task.task_instance_id
        folder.mkdir(parents=True)
    monkeypatch.setattr(runtime, "collect_group", collect)
    def group(folder, *a, **kwargs):
        assert kwargs["null_reward_policy"] == "zero"
        index = int(folder.name.split('-')[1])
        # Alternate all-equal, missing-action, and four learnable groups.
        if index % 6 == 1:
            raise InvalidRewardGroup("missing_policy_turns")
        rewards = [0] * 4 if index % 6 == 0 else [1, 0, 0, 1]
        return [[turn([2], [3])] for _ in range(4)], rewards, population_advantages(rewards, std_floor=.1)
    monkeypatch.setattr(runtime, "read_group", group)
    trainer.fit()
    report = read_json(tmp_path / "summary.json")
    assert report["optimizer_steps"] == expected_updates == len(updates)
    assert len(report["groups"]) == expected_groups
    assert saves == expected_saves
    assert report["target_met"] == (expected_updates == target)
    previous = 0
    for row in report["groups"]:
        previous += row["status"] == "updated"
        assert row["optimizer_step"] == previous
    assert syncs[-1] == expected_updates - 1
