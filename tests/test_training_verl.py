"""Offline verl migration contracts; no CUDA, network, or paid Judge requests."""
import asyncio
from types import SimpleNamespace

import pytest

from harness_awm.contracts import RunConfig
from harness_awm.io import read_json, write_json
from harness_awm.training.grpo import clipped_objective, turn_logprobs
from harness_awm.training.rewards import InvalidRewardGroup, population_advantages
from harness_awm.training.verl.core import check_logprobs, read_group, turn_batch, validate_turn
from harness_awm.training.verl.rollout import VerlActorBackend, collect_group


def turn(prompt, response, version="p/0"):
    return {"prompt_ids": prompt, "completion_ids": response, "old_logprobs": [-2.0] * len(response),
            "policy_version": version, "test_only": False,
            "action_mask": [0] * len(prompt) + [1] * len(response),
            "sampling": {"temperature": 1.0, "top_k": 0, "top_p": 1.0}}


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_turn_loss_and_gradient_equal_original_trajectory_mean(world_size):
    import torch
    # Deliberately unequal lengths and a non-prefix second prompt (compaction).
    trajectories = [[turn([2, 3], [4]), turn([8, 9], [5, 6, 7])],
                    [turn([1], [3, 4])]]
    advantages = population_advantages([1, 0.3])
    tensors = turn_batch(trajectories, advantages, 0, world_size)
    assert len(tensors["prompts"]) % world_size == 0
    assert tensors["prompts"][1].tolist() == [8, 9]
    assert tensors["response_mask"].sum() == 6
    current = tensors["old_log_probs"].clone().requires_grad_()
    # Ratios include unclipped and both clipped regions.
    perturb = torch.tensor([[0.4, 0, 0], [-0.5, 0.05, 0.3], [0.4, -0.4, 0]])
    perturbed = current.clone()
    perturbed[:3] = current[:3] + perturb
    mask = tensors["response_mask"]
    migrated = (clipped_objective(perturbed, tensors["old_log_probs"], tensors["advantages"])
                * mask).sum() / len(current)
    original = current.new_zeros(())
    row = 0
    for turns, advantage in zip(trajectories, advantages):
        total = sum(len(t["completion_ids"]) for t in turns)
        for t in turns:
            n = len(t["completion_ids"])
            original += clipped_objective(perturbed[row, :n], tensors["old_log_probs"][row, :n],
                                          advantage).sum() / (len(trajectories) * total)
            row += 1
    assert torch.allclose(migrated, original)
    assert torch.allclose(torch.autograd.grad(migrated, current, retain_graph=True)[0],
                          torch.autograd.grad(original, current)[0])


def test_logprob_mismatch_checks_only_real_response_tokens():
    batch = turn_batch([[turn([2], [3])], [turn([4], [5, 6])]], [1, -1], 0, 4)
    current = batch["old_log_probs"].clone()
    current[~batch["response_mask"].bool()] = float("nan")
    assert check_logprobs(batch, current) == 0
    current[0, 0] += .1
    with pytest.raises(ValueError, match="logprob_mismatch"):
        check_logprobs(batch, current)


@pytest.mark.parametrize("change", [{"policy_version": "old"}, {"test_only": True},
    {"old_logprobs": [float("nan")]}, {"old_logprobs": [0.1]}, {"action_mask": [1, 1]},
    {"sampling": {"temperature": .7, "top_k": 0, "top_p": 1.0}}])
def test_invalid_turn_rejected(change):
    with pytest.raises(ValueError):
        validate_turn({**turn([1], [2]), **change}, "p/0")


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return [7, 8]

    def decode(self, ids, **kwargs):
        return "done"


def test_remote_actor_preserves_sampling_eos_and_final_budget(tmp_path):
    class Client:
        async def generate(self, **kwargs):
            self.request = kwargs
            return SimpleNamespace(token_ids=[3, 9], log_probs=[-2., -1.], stop_reason="completed")
    client = Client()
    actor = VerlActorBackend(client, Tokenizer(), RunConfig(max_output_tokens=20),
                             policy_version="p/0", request_id="episode", eos_ids=[9, 10], seed=123)
    result = asyncio.run(actor.generate([{"role": "user", "content": "x"}], tmp_path, 2, output_tokens=2))
    assert result["finish_reason"] == "stop"
    assert client.request["sampling_params"]["max_tokens"] == 2
    assert client.request["sampling_params"]["top_k"] == -1
    assert client.request["sampling_params"]["stop_token_ids"] == [9, 10]
    assert read_json(tmp_path / "policy_turn.json")["old_logprobs"] == [-2., -1.]


@pytest.mark.parametrize("tokens,logprobs,reason", [([3], [-1.], "aborted"),
    ([3], None, "completed"), ([3], [-1.], "completed"), ([9, 3], [-1., -1.], "completed")])
def test_incomplete_remote_output_cannot_supply_training_tokens(tmp_path, tokens, logprobs, reason):
    class Client:
        async def generate(self, **kwargs):
            return SimpleNamespace(token_ids=tokens, log_probs=logprobs, stop_reason=reason)
    actor = VerlActorBackend(Client(), Tokenizer(), RunConfig(max_output_tokens=2),
                             policy_version="p/0", request_id="episode", eos_ids=[9], seed=0)
    with pytest.raises(ValueError):
        asyncio.run(actor.generate([], tmp_path, 1))


def test_group_collects_concurrently_with_separate_directories_and_stable_order(tmp_path):
    class Task:
        task_instance_id = "t"
        def model_dump(self):
            return {"task_instance_id": "t"}
    class Runner:
        config = RunConfig()
        active = peak = 0
        seeds = []
        async def run(self, task, folder, request_id, actor):
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.seeds.append(actor.seed)
            folder.mkdir()
            await asyncio.sleep(.01)
            self.active -= 1
            return {"reward": None}
    runner = Runner()
    asyncio.run(collect_group(runner, None, Tokenizer(), Task(), tmp_path / "group",
        policy_version="p/0", group_size=4, eos_ids=[9], concurrency=2))
    manifest = read_json(tmp_path / "group/group.json")
    assert runner.peak == 2
    assert len(set(runner.seeds)) == 4
    assert [c["path"] for c in manifest["candidates"]] == [f"candidate-{i:03d}" for i in range(4)]


def test_task_failure_joins_other_candidates_before_return(tmp_path):
    class Task:
        task_instance_id = "t"
    class Runner:
        config = RunConfig()
        active = 0
        async def run(self, task, folder, request_id, actor):
            self.active += 1
            try:
                if request_id.endswith("000"):
                    await asyncio.sleep(.01)
                    raise RuntimeError("environment failed")
                await asyncio.sleep(10)
            finally:
                self.active -= 1
    runner = Runner()
    with pytest.raises(ExceptionGroup):
        asyncio.run(collect_group(runner, None, Tokenizer(), Task(), tmp_path / "group",
            policy_version="p/0", group_size=2, eos_ids=[9], concurrency=2))
    assert runner.active == 0
    assert not (tmp_path / "group/group.json").exists()


def test_sealed_judge_results_and_compacted_turns_reused(tmp_path, monkeypatch):
    from test_training_grpo import engine_fixture
    engine, task, *_ = engine_fixture(tmp_path, monkeypatch, test_only=False, states=("pass", "unknown"))
    group = asyncio.run(engine.collect(task))
    trajectories, rewards, advantages = read_group(group, task, engine.runner.config,
                                                   engine.policy_version, 2, engine.tokenizer)
    assert rewards == [1, 0.3]
    assert len(trajectories) == 2 and all(len(t) == 2 for t in trajectories)
    assert advantages == pytest.approx([1, -1], abs=4e-6)
    # Revocation is read again at the update boundary.
    admissions = read_json(engine.runner.config.reward_admissions)
    admissions["tasks"][task.task_instance_id]["status"] = "revoked"
    write_json(engine.runner.config.reward_admissions, admissions)
    with pytest.raises((ValueError, InvalidRewardGroup)):
        read_group(group, task, engine.runner.config, engine.policy_version, 2, engine.tokenizer)


@pytest.mark.parametrize("correction_mode", [False, True])
def test_real_verl_padding_loss_and_optimizer_match_turnwise_reference(monkeypatch, correction_mode):
    pytest.importorskip("verl")
    import torch
    from verl.protocol import DataProto
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import left_right_2_no_padding
    from harness_awm.training.verl.loss import policy_loss
    from test_training_grpo import tiny_policy
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        function = getattr(qwen, name)
        monkeypatch.setattr(qwen, name, getattr(function, "__wrapped__", function))

    # Only FlashAttention's gather helper is replaced on CPU. DataProto,
    # nested conversion, response alignment, and the loss callback are real verl.
    def cpu_unpad(hidden, attention):
        lengths = attention.sum(-1, dtype=torch.int32)
        indices = attention.flatten().nonzero().flatten()
        offsets = torch.cat([lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)])
        return hidden.flatten(0, 1)[indices], indices, offsets, int(lengths.max())
    monkeypatch.setattr("verl.workers.utils.padding.unpad_input", cpu_unpad)

    model, _ = tiny_policy("qwen3.5")
    model.eval()
    trajectories = [[turn([3, 4], [5, 6]), turn([7, 3, 2], [8])], [turn([6], [7, 8, 9])]]
    for turns in trajectories:
        for t in turns:
            t["old_logprobs"] = (turn_logprobs(model, t).detach() - (0.1 if correction_mode else 0)).tolist()
    tensors = turn_batch(trajectories, [1., -1.], 0, 2)
    if correction_mode:
        from harness_awm.training.verl.core import rollout_correction
        sampled = tensors["old_log_probs"].clone()
        extra, _ = rollout_correction(tensors, sampled + 0.1)
        tensors.update(extra)
        assert torch.equal(tensors["old_log_probs"], sampled)
    td = left_right_2_no_padding(DataProto.from_dict(tensors=tensors).to_tensordict())
    tu.assign_non_tensor(td, dp_size=1, global_batch_size=len(tensors["prompts"]))
    # Real verl expects log probabilities over unpadded full sequences, aligned
    # to the token predicted at the next position (last position unused).
    log_probs = []
    for ids in td["input_ids"].unbind():
        logits = model(input_ids=ids[None], use_cache=False).logits[0].float().log_softmax(-1)
        targets = torch.cat([ids[1:], ids.new_zeros(1)])
        log_probs.append(logits.gather(-1, targets[:, None]).squeeze(-1))
    nested = torch.nested.as_nested_tensor(log_probs, layout=torch.jagged)
    loss, _ = policy_loss({"log_probs": nested}, td)
    loss.backward()
    actual = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    reference = 0
    for turns, adv in zip(trajectories, [1., -1.]):
        total = sum(len(t["completion_ids"]) for t in turns)
        for t in turns:
            current = turn_logprobs(model, t)
            old = current.new_tensor(t["old_logprobs"])
            anchor = old + 0.1 if correction_mode else old
            weight = (anchor - old).exp() if correction_mode else 1.0
            reference += (clipped_objective(current, anchor, adv) * weight).sum() / (2 * total)
    reference.backward()
    for p, grad in zip(model.parameters(), actual):
        assert torch.allclose(p.grad, grad, atol=2e-6, rtol=2e-4)
    before = [p.detach().clone() for p in model.parameters()]
    torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    assert any(not torch.equal(p, old) for p, old in zip(model.parameters(), before))


@pytest.mark.parametrize("save_unit", ["updates", "groups"])
@pytest.mark.parametrize("outcome", ["held", "zero_advantage", "updated"])
@pytest.mark.parametrize("groups,target,offset", [(1, None, 0), (5, 2, 21)])
@pytest.mark.parametrize("api_fault", [False, True])
def test_verl_fit_skips_invalid_and_zero_signal_groups(tmp_path, monkeypatch, outcome, groups, target, offset, api_fault, save_unit):
    pytest.importorskip("verl")
    from harness_awm.training.verl import runtime
    from harness_support import fixture
    task = fixture(tmp_path / "assets").model_copy(update={"split": "train"})
    if save_unit == "groups":
        target = None
    save_every = 100 if groups == 400 else 2
    plan = {"run_config": RunConfig().model_dump(), "tasks": [task.model_dump()],
            "output": str(tmp_path), "group_size": 2, "concurrency": 2,
            "groups": groups, "target_updates": target, "group_offset": offset, "pilot_guards": api_fault,
            "gpus": 1, "save_every": save_every, "save_unit": save_unit, "clip": .2,
            "model": "tiny", "reward_policy": "test", "checkpoint": {"sha256": "initial"}, "eos_ids": [1]}
    write_json(tmp_path / "plan.json", plan)
    trainer = runtime.AWMTrainer.__new__(runtime.AWMTrainer)
    trainer.config = SimpleNamespace(awm=SimpleNamespace(plan=str(tmp_path / "plan.json")))
    trainer.train_dataloader = [[task.task_instance_id]]
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.async_rollout_manager = SimpleNamespace(llm_client=None)
    trainer.actor_rollout_wg = SimpleNamespace(parameter_probe=lambda: [{"finite": True, "changed_samples": 1}])
    actions = []
    trainer.checkpoint_manager = SimpleNamespace(
        update_weights=lambda step: actions.append(("sync", step)),
        sleep_replicas=lambda: actions.append(("sleep",)),
        abort_replicas=lambda: actions.append(("abort",)))
    trainer._save_checkpoint = lambda: actions.append(("save", trainer.global_steps))
    trainer._save_group_checkpoint = lambda count, plan: actions.append(("save_group", count, trainer.global_steps))
    trainer._shutdown_dump_executor = lambda: None
    trainer._update_turns = lambda tensors: actions.append(("update",)) or {}
    monkeypatch.setattr(runtime, "preflight", lambda *args: None)
    if api_fault:
        from harness_awm.training.verl import pilot
        monkeypatch.setattr(pilot, "resource_check", lambda *args, **kwargs: None)
        monkeypatch.setattr(pilot, "group_health", lambda *args: {"api_fault": True, "quarantine": True})
        monkeypatch.setattr(pilot, "archive_group", lambda *args: "test-archive")
        plan["checkpoint_dir"] = str(tmp_path / "checkpoints")
        write_json(tmp_path / "plan.json", plan)
    class Tracker:
        mode = "disabled"
        url = None
        def __init__(self, *args, **kwargs): pass
        def log(self, *args, **kwargs): pass
        def finish(self, *args, **kwargs): pass
    monkeypatch.setattr("harness_awm.tracking.SwanLabTracker", Tracker)
    async def collect(*args, **kwargs):
        args[4].mkdir(parents=True)
    monkeypatch.setattr(runtime, "collect_group", collect)
    def group(*args, **kwargs):
        if outcome == "held":
            raise InvalidRewardGroup("null reward")
        return [[turn([2], [3])], [turn([2], [4])]], [1, 0], ([1, -1] if outcome == "updated" else [0, 0])
    monkeypatch.setattr(runtime, "read_group", group)
    trainer.fit()
    report = read_json(tmp_path / "summary.json")
    assert report["groups"][0]["status"] == outcome
    updates = (target or groups) if outcome == "updated" else 0
    assert report["optimizer_steps"] == updates
    assert actions.count(("update",)) == updates
    assert len(report["groups"]) == (updates or groups)
    assert [r['group'] for r in report['groups']] == list(range(offset, offset + len(report['groups'])))
    assert report["target_met"] == (target is None or updates >= target)
    if save_unit == "groups":
        expected_saves = list(range(save_every, groups + 1, save_every))
        if groups % save_every:
            expected_saves.append(groups)
        assert [a[1] for a in actions if a[0] == "save_group"] == expected_saves
        assert trainer.global_steps == updates
    if outcome == "updated":
        assert actions.index(("sleep",)) < actions.index(("update",))


@pytest.mark.parametrize("groups,target", [(1, None), (3200, 800)])
def test_pinned_verl_config_composes_and_validates(groups, target):
    pytest.importorskip("verl")
    from harness_awm.training.verl.config import compose_config
    from verl.utils.config import validate_config
    plan = {"run_config": {"seed": 1, "context_limit": 32768, "max_output_tokens": 2048},
            "model": "/tmp/model", "learning_rate": 1e-6, "group_size": 4, "gpus": 1,
            "concurrency": 4, "gpu_memory_utilization": .35, "groups": groups,
            "target_updates": target, "output": "/tmp/test"}
    config = compose_config(plan)
    validate_config(config, use_reference_policy=False, use_critic=False)
    assert config.actor_rollout_ref.actor.strategy == "fsdp2"
    assert config.actor_rollout_ref.actor.fsdp_config.optimizer_offload is True
    assert config.actor_rollout_ref.actor.clip_ratio_low == .2
    assert config.actor_rollout_ref.actor.clip_ratio_high == .2
    assert not config.algorithm.use_kl_in_reward
    assert not config.actor_rollout_ref.actor.use_kl_loss
    assert config.actor_rollout_ref.rollout.temperature == 1.0
    assert config.actor_rollout_ref.rollout.top_p == 1.0
    assert config.data.train_batch_size == 1
    assert config.actor_rollout_ref.rollout.n == 4
    assert config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 1
    assert config.actor_rollout_ref.actor.ppo_epochs == 1
    assert config.trainer.total_training_steps == (target or groups)
    assert not config.actor_rollout_ref.model.use_remove_padding
    assert config.actor_rollout_ref.rollout.logprobs_mode == "raw_logprobs"
    assert config.ray_kwargs.ray_init.runtime_env.env_vars.AWM_VERL_TEXT_MODEL == "1"


@pytest.mark.parametrize("preset", [None, "single-gpu-800"])
def test_prepare_is_offline_and_binds_checkpoint_and_sources(tmp_path, monkeypatch, preset):
    from test_training_grpo import setup_sources, tiny_policy
    from harness_awm.training.verl import __main__ as cli
    task, config, *_ = setup_sources(tmp_path / "assets")
    model, tokenizer = tiny_policy("qwen3.5")
    model_path = tmp_path / "model"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    config_path = tmp_path / "config.json"
    write_json(config_path, config.model_dump())
    monkeypatch.setattr(cli, "load_tasks", lambda *args: ([task], {}, config.judge_inputs_manifest))
    args = SimpleNamespace(config=config_path, tasks=tmp_path, task=None, model=model_path,
        output=tmp_path / "run", group_size=2, groups=1, gpus=1, concurrency=2,
        learning_rate=1e-6, gpu_memory_utilization=.35, save_every=10)
    if preset:
        args.preset = preset
        for field in ("group_size", "groups", "concurrency", "save_every"):
            setattr(args, field, None)
    report = cli.prepare(args)
    assert report["judge_requests"] == 0 and report["training_started"] is False
    plan = cli.load_plan(args.output)
    assert plan["tasks"][0]["split"] == "train" and plan["eos_ids"] == [1]
    if preset:
        assert plan["group_size"] == plan["concurrency"] == 4
        assert plan["target_updates"] == 800 and plan["save_every"] == 200
        assert plan["training_reward_policy"] == "null-as-zero-v1"
        assert plan["run_config"]["recover_tool_errors"] is True
        assert plan["clip"] == .2 and plan["beta"] == 0
        assert not (args.output / "started.json").exists()
    (model_path / "generation_config.json").write_text('{}')
    with pytest.raises(ValueError, match="checkpoint changed"):
        cli.load_plan(args.output)



def test_rollout_correction_preserves_sampled_probs_masks_padding_and_bounds():
    import torch
    from harness_awm.training.verl.core import rollout_correction
    batch = turn_batch([[turn([2], [3])], [turn([4], [5, 6])]], [1, -1], 0, 4)
    sampled = batch["old_log_probs"].clone()
    current = sampled.clone().requires_grad_()
    with torch.no_grad():
        current[batch["response_mask"].bool()] += 0.17
        current[~batch["response_mask"].bool()] = float("nan")
    extra, report = rollout_correction(batch, current)
    assert torch.equal(batch["old_log_probs"], sampled)
    assert report["ratio_max"] == pytest.approx(float(torch.exp(torch.tensor(0.17))))
    assert torch.isfinite(extra["rollout_is_weights"]).all()
    assert not extra["rollout_is_weights"].requires_grad
    assert not extra["learner_old_log_probs"].requires_grad
    assert (extra["rollout_is_weights"][~batch["response_mask"].bool()] == 0).all()
    with torch.no_grad():
        current[0, 0] = sampled[0, 0] + 0.8
    with pytest.raises(ValueError, match="ratio_out_of_bounds"):
        rollout_correction(batch, current)
    with torch.no_grad():
        current[0, 0] = float("nan")
    with pytest.raises(ValueError, match="invalid_rollout_correction"):
        rollout_correction(batch, current)


def test_400_groups_stop_and_save_without_any_optimizer_update(tmp_path, monkeypatch):
    test_verl_fit_skips_invalid_and_zero_signal_groups(
        tmp_path, monkeypatch, "zero_advantage", 400, None, 0, False, "groups")


def test_group_checkpoint_keeps_optimizer_count_and_distinct_snapshots(tmp_path):
    pytest.importorskip("verl")
    import torch
    from harness_awm.training.verl.runtime import AWMTrainer
    trainer = AWMTrainer.__new__(AWMTrainer)
    trainer.global_steps = 7
    calls = []
    trainer.actor_rollout_wg = SimpleNamespace(save_checkpoint=lambda *a, **kw: calls.append((a, kw)))
    trainer.train_dataloader = SimpleNamespace(state_dict=lambda: {"consumed": 100})
    plan = {"checkpoint_dir": str(tmp_path)}
    trainer._save_group_checkpoint(100, plan)
    trainer._save_group_checkpoint(200, plan)
    assert trainer.global_steps == 7
    assert [call[0][2] for call in calls] == [7, 7]
    assert calls[0][0][0] != calls[1][0][0]
    for count in (100, 200):
        folder = tmp_path / f"group_step_{count}"
        state = read_json(folder / "training_state.json")
        assert state["groups"] == count and state["optimizer_steps"] == 7
        assert torch.load(folder / "data.pt", weights_only=True) == {"consumed": 100}
    assert read_json(tmp_path / "latest_group_checkpoint.json")["groups"] == 200


def test_group_steps_override_preset_update_target():
    from harness_awm.training.verl.__main__ import resolve_prepare_defaults
    args = resolve_prepare_defaults(SimpleNamespace(preset="single-gpu-800", steps=400, save_every=100))
    assert args.groups == 400 and args.target_updates is None and args.save_every == 100
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_prepare_defaults(SimpleNamespace(steps=400, target_updates=400))


@pytest.mark.parametrize("failure", [None, "archive", "rollout"])
def test_fit_overlaps_archival_and_drains_before_summary(tmp_path, monkeypatch, failure):
    pytest.importorskip("verl")
    import threading
    from harness_awm.training.verl import runtime, pilot
    from harness_support import fixture

    task = fixture(tmp_path / "assets").model_copy(update={"split": "train"})
    plan = {"run_config": RunConfig().model_dump(), "tasks": [task.model_dump()],
            "output": str(tmp_path), "group_size": 2, "concurrency": 2, "groups": 2,
            "pilot_guards": True, "checkpoint_dir": str(tmp_path / "checkpoints"),
            "gpus": 1, "save_every": 2, "save_unit": "groups", "clip": .2,
            "model": "tiny", "reward_policy": "test", "checkpoint": {"sha256": "initial"}, "eos_ids": [1]}
    write_json(tmp_path / "plan.json", plan)
    trainer = runtime.AWMTrainer.__new__(runtime.AWMTrainer)
    trainer.config = SimpleNamespace(awm=SimpleNamespace(plan=str(tmp_path / "plan.json")))
    trainer.train_dataloader = [[task.task_instance_id]]
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.async_rollout_manager = SimpleNamespace(llm_client=None)
    trainer.actor_rollout_wg = SimpleNamespace(parameter_probe=lambda: [])
    trainer.checkpoint_manager = SimpleNamespace(
        update_weights=lambda step: None, sleep_replicas=lambda: None, abort_replicas=lambda: None)
    trainer._save_group_checkpoint = lambda *args: None
    trainer._shutdown_dump_executor = lambda: None
    monkeypatch.setattr(runtime, "preflight", lambda *args: None)
    monkeypatch.setattr(runtime, "read_group", lambda *args, **kwargs: ([], [0, 0], [0, 0]))
    monkeypatch.setattr(pilot, "resource_check", lambda *args, **kwargs: None)
    monkeypatch.setattr(pilot, "group_health", lambda *args: {"api_fault": False, "quarantine": False})
    finishes = []
    class Tracker:
        mode, url = "disabled", None
        def __init__(self, *args, **kwargs): pass
        def log(self, *args, **kwargs): pass
        def finish(self, failed): finishes.append(failed)
    monkeypatch.setattr("harness_awm.tracking.SwanLabTracker", Tracker)
    archive_started, next_rollout = threading.Event(), threading.Event()
    original_archive = pilot.archive_group
    completed = []
    def archive(folder, dest):
        archive_started.set()
        assert next_rollout.wait(5), "archival blocked the next rollout"
        assert not (tmp_path / "summary.json").exists()
        if failure == "archive":
            raise OSError("archive verification failed")
        result = original_archive(folder, dest)
        completed.append(folder.name)
        return result
    monkeypatch.setattr(pilot, "archive_group", archive)
    async def collect(*args, **kwargs):
        folder = args[4]
        folder.mkdir(parents=True)
        if folder.name == "group-000001":
            assert archive_started.wait(5)
            assert not completed
            assert read_json(tmp_path / "progress.json")["groups"] == 1
            assert not (tmp_path / "archives/group-000000.tar.manifest.json").exists()
            next_rollout.set()
            if failure == "rollout":
                raise RuntimeError("rollout failed")
    monkeypatch.setattr(runtime, "collect_group", collect)
    if failure:
        with pytest.raises((OSError, RuntimeError), match=f"{failure} .*failed"):
            trainer.fit()
    else:
        trainer.fit()
    report = read_json(tmp_path / "summary.json")
    assert report["failed"] is (failure is not None)
    assert finishes == [failure is not None]
    assert bool(report["archive_error"]) is (failure == "archive")
    assert report["optimizer_steps"] == 0
    if failure != "archive":
        assert len(completed) == (1 if failure == "rollout" else 2)
        assert all(read_json(row["archive_manifest"])["files_verified"] > 0 for row in report["groups"])
    else:
        assert not list((tmp_path / "archives").glob("*.manifest.json"))
    assert (tmp_path / "groups/group-000000/update.json").exists()
