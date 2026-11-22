"""Offline reward admission and real tiny-model GRPO integration contracts."""
import asyncio
import copy

import pytest

from harness_awm.backend import Budget, TrainingActorBackend
from harness_awm.contracts import EpisodeResult, RunConfig
from harness_awm.evaluation import evaluate_episode
from harness_awm.io import digest, file_hash, read_json, write_json
from harness_awm.judge_runtime import config_hash
from harness_awm.judge_v2 import JudgeResult, aggregate
from harness_awm.reward_policy import VERSION, map_reward
from harness_awm.runner import EpisodeRunner
from harness_awm.training.grpo import GRPOEngine, clipped_objective, preflight, turn_logprobs
from harness_support import fixture
from test_harness_v2 import MemoryEnv, TOOLS
from test_judge_v2 import assessment, rubric_fixture, validated, Responder


@pytest.fixture(autouse=True)
def cpu_delta_rule_reference(monkeypatch):
    # These tiny-model semantic tests run on CPU even when CUDA FLA is installed.
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    except ImportError:
        return
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        function = getattr(qwen, name)
        monkeypatch.setattr(qwen, name, getattr(function, "__wrapped__", function))


def result_for(states=("pass", "pass"), **kwargs):
    return JudgeResult(task_id="harness_fixture:0", task_instance_id="fixture:0", evaluation_id="evaluation",
        versions={}, usage={}, **aggregate(validated(assessment(states, **kwargs)))).model_dump()


@pytest.mark.parametrize("states,expected", [(("pass", "fail"), 1), (("fail", "pass"), 0), (("unknown", "pass"), 0.3)])
def test_reward_truth_table_and_quality_exclusion(states, expected):
    r = map_reward(result_for(states), approved=True, execution_valid=True, test_only=False)
    assert r == {"reward": expected, "reward_valid": expected is not None}


@pytest.mark.parametrize("gate", ["approved", "execution_valid", "test_only"])
@pytest.mark.parametrize("states", [("pass", "pass"), ("unknown", "pass")])
def test_reward_gates_fail_closed(gate, states):
    flags = dict(approved=True, execution_valid=True, test_only=False)
    flags[gate] = not flags[gate]
    assert map_reward(result_for(states), **flags) == {"reward": None, "reward_valid": False}


@pytest.mark.parametrize("change", [{"test_only": True}, {"evaluation_status": "evaluation_error"},
    {"evaluation_valid": False}, {"sample_validity": {"status": "unknown"}},
    {"errors": ["conflicting evidence"]}, {"rubric_issues": [{"scope": "required"}]},
    {"task_status": "provisional_pass"}])
def test_invalid_results_cannot_supply_reward(change):
    assert not map_reward({**result_for(), **change}, approved=True, execution_valid=True, test_only=False)["reward_valid"]


def test_result_contract_rejects_unadmitted_reward():
    with pytest.raises(ValueError, match="invalid_admitted_reward"):
        JudgeResult.model_validate({**result_for(), "reward": 1, "reward_valid": True})
    with pytest.raises(ValueError, match="invalid_reward_must_be_null"):
        JudgeResult.model_validate({**result_for(), "reward": 0})
    with pytest.raises(ValueError, match="reward_must_be"):
        JudgeResult.model_validate({**result_for(), "reward": True})


@pytest.mark.parametrize("kind,expected", [("semantic_ambiguity", 0.3), ("evidence_gap", 0.3),
    ("evidence_conflict", 0.3), ("evidence_error", None), ("budget_exhausted", None), ("not_investigated", None)])
def test_discount_only_for_genuine_uncertainty(kind, expected):
    result = result_for(("unknown", "pass"), kind=kind)
    mapped = map_reward(result, approved=True, execution_valid=True, test_only=False)
    assert mapped == {"reward": expected, "reward_valid": expected is not None}
    settled = JudgeResult.model_validate({**result, **mapped, "reward_policy": VERSION,
                                         "reward_admission_id": "test-admission"})
    assert JudgeResult.model_validate_json(settled.model_dump_json()).reward == expected


def test_still_resolvable_unknown_gets_no_discount():
    answer = assessment(("unknown", "pass"))
    answer["items"][0]["unknown"]["can_resolve_with_tools"] = True
    result = aggregate(validated(answer))
    assert result["evaluation_status"] == "evaluation_error"
    assert map_reward({**result, "test_only": False}, approved=True, execution_valid=True,
                      test_only=False) == {"reward": None, "reward_valid": False}


@pytest.mark.parametrize("value", [True, "0.3", 0.5, float("nan"), float("inf")])
def test_reward_value_contract_rejects_coercion_and_unconfigured_values(value):
    with pytest.raises(ValueError, match="reward_must_be"):
        JudgeResult.model_validate({**result_for(), "reward": value})
    with pytest.raises(ValueError, match="reward_must_be"):
        EpisodeResult(rollout_id="r", task_instance_id="t", termination_reason="finished", reward=value)


def test_discount_cannot_be_attached_to_pass_failure_or_old_policy():
    for states, policy in [(("pass", "pass"), VERSION), (("fail", "pass"), VERSION),
                           (("unknown", "pass"), "judge-v2-terminal-binary-v1")]:
        with pytest.raises(ValueError, match="invalid_admitted_reward"):
            JudgeResult.model_validate({**result_for(states), "reward": 0.3, "reward_valid": True,
                                       "reward_policy": policy, "reward_admission_id": "test"})


def setup_sources(root):
    root.mkdir(parents=True, exist_ok=True)
    task = fixture(root / "assets").model_copy(update={"split": "train", "tool_schema_hash": digest(TOOLS)})
    original, rubric = rubric_fixture()
    write_json(root / "original.json", original)
    write_json(root / "rubric.json", rubric)
    write_json(root / "sources.json", {"schema_version": "awm-judge-input-manifest-v2", "tasks": {
        task.task_instance_id: {name: {"path": name + ".json", "sha256": file_hash(root / (name + ".json"))}
                                for name in ("original", "rubric")}}})
    config = RunConfig(protocol_id="awm-single-json-v2", actor="training", judge="api", judge_protocol="rubric_v2",
        judge_inputs_manifest=str(root / "sources.json"), reward_admissions=str(root / "admissions.json"),
        judge_v2={"review_all_decisions": False}, max_turns=2, max_output_tokens=3, final_output_tokens=2,
        context_limit=32768, episode_token_limit=100000, max_physical_requests=8, max_total_tokens=1000000)
    entry = {"status": "admitted", "calibration_ref": "synthetic-test-fixture-only", "policy_version": VERSION,
        "task_instance_id": task.task_instance_id, "original_hash": digest(original), "rubric_hash": digest(rubric),
        "initial_state_hash": task.initial_state_hash, "environment_code_hash": task.environment_code_hash,
        "tool_schema_hash": task.tool_schema_hash, "judge_config_hash": config_hash(config)}
    write_json(config.reward_admissions, {"schema_version": "awm-reward-admissions-v1", "tasks": {task.task_instance_id: entry}})
    return task, config, original, rubric


def tiny_policy(architecture="gpt2"):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    torch.set_num_threads(1)
    torch.manual_seed(714)
    vocab = {s: i for i, s in enumerate(["<pad>", "<eos>", "<unk>", "user", "assistant", "system", "value", "done", "80", ":"])}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    if architecture == "qwen3.5":
        from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
        config = Qwen3_5TextConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
            layer_types=["linear_attention", "full_attention"],
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                             "partial_rotary_factor": 1.0, "mrope_section": [2, 3, 3]},
            pad_token_id=0, eos_token_id=1, tie_word_embeddings=True)
        model = Qwen3_5ForCausalLM(config)
    else:
        model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_positions=4096, n_embd=16, n_layer=1,
            n_head=2, eos_token_id=1, pad_token_id=0, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))
    return model, tokenizer


def engine_fixture(tmp_path, monkeypatch, *, test_only=True, states=("pass", "fail"), architecture="gpt2", **engine_options):
    import torch
    task, config, original, rubric = setup_sources(tmp_path)
    responder = Responder([assessment((state, "pass"), refs=["E00002"])
                           for state in states for _ in range(2 if state == "unknown" else 1)])
    monkeypatch.setattr("harness_awm.judge_runtime.api_request", responder)
    model, tokenizer = tiny_policy(architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    runner = EpisodeRunner(config, Budget(tmp_path / "budget.db", 8, 1000000), asyncio.Semaphore(1), env_factory=MemoryEnv)
    engine = GRPOEngine(model, tokenizer, optimizer, runner, tmp_path / "training", group_size=len(states),
                        test_only=test_only, **engine_options)
    return engine, task, original, rubric, responder


@pytest.mark.parametrize("architecture", ["gpt2", "qwen3.5"])
def test_real_sampling_judge_mapping_and_optimizer_chain(tmp_path, monkeypatch, architecture):
    import torch
    # Memory environment and synthetic Judge transport; no API or formal model.
    engine, task, _, _, responder = engine_fixture(tmp_path, monkeypatch, test_only=False, architecture=architecture)
    before = {n: p.detach().clone() for n, p in engine.model.named_parameters()}
    group = asyncio.run(engine.collect(task))
    results = [read_json(p) for p in sorted(group.glob("candidate-*/episode_result.json"))]
    assert [r["reward"] for r in results] == [1, 0]
    assert all(r["reward_admission_id"] and r["reward_policy"] == VERSION for r in results)
    assert responder.calls == 2
    turns = list(group.glob("candidate-*/model_turns/*/policy_turn.json"))
    assert len(turns) == 4
    assert len({read_json(p)["messages_hash"] for p in turns}) >= 2
    report = engine.update(group)
    assert report["status"] == "updated" and report["optimizer_step"] == 1
    assert report["advantages"] == pytest.approx([1, -1], abs=3e-6)
    assert any(not torch.equal(p, before[n]) for n, p in engine.model.named_parameters())
    assert all(p.grad is None for p in engine.model.parameters())
    with pytest.raises(ValueError, match="consumed"):
        engine.update(group)


def test_test_only_never_receives_business_reward_but_allows_diagnostic_step(tmp_path, monkeypatch):
    engine, task, _, _, _ = engine_fixture(tmp_path, monkeypatch)
    group = asyncio.run(engine.collect(task))
    assert all(read_json(p)["reward"] is None for p in group.glob("candidate-*/episode_result.json"))
    report = engine.update(group, diagnostic_rewards=[1, 0])
    assert report["test_only"] and report["diagnostic_rewards"] and report["status"] == "updated"


@pytest.mark.parametrize("states,status", [(("pass", "pass"), "zero_advantage"), (("fail", "fail"), "zero_advantage"),
                                           (("unknown", "unknown"), "zero_advantage")])
def test_equal_reward_groups_do_not_change_weights(tmp_path, monkeypatch, states, status):
    import torch
    engine, task, _, _, _ = engine_fixture(tmp_path, monkeypatch, test_only=False, states=states)
    before = {n: p.detach().clone() for n, p in engine.model.named_parameters()}
    group = asyncio.run(engine.collect(task))
    assert engine.update(group)["status"] == status
    assert not engine.optimizer.state
    assert all(torch.equal(p, before[n]) for n, p in engine.model.named_parameters())


def test_discount_survives_judge_episode_cache_and_grpo_update(tmp_path, monkeypatch):
    import torch
    engine, task, _, _, responder = engine_fixture(tmp_path, monkeypatch, test_only=False,
                                                  states=("pass", "unknown", "fail"))
    before = {n: p.detach().clone() for n, p in engine.model.named_parameters()}
    group = asyncio.run(engine.collect(task))
    results = [EpisodeResult.model_validate_json(p.read_text())
               for p in sorted(group.glob("candidate-*/episode_result.json"))]
    assert [r.reward for r in results] == [1, 0.3, 0]
    assert results[1].business_label == "unknown" and results[1].task_status == "provisional_pass"
    assert all(r.reward_valid and r.reward_policy == VERSION for r in results)
    assert responder.calls == 4  # The provisional candidate includes mandatory review.
    cached = asyncio.run(evaluate_episode(group / "candidate-001", task, engine.runner.config,
                                         engine.runner.budget, engine.runner.judge_semaphore))
    assert cached["reward"] == 0.3 and responder.calls == 4
    report = engine.update(group)
    assert report["status"] == "updated" and report["optimizer_step"] == 1
    assert report["rewards"] == [1., 0.3, 0.]
    assert report["advantages"][0] > report["advantages"][1] > report["advantages"][2]
    assert any(not torch.equal(p, before[n]) for n, p in engine.model.named_parameters())


@pytest.mark.parametrize("kind", ["evidence_error", "budget_exhausted", "not_investigated"])
def test_unsettled_unknown_still_holds_group(tmp_path, monkeypatch, kind):
    import torch
    engine, task, _, _, _ = engine_fixture(tmp_path, monkeypatch, test_only=False)
    unknown = assessment(("unknown", "pass"), refs=["E00002"], kind=kind)
    monkeypatch.setattr("harness_awm.judge_runtime.api_request",
                        Responder([assessment(refs=["E00002"]), unknown, unknown]))
    before = {n: p.detach().clone() for n, p in engine.model.named_parameters()}
    group = asyncio.run(engine.collect(task))
    result = read_json(group / "candidate-001/episode_result.json")
    assert result["reward"] is None and not result["reward_valid"]
    assert result["evaluation_status"] == "evaluation_error"
    assert engine.update(group)["status"] == "held"
    assert not engine.optimizer.state
    assert all(torch.equal(p, before[n]) for n, p in engine.model.named_parameters())


@pytest.mark.parametrize("policy", [None, "judge-v2-terminal-binary-v1"])
def test_old_or_unversioned_admission_cannot_silently_enable_discount(tmp_path, policy):
    task, config, _, _ = setup_sources(tmp_path)
    manifest = read_json(config.reward_admissions)
    entry = manifest["tasks"][task.task_instance_id]
    if policy is None:
        entry.pop("policy_version")
    else:
        entry["policy_version"] = policy
    write_json(config.reward_admissions, manifest)
    with pytest.raises(ValueError, match="policy_version"):
        preflight([task], config)


@pytest.mark.parametrize("state", ["pass", "unknown"])
def test_revoked_admission_invalidates_cached_judge_without_api_retry(tmp_path, monkeypatch, state):
    engine, task, _, _, responder = engine_fixture(tmp_path, monkeypatch, test_only=False, states=(state, "fail"))
    group = asyncio.run(engine.collect(task))
    path = engine.runner.config.reward_admissions
    manifest = read_json(path)
    manifest["tasks"][task.task_instance_id]["status"] = "revoked"
    write_json(path, manifest)
    cached = asyncio.run(evaluate_episode(group / "candidate-000", task, engine.runner.config,
                                         engine.runner.budget, engine.runner.judge_semaphore))
    assert cached["business_label"] == state and cached["reward"] is None
    assert "revoked" in cached["reward_exclusion"] and responder.calls == (3 if state == "unknown" else 2)
    assert engine.update(group)["status"] == "held"
    assert not engine.optimizer.state


@pytest.mark.parametrize("field", ["original_hash", "rubric_hash", "initial_state_hash", "environment_code_hash",
                                    "tool_schema_hash", "judge_config_hash", "task_instance_id"])
def test_admission_binds_every_frozen_component(tmp_path, field):
    task, config, _, _ = setup_sources(tmp_path)
    data = read_json(config.reward_admissions)
    data["tasks"][task.task_instance_id][field] = "different"
    write_json(config.reward_admissions, data)
    with pytest.raises(ValueError, match="binding_mismatch"):
        preflight([task], config)


def test_train_split_and_budget_are_required(tmp_path):
    task, config, _, _ = setup_sources(tmp_path)
    assert preflight([task], config)["training_started"] is False
    with pytest.raises(ValueError, match="train split"):
        preflight([task.model_copy(update={"split": "test"})], config)
    with pytest.raises(ValueError, match="budget"):
        preflight([task], config.model_copy(update={"max_physical_requests": 0}))


def test_tampered_turn_and_policy_drift_are_rejected(tmp_path, monkeypatch):
    import torch
    engine, task, _, _, _ = engine_fixture(tmp_path, monkeypatch, test_only=False)
    group = asyncio.run(engine.collect(task))
    with torch.no_grad():
        engine.model.transformer.wte.weight[3, 0].add_(3)
    with pytest.raises(ValueError, match="logprob_mismatch"):
        engine.update(group)
    assert not engine.optimizer.state
    path = next(group.glob("candidate-*/model_turns/*/policy_turn.json"))
    data = read_json(path)
    data["old_logprobs"][0] -= 1
    write_json(path, data)
    with pytest.raises(ValueError, match="evidence_hash_mismatch"):
        engine.update(group)


def test_clipping_and_kl_math():
    import torch
    current = torch.tensor([1.5, .5]).log()
    old = torch.zeros(2)
    assert clipped_objective(current, old, 1).tolist() == pytest.approx([-1.2, -.5])
    assert clipped_objective(current, old, -1).tolist() == pytest.approx([1.5, .8])
    kl = clipped_objective(current, old, 0, reference=old, beta=.1)
    assert torch.all(kl >= 0)
    with pytest.raises(ValueError, match="reference"):
        clipped_objective(current, old, 0, beta=.1)


def test_changed_admission_is_not_reused_for_old_rollouts(tmp_path, monkeypatch):
    engine, task, _, _, _ = engine_fixture(tmp_path, monkeypatch, test_only=False)
    group = asyncio.run(engine.collect(task))
    path = engine.runner.config.reward_admissions
    manifest = read_json(path)
    manifest["tasks"][task.task_instance_id]["calibration_ref"] = "new-calibration"
    write_json(path, manifest)
    assert engine.update(group)["reason"] == "reward_admission_changed_or_missing"
    assert not engine.optimizer.state


def test_frozen_reference_kl_update_and_diagnostic_guard(tmp_path, monkeypatch):
    import torch
    engine, task, _, _, _ = engine_fixture(tmp_path, monkeypatch, test_only=False)
    reference = copy.deepcopy(engine.model).requires_grad_(False).eval()
    engine.reference, engine.beta = reference, .01
    before = {n: p.clone() for n, p in reference.named_parameters()}
    group = asyncio.run(engine.collect(task))
    with pytest.raises(ValueError, match="test_only"):
        engine.update(group, diagnostic_rewards=[1, 0])
    assert engine.update(group)["status"] == "updated"
    assert all(p.grad is None and torch.equal(p, before[n]) for n, p in reference.named_parameters())


def test_distributed_launch_is_rejected_before_sampling(tmp_path, monkeypatch):
    task, config, _, _ = setup_sources(tmp_path)
    monkeypatch.setenv("WORLD_SIZE", "3")
    with pytest.raises(ValueError, match="one_process"):
        preflight([task], config)


@pytest.mark.parametrize("architecture", ["gpt2", "qwen3.5"])
def test_compacted_prompts_use_their_own_logprobs(tmp_path, architecture):
    import torch
    model, tokenizer = tiny_policy(architecture)
    actor = TrainingActorBackend(model, tokenizer, policy_version="v1", max_output_tokens=3, test_only=True)
    async def sample():
        await actor.generate([{"role": "user", "content": "value 80"}], tmp_path / "1", 1)
        await actor.generate([{"role": "user", "content": "done"}], tmp_path / "2", 2)
    asyncio.run(sample())
    assert actor.records[0]["prompt_ids"] != actor.records[1]["prompt_ids"]
    for turn in actor.records:
        with torch.no_grad():
            assert torch.allclose(turn_logprobs(model, turn), torch.tensor(turn["old_logprobs"]), atol=1e-5)
        assert turn["action_mask"][:len(turn["prompt_ids"])] == [0] * len(turn["prompt_ids"])
