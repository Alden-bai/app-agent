"""Turn-wise, on-policy GRPO for AWM. Importing this module never trains.

One collected group per optimizer step, valid-subset statistics and sequence-mean loss,
temperature 1, no token filtering. Each turn uses its actual sampled context.
The first implementation is single-process; it rejects distributed launch.
"""
import argparse
import asyncio
import json
import inspect
import math
import os
from pathlib import Path

from ..backend import Budget, TrainingActorBackend
from ..contracts import RunConfig, TaskSpec
from ..io import digest, file_hash, read_json, verify_seal, write_json
from ..judge_inputs import sources_from_manifest
from ..reward_policy import VERSION as REWARD_VERSION, admission_for
from ..runner import EpisodeRunner
from .rewards import InvalidRewardGroup, admitted_group, admitted_reward, population_advantages


def preflight(tasks, config, group_size=4):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("turnwise_grpo_currently_requires_one_process")
    if group_size < 2 or config.actor != "training" or config.judge_protocol != "rubric_v2":
        raise ValueError("GRPO requires training actor, Judge V2 and group_size >= 2")
    isolated_development = config.isolation in {"landlock", "private-root"} and config.purpose == "development"
    if (config.isolation != "strict" and not isolated_development) or config.judge != "api":
        raise ValueError("GRPO requires admitted worker isolation and a real Judge")
    if config.purpose == "evaluation":
        raise ValueError("GRPO cannot use scoring-only evaluation configuration")
    if not config.judge_inputs_manifest or not tasks:
        raise ValueError("GRPO requires source manifest and tasks")
    if ((config.max_physical_requests is not None and config.max_physical_requests < 1)
            or (config.max_total_tokens is not None and config.max_total_tokens < 1)):
        raise ValueError("GRPO requires an explicit positive Judge request/token budget")
    from ..reward_policy import AdmissionManifest
    from ..judge_runtime import config_hash
    if not config.reward_admissions:
        raise ValueError("reward_admission_missing")
    manifest = AdmissionManifest.model_validate(read_json(config.reward_admissions))
    judge_hash = config_hash(config)
    shared_verified = {}
    from ..judge_inputs import load_json
    source_root = Path(config.judge_inputs_manifest).resolve()
    source_index = load_json(source_root)
    if source_index.get("schema_version") != "awm-judge-input-manifest-v2":
        raise ValueError("unsupported_judge_input_manifest")
    for task in tasks:
        if task.split != "train":
            raise ValueError("GRPO only accepts train split")
        sources = {}
        for name in ("original", "rubric"):
            ref = source_index["tasks"][task.task_instance_id][name]
            path = (source_root.parent / ref["path"]).resolve()
            if file_hash(path) != ref["sha256"]:
                raise ValueError("judge_manifest_source_hash_mismatch:" + name)
            sources[name] = load_json(path)
        admission_for(task, config, **sources, _manifest=manifest,
                      _judge_hash=judge_hash, _shared_verified=shared_verified)
    return {"checked_tasks": len(tasks), "group_size": group_size,
            "reward_policy": config.reward_policy, "std": "population",
            "loss_normalization": "mean of per-trajectory token means", "training_started": False}


def turn_logprobs(model, turn):
    import torch
    ids = turn["prompt_ids"] + turn["completion_ids"]
    tokens = torch.tensor([ids], device=next(model.parameters()).device)
    count = len(turn["completion_ids"])
    options = {"logits_to_keep": count + 1} if "logits_to_keep" in inspect.signature(model.forward).parameters else {}
    logits = model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False, **options).logits
    selected = logits[0, -(count + 1):-1].float().log_softmax(-1)
    return selected.gather(-1, tokens[0, -count:].unsqueeze(-1)).squeeze(-1)


def clipped_objective(current, old, advantage, *, clip=0.2, reference=None, beta=0.0):
    """Per-action-token PPO objective plus optional unbiased sampled KL."""
    import torch
    ratio = (current - old).exp()
    loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    if beta:
        if reference is None:
            raise ValueError("positive beta requires frozen reference model")
        delta = reference - current
        loss = loss + beta * (delta.exp() - delta - 1)
    return loss


class GRPOEngine:
    def __init__(self, model, tokenizer, optimizer, runner, output, *, group_size=4,
                 policy_version="initial", clip=0.2, beta=0.0, reference_model=None,
                 epsilon=1e-6, max_grad_norm=1.0, test_only=False):
        if (group_size < 2 or not 0 < clip < 1 or not math.isfinite(beta) or beta < 0
                or not math.isfinite(epsilon) or epsilon <= 0 or not math.isfinite(max_grad_norm)
                or max_grad_norm <= 0 or not policy_version):
            raise ValueError("invalid_GRPO_settings")
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("turnwise_grpo_currently_requires_one_process")
        if reference_model is model or (beta and reference_model is None):
            raise ValueError("positive beta requires separate frozen reference model")
        model_ids = {id(p) for p in model.parameters() if p.requires_grad}
        optimizer_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
        if model_ids != optimizer_ids:
            raise ValueError("optimizer_must_own_exactly_trainable_policy_parameters")
        if reference_model is not None:
            if {id(p) for p in reference_model.parameters()} & {id(p) for p in model.parameters()}:
                raise ValueError("reference_must_not_share_policy_parameters")
            reference_model.requires_grad_(False).eval()
        self.model, self.tokenizer, self.optimizer, self.runner = model, tokenizer, optimizer, runner
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.group_size, self.base_version, self.step = group_size, policy_version, 0
        self.clip, self.beta, self.reference = clip, beta, reference_model
        self.epsilon, self.max_grad_norm, self.test_only = epsilon, max_grad_norm, test_only
        self.group_index = 0
        self.pending = None
        write_json(self.output / "training_config.json", {"config": runner.config.model_dump(),
            "group_size": group_size, "policy_version": policy_version, "clip": clip, "beta": beta,
            "epsilon": epsilon, "max_grad_norm": max_grad_norm, "test_only": test_only,
            "std": "population", "loss_normalization": "sequence_mean"})

    @property
    def policy_version(self):
        return f"{self.base_version}/step-{self.step}"

    async def collect(self, task):
        if self.pending is not None:
            raise ValueError("settle_pending_group_before_collecting")
        if not self.test_only:
            preflight([task], self.runner.config, self.group_size)
        config = self.runner.config
        if config.actor != "training" or config.judge_protocol != "rubric_v2":
            raise ValueError("training actor and Judge V2 required")
        group = self.output / f"group-{self.group_index:06d}"
        group.mkdir()
        self.group_index += 1
        candidates = []
        # All K candidates share weights and settings but get distinct environment
        # directories and independently sampled completions. Never resample a null.
        for i in range(self.group_size):
            folder = group / f"candidate-{i:03d}"
            actor = TrainingActorBackend(self.model, self.tokenizer, policy_version=self.policy_version,
                max_output_tokens=config.max_output_tokens, context_limit=config.context_limit, test_only=self.test_only)
            result = await self.runner.run(task, folder, f"{group.name}/{folder.name}", actor)
            candidates.append({"path": folder.name, "result_hash": digest(result)})
        manifest = {"task": task.model_dump(), "config_hash": digest(config.model_dump()),
                    "policy_version": self.policy_version, "test_only": self.test_only, "candidates": candidates}
        write_json(group / "group.json", manifest)
        self.pending = (group, digest(manifest))
        return group

    def update(self, group, *, diagnostic_rewards=None):
        import torch
        group = Path(group).resolve()
        manifest = read_json(group / "group.json")
        if self.pending != (group, digest(manifest)) or manifest["policy_version"] != self.policy_version:
            raise ValueError("stale_foreign_or_consumed_group")
        if digest(self.runner.config.model_dump()) != manifest["config_hash"]:
            raise ValueError("run_config_changed")
        if diagnostic_rewards is not None and not self.test_only:
            raise ValueError("diagnostic_rewards_are_test_only")
        if len(manifest["candidates"]) != self.group_size:
            raise InvalidRewardGroup("incomplete candidate group")
        excluded, included = [], []

        def finish(status, **values):
            report = {"status": status, "policy_version": self.policy_version, "optimizer_step": self.step,
                      "test_only": self.test_only, "included_candidates": included,
                      "excluded_candidates": excluded, **values}
            write_json(group / "update.json", report)
            self.pending = None
            return report

        if not self.test_only:
            try:
                task = TaskSpec.model_validate(manifest["task"])
                preflight([task], self.runner.config, self.group_size)
                admission = admission_for(task, self.runner.config,
                    **sources_from_manifest(task, self.runner.config.judge_inputs_manifest))
                current_admission_id = digest(admission.model_dump())
            except ValueError as exc:
                return finish("held", reason=str(exc))
        results, trajectories = [], []
        for candidate in manifest["candidates"]:
            folder = group / candidate["path"]
            sealed = verify_seal(folder)
            result = read_json(folder / "episode_result.json")
            if digest(result) != candidate["result_hash"]:
                raise ValueError("episode_result_changed")
            if result["task_instance_id"] != manifest["task"]["task_instance_id"]:
                raise ValueError("candidate_task_mismatch")
            if not self.test_only and result.get("reward_admission_id") != current_admission_id:
                if result.get("reward_admission_id") is not None:
                    return finish("held", reason="reward_admission_changed_or_missing")
                excluded.append({"path": candidate["path"], "reason": "reward_admission_changed_or_missing"})
                continue
            if not self.test_only:
                try:
                    admitted_reward(result)
                except InvalidRewardGroup as exc:
                    excluded.append({"path": candidate["path"], "reason": str(exc)})
                    continue
            turns = []
            for path in sorted((folder / "model_turns").glob("*/policy_turn.json")):
                if str(path.relative_to(folder)) not in sealed:
                    raise ValueError("unsealed_policy_turn")
                turn = read_json(path)
                prompt, completion, old = turn["prompt_ids"], turn["completion_ids"], turn["old_logprobs"]
                if (turn["policy_version"] != self.policy_version or turn["test_only"] != self.test_only
                        or not prompt or not completion or len(completion) != len(old)
                        or any(type(t) is not int or t < 0 for t in prompt + completion)
                        or any(type(x) not in (int, float) or not math.isfinite(x) or x > 0 for x in old)
                        or turn["action_mask"] != [0] * len(prompt) + [1] * len(completion)
                        or turn["sampling"] != {"temperature": 1.0, "top_k": 0, "top_p": 1.0}
                        or turn["messages_hash"] != digest(read_json(path.parent / "input.json")["messages"])
                        or self.tokenizer.decode(completion, skip_special_tokens=True) != read_json(path.parent / "output.json")["content"]):
                    raise ValueError("invalid_or_stale_policy_turn")
                turns.append(turn)
            output_count = len(list((folder / "model_turns").glob("*/output.json")))
            if not turns or len(turns) != output_count:
                excluded.append({"path": candidate["path"], "reason": "missing_policy_turns"})
                continue
            results.append(result)
            trajectories.append(turns)
            included.append(candidate["path"])
        if len(results) < 2:
            return finish("held", reason="fewer_than_two_admitted_candidates")
        try:
            rewards = admitted_group(results, len(results))
        except InvalidRewardGroup as exc:
            if diagnostic_rewards is None:
                return finish("held", reason=str(exc))
            if (not all(r["test_only"] is True and r["execution_integrity"] == "valid"
                        and r["terminal_state_status"] == "trusted" for r in results)
                    or len(diagnostic_rewards) != self.group_size
                    or any(type(r) not in (int, float) or r not in (0, 1) for r in diagnostic_rewards)):
                raise ValueError("invalid_diagnostic_group")
            rewards = list(diagnostic_rewards)
        advantages = population_advantages(rewards, self.epsilon, std_floor=0.10)
        if not any(advantages) and self.beta == 0:
            return finish("zero_advantage", rewards=rewards, advantages=advantages)
        self.model.eval()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        try:
            for turns, advantage in zip(trajectories, advantages):
                denominator = len(trajectories) * sum(len(t["completion_ids"]) for t in turns)
                for turn in turns:
                    current = turn_logprobs(self.model, turn)
                    old = current.new_tensor(turn["old_logprobs"])
                    # Reject hidden weight/tokenizer/sampling changes, before any step.
                    if not torch.allclose(current.detach(), old, atol=5e-3, rtol=1e-3):
                        raise ValueError("sampling_policy_logprob_mismatch")
                    reference = None
                    if self.beta:
                        with torch.no_grad():
                            reference = turn_logprobs(self.reference, turn).to(current.device)
                    loss = clipped_objective(current, old, advantage, clip=self.clip,
                                             reference=reference, beta=self.beta).sum() / denominator
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite_GRPO_loss")
                    loss.backward()
                    total_loss += float(loss.detach())
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm, error_if_nonfinite=True)
            self.optimizer.step()
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True)
            raise
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        return finish("updated", rewards=rewards, advantages=advantages, loss=total_loss,
                      grad_norm=float(grad_norm), diagnostic_rewards=diagnostic_rewards is not None)


async def train(model, tokenizer, optimizer, tasks, config, output, *, groups, group_size=4,
                policy_version="initial", beta=0.0, reference_model=None, launch_metadata=None):
    """Explicit opt-in entry point. Does not run until called by the operator."""
    preflight(tasks, config, group_size)
    if groups < 1:
        raise ValueError("groups must be positive")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if launch_metadata is not None:
        write_json(output / "launch.json", launch_metadata)
    budget = Budget(output / "budget.sqlite", config.max_physical_requests, config.max_total_tokens)
    runner = EpisodeRunner(config, budget, asyncio.Semaphore(config.judge_concurrency))
    engine = GRPOEngine(model, tokenizer, optimizer, runner, output / "groups", group_size=group_size,
                        policy_version=policy_version, beta=beta, reference_model=reference_model)
    reports = []
    for index in range(groups):
        group = await engine.collect(tasks[index % len(tasks)])
        reports.append(engine.update(group))
    model.save_pretrained(output / "final")
    tokenizer.save_pretrained(output / "final")
    import torch
    torch.save({"optimizer": optimizer.state_dict(), "step": engine.step,
                "policy_version": engine.policy_version}, output / "optimizer.pt")
    write_json(output / "summary.json", {"groups": reports, "budget": budget.summary()})
    return reports


def main():
    from dotenv import load_dotenv
    from ..contracts import ROOT
    from .tasks import load_tasks, select_tasks
    from .local import checkpoint_identity, load_checkpoint
    load_dotenv(ROOT / ".env", override=False)
    p = argparse.ArgumentParser(description="AWM GRPO: preflight by default, --execute starts local training.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--task", action="append", help="Optional exact task ID; repeat to select a pilot subset")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--model", type=Path, help="SFT checkpoint or base model directory")
    p.add_argument("--output", type=Path)
    p.add_argument("--groups", type=int, help="Number of groups to collect, including held/zero-advantage groups")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--reference-model", type=Path)
    p.add_argument("--seed", type=int, default=20261106)
    args = p.parse_args()
    config = RunConfig.model_validate(read_json(args.config))
    config = config.model_copy(update={"seed": args.seed})
    tasks, task_manifest, sources = load_tasks(args.tasks, "grpo_train")
    tasks = select_tasks(tasks, args.task)
    if sources:
        config = config.model_copy(update={"judge_inputs_manifest": sources})
    checked = preflight(tasks, config, args.group_size)
    if not args.execute:
        print(json.dumps(checked, indent=2))
        return
    if not args.model or not args.output or not args.groups or args.groups < 1:
        p.error("--execute requires --model, --output and positive --groups")
    if args.output.exists():
        p.error("output exists; choose a new directory")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or not math.isfinite(args.beta) or args.beta < 0:
        p.error("invalid learning rate or beta")
    if args.beta and not args.reference_model:
        p.error("positive --beta requires --reference-model")
    from transformers import set_seed
    import torch
    set_seed(args.seed)
    identity = checkpoint_identity(args.model)
    model, tokenizer = load_checkpoint(args.model, device=args.device, dtype=args.dtype, trainable=True)
    reference = None
    reference_identity = None
    if args.reference_model:
        reference_identity = checkpoint_identity(args.reference_model)
        reference, reference_tokenizer = load_checkpoint(args.reference_model, device=args.device, dtype=args.dtype)
        if reference_tokenizer.get_vocab() != tokenizer.get_vocab() or reference.config.vocab_size != model.config.vocab_size:
            p.error("reference and policy token vocabularies differ")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    import random
    random.Random(args.seed).shuffle(tasks)
    reports = asyncio.run(train(model, tokenizer, optimizer, tasks, config, args.output,
        groups=args.groups, group_size=args.group_size, policy_version=identity["sha256"], beta=args.beta,
        reference_model=reference, launch_metadata={"checkpoint": identity, "reference_checkpoint": reference_identity,
            "task_manifest": task_manifest, "seed": args.seed, "device": args.device, "dtype": args.dtype,
            "learning_rate": args.learning_rate, "groups": args.groups,
            "task_order": [t.task_instance_id for t in tasks], "task_sampling": "seeded_shuffle_then_cycle"}))
    from collections import Counter
    print(json.dumps({"output": str(args.output.resolve()), "groups": len(reports),
                      "statuses": dict(Counter(r["status"] for r in reports))}, indent=2))


if __name__ == "__main__":
    main()
