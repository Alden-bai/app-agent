"""Concurrent AWM environments using verl's shared inference client."""
import asyncio
import math
from collections.abc import Mapping
from pathlib import Path

from ...io import digest, verify_seal, write_json
from ..actor import STOP_POLICY, TrainingActorBackend


class VerlActorBackend(TrainingActorBackend):
    """Use the existing runner's training-actor branch, including final token budgets."""

    def __init__(self, client, tokenizer, config, *, policy_version, request_id, eos_ids, seed):
        self.client, self.tokenizer, self.config = client, tokenizer, config
        self.policy_version, self.request_id = policy_version, request_id
        self.eos_ids, self.seed, self.records = list(eos_ids), seed, []
        self.test_only = False

    async def generate(self, messages, folder, turn, *, output_tokens=None, **options):
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if isinstance(prompt, Mapping):
            prompt = prompt["input_ids"]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        count = self.config.max_output_tokens if output_tokens is None else output_tokens
        if not prompt or count < 1 or len(prompt) + count > self.config.context_limit:
            raise ValueError("training_context_limit_exceeded")
        sampled = await self.client.generate(
            request_id=self.request_id, prompt_ids=prompt,
            sampling_params={"temperature": 1.0, "top_k": -1, "top_p": 1.0,
                "max_tokens": count, "logprobs": True, "stop_token_ids": self.eos_ids,
                "ignore_eos": False, "repetition_penalty": 1.0,
                "presence_penalty": 0.0, "frequency_penalty": 0.0,
                "seed": int(digest([self.seed, turn])[:8], 16)})
        completion, old = list(sampled.token_ids), sampled.log_probs
        if (sampled.stop_reason == "aborted" or not completion or len(completion) > count
                or old is None or len(old) != len(completion)
                or any(not math.isfinite(x) or x > 0 for x in old)
                or any(t in self.eos_ids for t in completion[:-1])):
            raise ValueError("invalid_verl_generation")
        # A shorter response without a known stop token cannot be trusted as complete.
        stopped = completion[-1] in self.eos_ids
        if not stopped and len(completion) != count:
            raise ValueError("verl_generation_missing_stop_token")
        content = self.tokenizer.decode(completion, skip_special_tokens=True)
        record = {"schema_version": "awm-policy-turn-v1", "turn": turn,
            "policy_version": self.policy_version, "messages_hash": digest(messages),
            "prompt_ids": prompt, "completion_ids": completion, "old_logprobs": list(old),
            "action_mask": [0] * len(prompt) + [1] * len(completion),
            "stop_policy": STOP_POLICY, "eos_token_ids": self.eos_ids,
            "sampling": {"temperature": 1.0, "top_k": 0, "top_p": 1.0}, "test_only": False}
        write_json(Path(folder) / "policy_turn.json", record)
        self.records.append(record)
        return {"content": content, "finish_reason": "stop" if stopped else "length",
                "test_only": False, "model_version": self.policy_version,
                "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(completion),
                          "total_tokens": len(prompt) + len(completion)}}


async def score_coverage_group(runner, task, episodes):
    from ...evaluation import apply_evaluation, evaluate_episode, primary_assessment
    from ...reward_policy import settle_reward

    async def primary_for(path, result):
        if result["terminal_state_status"] != "trusted" or result["termination_reason"] == "cancelled":
            return None, None
        primary = await primary_assessment(path, task, runner.config, runner.budget, runner.judge_semaphore)
        bundle, assessment = primary
        checked = settle_reward(assessment, bundle, task, runner.config, primary_only=True) if bundle else None
        return primary, checked

    async with asyncio.TaskGroup() as jobs:
        primary_jobs = [jobs.create_task(primary_for(path, result)) for path, result in episodes]
    primaries = [job.result() for job in primary_jobs]
    labels = [item["task_status"] if item and item["reward_exclusion"] == "awaiting_group_decision" else None
              for _, item in primaries]
    uniform = labels[0] if len(labels) == 4 and len(set(labels)) == 1 and labels[0] in {"confirmed_pass", "failed"} else None

    async def finish(path, result, primary):
        if primary is None:
            return result
        if uniform != "failed":
            evaluation = await evaluate_episode(path, task, runner.config, runner.budget, runner.judge_semaphore,
                                                primary=primary)
        else:
            bundle, assessment = primary
            evaluation = settle_reward(assessment, bundle, task, runner.config)
            if not evaluation["reward_valid"] or evaluation["reward"] != 0:
                raise ValueError("all-failed fast path requires four admitted zero rewards")
            evaluation["trajectory_quality"] = {"status": "skipped", "reason": "all_failed", "assessment": None,
                                                 "reviewed": False, "usage": {"physical_requests": 0, "total_tokens": 0}}
            write_json(path / "rewards_v2" / (digest(evaluation) + ".json"), evaluation)
        return apply_evaluation(path, evaluation)

    async with asyncio.TaskGroup() as jobs:
        final = [jobs.create_task(finish(path, result, primary))
                 for (path, result), (primary, _) in zip(episodes, primaries, strict=True)]
    return [job.result() for job in final], uniform


async def collect_group(runner, client, tokenizer, task, folder, *, policy_version, group_size,
                        eos_ids, concurrency, full_coverage=False):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    semaphore = asyncio.Semaphore(concurrency)

    async def candidate(index):
        async with semaphore:
            name = f"candidate-{index:03d}"
            request_id = f"{folder.name}/{name}"
            seed = int(digest([runner.config.seed, policy_version, task.task_instance_id,
                               folder.name, index])[:8], 16)
            actor = VerlActorBackend(client, tokenizer, runner.config, policy_version=policy_version,
                                     request_id=request_id, eos_ids=eos_ids, seed=seed)
            if full_coverage:
                result = await runner.run(task, folder / name, request_id, actor, evaluate_after=False)
            else:
                result = await runner.run(task, folder / name, request_id, actor)
            return result

    # TaskGroup joins/cancels every candidate before weights can change.
    async with asyncio.TaskGroup() as group:
        jobs = [group.create_task(candidate(i)) for i in range(group_size)]
    results = [job.result() for job in jobs]
    names = [f"candidate-{i:03d}" for i in range(group_size)]
    scoring = {}
    if full_coverage:
        episodes = [(folder / name, result) for name, result in zip(names, results, strict=True)]
        for index, (path, result) in enumerate(episodes):
            seal = verify_seal(path)
            if (result.get("rollout_id") != f"{folder.name}/{names[index]}"
                    or result.get("task_instance_id") != task.task_instance_id
                    or result.get("termination_reason") == "cancelled"
                    or not any(name.startswith("model_turns/") and name.endswith("/policy_turn.json") for name in seal)):
                raise ValueError(f"incomplete trajectory for {task.task_instance_id}: {path}")
        results, uniform = await score_coverage_group(runner, task, episodes)
        scoring = {"uniform_primary": uniform, "candidate_count": len(results),
                   "final_rewards": [r.get("reward") for r in results],
                   "quality_scored": sum((r.get("trajectory_quality") or {}).get("status") == "valid" for r in results),
                   "unknown": sum(r.get("business_label") == "unknown" for r in results),
                   "null_rewards": sum(r.get("reward") is None for r in results),
                   "unscored": sum(r.get("evaluation_valid") is not True for r in results),
                   "evaluation_errors": sum(r.get("evaluation_status") in {"evaluation_error", "rubric_error"}
                                            for r in results),
                   "quality_errors": sum((r.get("trajectory_quality") or {}).get("status") == "error" for r in results)}
    write_json(folder / "group.json", {"task": task.model_dump(),
        "config_hash": digest(runner.config.model_dump()), "policy_version": policy_version,
        "test_only": False, **scoring,
        "candidates": [{"path": name, "result_hash": digest(result)}
                       for name, result in zip(names, results, strict=True)]})
    return scoring

