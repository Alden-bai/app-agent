"""Pinned verl worker orchestration; imported only by an explicit train command."""
import asyncio
import math
import json
import time
from collections import Counter
from functools import partial
from pathlib import Path

from verl.trainer.main_ppo import TaskRunner
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from ...backend import Budget
from ...contracts import RunConfig, TaskSpec
from ...io import digest, read_json, write_json
from ...runner import EpisodeRunner
from ..grpo import preflight
from ..rewards import InvalidRewardGroup
from .core import check_logprobs, read_group, rollout_correction, turn_batch
from .loss import policy_loss
from .rollout import collect_group


class HarnessRolloutManager:
    """The AWM runner owns environments; verl owns shared inference replicas.

This manager uses verl's supported replacement point. No unused default
AgentLoop workers or synthetic single-completion trajectories are created.
"""
    @classmethod
    def create(cls, config, llm_client, **kwargs):
        instance = cls()
        instance.llm_client = llm_client
        return instance


def collate_task_ids(rows):
    return rows


class AWMTaskRunner(TaskRunner):
    def add_actor_rollout_worker(self, config):
        import ray
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role
        from .workers import AWMActorRolloutRefWorker
        self.role_worker_mapping[Role.ActorRollout] = ray.remote(AWMActorRolloutRefWorker)
        self.mapping[Role.ActorRollout] = "global_pool"
        return AWMActorRolloutRefWorker, RayWorkerGroup

    def run(self, config):
        from .compat import worker_setup
        worker_setup()
        from transformers import AutoTokenizer
        from .config import validate_turnwise_config

        _, worker_group_cls = self.add_actor_rollout_worker(config)
        validate_turnwise_config(config)
        tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path,
                                                  local_files_only=True, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            raise ValueError("checkpoint_requires_pad_token")
        plan = read_json(config.awm.plan)
        # This dataset carries task identities only. Prompt construction remains
        # inside AWM; Judge-private inputs never enter the training dataset.
        tasks = [row["task_instance_id"] for row in plan["tasks"]]
        offset = plan.get("group_offset", 0) % len(tasks)
        tasks = tasks[offset:] + tasks[:offset]
        trainer = AWMTrainer(config=config, tokenizer=tokenizer, processor=None,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=self.init_resource_pool_mgr(config),
            ray_worker_group_cls=worker_group_cls, train_dataset=tasks,
            # Upstream requires a nonempty val loader. Our fit never evaluates it.
            val_dataset=tasks[:1], collate_fn=collate_task_ids)
        trainer.init_workers()
        trainer.actor_rollout_wg.set_loss_fn(partial(policy_loss, clip=plan["clip"]))
        if config.awm.get("probe_only", False):
            trainer.probe()
        else:
            trainer.fit()


class AWMTrainer(RayPPOTrainer):
    def _save_group_checkpoint(self, completed_groups, plan):
        """Save by rollout group without relabeling the optimizer's step counter."""
        import torch
        folder = Path(plan["checkpoint_dir"]) / f"group_step_{completed_groups}"
        folder.mkdir(parents=True, exist_ok=False)
        self.actor_rollout_wg.save_checkpoint(
            str(folder / "actor"), None, self.global_steps, max_ckpt_to_keep=None)
        torch.save(self.train_dataloader.state_dict(), folder / "data.pt")
        metadata = {"step_unit": "groups", "groups": completed_groups,
                    "optimizer_steps": self.global_steps, "path": str(folder)}
        write_json(folder / "training_state.json", metadata)
        write_json(Path(plan["checkpoint_dir"]) / "latest_group_checkpoint.json", metadata)

    def _compute_old_log_prob(self, batch):
        # This recipe never consumes entropy; upstream computes it unconditionally.
        from verl.protocol import DataProto
        from verl.utils import tensordict_utils as tu
        from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding
        td = left_right_2_no_padding(batch.to_tensordict())
        tu.assign_non_tensor(td, calculate_entropy=False, calculate_sum_pi_squared=False, compute_loss=False)
        output = self.actor_rollout_wg.compute_log_prob(td)
        probs = no_padding_2_padding(tu.get(output, "log_probs"), td)
        result = DataProto.from_tensordict(tu.get_tensordict({"old_log_probs": probs.float()}))
        return result, tu.get(output, "metrics")["mfu"]

    def probe(self):
        """Same actor/rollout workers as training, no Judge and no parameter updates."""
        from .rollout import VerlActorBackend
        from verl.protocol import DataProto
        plan = read_json(self.config.awm.plan)
        folder = Path(plan["output"]) / "gpu_probe"
        folder.mkdir(exist_ok=False)
        config = RunConfig.model_validate(plan["run_config"])
        try:
            self.checkpoint_manager.update_weights(0)
            actor = VerlActorBackend(self.async_rollout_manager.llm_client, self.tokenizer, config,
                policy_version="probe/0", request_id="probe/0", eos_ids=plan["eos_ids"], seed=config.seed)
            messages = [{"role": "user", "content": 'Return only this JSON: {"ok":true}'}]
            asyncio.run(actor.generate(messages, folder, 0, output_tokens=32))
            if plan.get("stress_probe"):
                # Exercise a real 32K prompt through the same inference worker.
                # This diagnostic worker is discarded; no probe weights are trained on.
                count = config.context_limit - 32
                ids = actor.records[0]["prompt_ids"]
                prompt = (ids * math.ceil(count / len(ids)))[:count]
                sampled = asyncio.run(actor.client.generate(request_id="probe/stress", prompt_ids=prompt,
                    sampling_params={"temperature": 1.0, "top_k": -1, "top_p": 1.0,
                                     "max_tokens": 32, "logprobs": True,
                                     "stop_token_ids": plan["eos_ids"], "ignore_eos": False}))
                actor.records = [{"prompt_ids": prompt, "completion_ids": list(sampled.token_ids),
                                  "old_logprobs": list(sampled.log_probs)}]
            self.checkpoint_manager.sleep_replicas()
            tensors = turn_batch([[actor.records[0]]], [1.0], self.tokenizer.pad_token_id, plan["gpus"])
            batch = DataProto.from_dict(tensors=tensors)
            batch.meta_info.update(temperature=1.0, global_token_num=tensors["attention_mask"].sum(-1).tolist())
            recomputed, _ = self._compute_old_log_prob(batch)
            if plan.get("rollout_correction") == "token-is-v1":
                _, correction = rollout_correction(tensors, recomputed.batch["old_log_probs"])
                delta = correction["max_logprob_abs_delta"]
            else:
                delta = check_logprobs(tensors, recomputed.batch["old_log_probs"])
            stress_update = None
            if plan.get("stress_probe"):
                stress_update = self._update_turns(tensors)
                self.checkpoint_manager.update_weights(1)
                self.checkpoint_manager.sleep_replicas()
            write_json(folder / "report.json", {"passed": True, "max_logprob_abs_delta": delta,
                "judge_requests": 0, "optimizer_steps": int(stress_update is not None),
                "diagnostic_only": True, "stress_update": stress_update,
                "sequence_tokens": tensors["input_ids"].shape[-1], "weight_sync": True,
                "parameters": self.actor_rollout_wg.parameter_probe()})
        finally:
            self.checkpoint_manager.abort_replicas()
            self.checkpoint_manager.sleep_replicas()
            self._shutdown_dump_executor()

    def _update_turns(self, tensors):
        from verl.protocol import DataProto
        from verl.utils import tensordict_utils as tu
        from verl.workers.utils.padding import left_right_2_no_padding

        world = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        if len(tensors["input_ids"]) % world:
            raise ValueError("turn batch must be padded to the training DP size")
        batch = DataProto.from_dict(tensors=tensors)
        batch.meta_info.update(temperature=1.0,
                               global_token_num=tensors["attention_mask"].sum(-1).tolist())
        recomputed, _ = self._compute_old_log_prob(batch)
        mode = read_json(self.config.awm.plan).get("rollout_correction", "strict")
        correction = None
        if mode == "token-is-v1":
            extra, correction = rollout_correction(tensors, recomputed.batch["old_log_probs"])
            for key, value in extra.items():
                batch.batch[key] = value
            delta = correction["max_logprob_abs_delta"]
        elif mode == "strict":
            delta = check_logprobs(tensors, recomputed.batch["old_log_probs"])
        else:
            raise ValueError("unknown rollout correction mode")
        # A single global mini-batch containing all turns => one optimizer step.
        # Never replace rollout old_log_probs with the recomputed values.
        td = left_right_2_no_padding(batch.to_tensordict())
        tu.assign_non_tensor(td, temperature=1.0, calculate_entropy=False,
            global_batch_size=len(batch), mini_batch_size=len(batch), epochs=1,
            seed=self.config.data.seed, dataloader_kwargs={"shuffle": False}, compute_loss=True)
        output = self.actor_rollout_wg.update_actor(td)
        metrics = tu.get(output, "metrics")
        metrics = json.loads(json.dumps(metrics, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
        def values(value):
            if isinstance(value, list):
                return [number for item in value for number in values(item)]
            return [float(value)]
        norms = values(metrics["grad_norm"])
        if not norms or not all(math.isfinite(n) and n > 0 for n in norms):
            raise ValueError("optimizer_update_has_no_finite_nonzero_gradient")
        return {"max_logprob_abs_delta": delta, "worker_metrics": metrics,
                "rollout_correction": correction}

    def fit(self):
        plan = read_json(self.config.awm.plan)
        config = RunConfig.model_validate(plan["run_config"])
        tasks = {row["task_instance_id"]: TaskSpec.model_validate(row) for row in plan["tasks"]}
        output = Path(plan["output"])
        budget = Budget(output / "budget.sqlite", config.max_physical_requests, config.max_total_tokens)
        from ...judge_rate_limit import install
        install(config, output / "rate.sqlite")
        client = self.async_rollout_manager.llm_client
        reports = []
        full_coverage = plan.get("full_coverage", False)
        processed = set()
        checkpoints = []
        quarantined = {}
        guarded = plan.get("pilot_guards", False)
        if guarded:
            from .pilot import resource_check, group_health, GroupArchiver
        self.global_steps = 0
        self.max_steps_duration = 0
        failed = True
        replicas_awake = False
        target = plan.get("target_updates")
        null_reward_policy = plan.get("null_reward_policy", "exclude")
        save_unit = plan.get("save_unit", "updates")
        if save_unit not in {"groups", "updates"}:
            raise ValueError("unknown checkpoint save unit")
        last_saved_step = None
        started = time.monotonic()
        def enough():
            return len(reports) >= plan["groups"] or (target is not None and self.global_steps >= target)
        def checkpoint_step():
            return len(reports) if save_unit == "groups" else self.global_steps
        def save_checkpoint():
            if save_unit == "groups":
                self._save_group_checkpoint(len(reports), plan)
            else:
                self._save_checkpoint()
            if full_coverage:
                checkpoints.append({"completed_tasks": len(reports), "optimizer_steps": self.global_steps})
        from ...tracking import SwanLabTracker
        tracker = SwanLabTracker(output, config={"stage": "verl-grpo", "group_size": plan["group_size"],
                                                "model": plan["model"], "reward_policy": plan["reward_policy"]})
        write_json(output / "tracking.json", {"mode": tracker.mode, "url": tracker.url})
        # One loop throughout training: keep Judge semaphore and client loop affinity.
        with asyncio.Runner() as loop:
            runner = EpisodeRunner(config, budget, asyncio.Semaphore(config.judge_concurrency))
            archiver = GroupArchiver() if guarded else None
            archive_error = None
            try:
                self.checkpoint_manager.update_weights(0)
                replicas_awake = True
                write_json(output / "initial_parameters.json", self.actor_rollout_wg.parameter_probe())
                for epoch in range(math.ceil(plan["groups"] / len(tasks))):
                    for task_ids in self.train_dataloader:
                        task = tasks[task_ids[0]]
                        if full_coverage and task.task_instance_id in processed:
                            raise ValueError("duplicate task in full-coverage run: " + task.task_instance_id)
                        if task.task_instance_id in quarantined:
                            continue
                        if guarded:
                            archiver.wait_for_slot()
                            resource_check(plan["checkpoint_dir"], gpu_limit_gb=plan.get("gpu_stop_gb", 78) - 1)
                            resource_check(output, free_gib=4, gpu_limit_gb=plan.get("gpu_stop_gb", 78) - 1)
                        index = plan.get("group_offset", 0) + len(reports)
                        group_started = time.monotonic()
                        folder = output / "groups" / f"group-{index:06d}"
                        version = f"{plan['checkpoint']['sha256']}/step-{self.global_steps}"
                        preflight([task], config, plan["group_size"])
                        # Inference weights are constant for the whole group.
                        scoring = loop.run(collect_group(runner, client, self.tokenizer, task, folder,
                            policy_version=version, group_size=plan["group_size"],
                            eos_ids=plan["eos_ids"], concurrency=plan["concurrency"],
                            **({"full_coverage": True} if full_coverage else {})))
                        self.checkpoint_manager.abort_replicas()
                        self.checkpoint_manager.sleep_replicas()
                        replicas_awake = False
                        report = {"group": index, "policy_version": version, "optimizer_step": self.global_steps}
                        if full_coverage:
                            if scoring["candidate_count"] != 4:
                                raise ValueError("incomplete group: " + task.task_instance_id)
                            report.update(task_instance_id=task.task_instance_id, **scoring)
                        if guarded:
                            health = group_health(folder, plan["group_size"])
                            report["health"] = health
                            if health["api_fault"]:
                                write_json(folder / "api-recovery-exhausted.json", health)
                            if health["quarantine"] and not health["api_fault"] and null_reward_policy != "zero":
                                quarantined[task.task_instance_id] = health
                                write_json(output / "quarantined.json", quarantined)
                        try:
                            if full_coverage and scoring["uniform_primary"] == "failed":
                                raise InvalidRewardGroup("all_failed")
                            if (full_coverage and scoring["uniform_primary"] == "confirmed_pass"
                                    and (scoring["quality_scored"] != 4 or scoring["null_rewards"])):
                                raise InvalidRewardGroup("unsettled_success_quality")
                            trajectories, rewards, advantages = read_group(
                                folder, task, config, version, plan["group_size"], self.tokenizer,
                                null_reward_policy=null_reward_policy)
                        except InvalidRewardGroup as exc:
                            report.update(status="held", reason=str(exc))
                        else:
                            report.update(rewards=rewards, advantages=advantages,
                                          effective_group_size=len(rewards))
                            selection = folder / "reward_selection.json"
                            if selection.exists():
                                report["reward_selection"] = read_json(selection)
                            if not any(advantages):
                                report["status"] = "zero_advantage"
                                if full_coverage:
                                    report["reason"] = "equal_training_rewards" if scoring["null_rewards"] else "equal_final_rewards"
                            else:
                                tensors = turn_batch(trajectories, advantages, self.tokenizer.pad_token_id,
                                                     plan["gpus"])
                                report.update(self._update_turns(tensors))
                                parameters = self.actor_rollout_wg.parameter_probe()
                                if not all(p["finite"] and p.get("changed_samples", 0) > 0 for p in parameters):
                                    raise ValueError("optimizer_update_has_no_finite_parameter_change")
                                report["parameters"] = parameters
                                self.global_steps += 1
                                report.update(status="updated", optimizer_step=self.global_steps)
                        report["wall_seconds"] = time.monotonic() - group_started
                        write_json(folder / "update.json", report)
                        if guarded:
                            report["archive"] = archiver.submit(folder,
                                Path(plan["checkpoint_dir"]).parent / "archives" / (folder.name + ".tar.gz"))
                            report["archive_manifest"] = str(Path(report["archive"]).with_suffix('.manifest.json'))
                        reports.append(report)
                        if full_coverage:
                            processed.add(task.task_instance_id)
                        write_json(output / "progress.json", {"groups": len(reports),
                            "optimizer_steps": self.global_steps, "latest": report, "budget": budget.summary()})
                        with (output / "progress.jsonl").open("a") as handle:
                            handle.write(json.dumps(report) + "\n")
                        metrics = {"progress/groups": len(reports), "train/optimizer_steps": self.global_steps,
                                   "time/group_seconds": report["wall_seconds"], "budget": budget.summary(),
                                   "logprob/max_abs_delta": report.get("max_logprob_abs_delta"),
                                   "group/held": report["status"] == "held",
                                   "group/null_penalties": len(report.get("reward_selection", {}).get("null_penalties", [])),
                                   "group/zero_advantage": report["status"] == "zero_advantage"}
                        for field in ("rewards", "advantages", "parameters"):
                            metrics[field] = {str(i): value for i, value in enumerate(report.get(field, []))}
                        for key, value in report.get("worker_metrics", {}).items():
                            if isinstance(value, list):
                                value = {str(i): item for i, item in enumerate(value)}
                            metrics["train/" + key] = value
                        tracker.log(metrics, step=len(reports))
                        if (save_unit == "groups" or report["status"] == "updated") and checkpoint_step() % plan["save_every"] == 0:
                            save_checkpoint()
                            last_saved_step = checkpoint_step()
                        if enough():
                            break
                        self.checkpoint_manager.update_weights(self.global_steps)
                        replicas_awake = True
                    if enough():
                        break
                self.checkpoint_manager.sleep_replicas()
                replicas_awake = False
                if full_coverage and processed != set(tasks):
                    raise ValueError("not every selected training task was processed")
                if last_saved_step != checkpoint_step():
                    save_checkpoint()
                failed = False
            finally:
                # Drain even on rollout/update failure. A failed archive must
                # never leave a successful summary, and originals stay intact.
                try:
                    if archiver is not None:
                        archiver.close()
                except BaseException as exc:
                    archive_error = repr(exc)
                    failed = True
                    raise
                finally:
                    # Also drains/aborts inference after an interrupted rollout.
                    try:
                        if replicas_awake:
                            self.checkpoint_manager.abort_replicas()
                            self.checkpoint_manager.sleep_replicas()
                    finally:
                        if full_coverage:
                            write_json(output / "coverage.json", {
                                "completed_tasks": len(processed), "expected_tasks": len(tasks),
                                "missing_task_ids": [key for key in tasks if key not in processed],
                                "trajectories": sum(r["candidate_count"] for r in reports),
                                "skipped_by_reason": dict(Counter(r.get("reason", r["status"])
                                    for r in reports if r["status"] != "updated")),
                                "scoring": {key: sum(r[key] for r in reports) for key in
                                    ("unknown", "null_rewards", "unscored", "evaluation_errors", "quality_errors")},
                                "selected_task_ids_sha256": digest(list(tasks)),
                                "optimizer_steps": self.global_steps, "checkpoints": checkpoints,
                                "final_update": self.global_steps if not failed else None,
                                "complete": not failed and processed == set(tasks),
                            })
                        write_json(output / "summary.json", {"backend": "verl", "failed": failed,
                            "optimizer_steps": self.global_steps, "groups": reports, "budget": budget.summary(),
                            "target_updates": target, "save_unit": save_unit,
                            "target_met": len(reports) >= plan["groups"] if target is None else self.global_steps >= target,
                            "archive_error": archive_error,
                            "wall_seconds": time.monotonic() - started,
                            "validation_performed": False, "test_performed": False})
                        tracker.finish(failed=failed)
                        self._shutdown_dump_executor()

