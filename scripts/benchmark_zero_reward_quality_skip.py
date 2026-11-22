"""Replay sealed GRPO archives without API calls; project saved recorded latency.

Runs real admission/settlement checks and the production evaluate_episode path on
temporary copies. Only the two external Judge stages return frozen results.
Latency projections use archive mtimes, retaining the concurrent group barrier.
They are not a new live training throughput measurement.
"""
import argparse
import asyncio
import copy
import json
from pathlib import Path
import tarfile
import tempfile
import time
from unittest.mock import AsyncMock, patch

from harness_awm.backend import Budget
from harness_awm.contracts import RunConfig, TaskSpec
from harness_awm.evaluation import evaluate_episode
from harness_awm.io import file_hash, read_json, write_json
from harness_awm.judge_inputs import input_from_manifest
from harness_awm.judge_v2 import JudgeResult
from harness_awm.reward_policy import map_reward, settle_reward
from harness_awm.training.rewards import population_advantages


REWARD_FIELDS = ("reward", "reward_valid", "reward_policy", "reward_admission_id",
                 "reward_exclusion", "reward_components", "business_label", "task_status")


def projected_saving(candidates):
    """Time saved at a join is the change in max finish time, not a sum."""
    old = max(c["original_finish"] for c in candidates)
    new = max(c["primary_finish"] if c["skipped"] else c["original_finish"] for c in candidates)
    return max(0.0, old - new)


async def replay_archive(archive, config, std_floor):
    with tempfile.TemporaryDirectory(prefix="awm-zero-quality-") as tmp:
        with tarfile.open(archive, "r:gz") as tf:
            members = tf.getmembers()
            if any(not (m.isfile() or m.isdir()) for m in members):
                raise ValueError("archive must contain only regular files and directories")
            times = {m.name: m.mtime for m in members}
            tf.extractall(tmp, filter="data")
        roots = list(Path(tmp).glob("group-*"))
        if len(roots) != 1:
            raise ValueError("expected one archived group")
        root = roots[0]
        update = read_json(root / "update.json")
        candidates = []
        old_rewards, new_rewards = [], []
        for folder in sorted(root.glob("candidate-*")):
            saved = read_json(folder / "episode_result.json")
            task = TaskSpec.model_validate(read_json(folder / "task.json"))
            bundle = input_from_manifest(folder, task, config.judge_inputs_manifest)
            assessments = sorted(folder.glob("evaluations_v2/*/attempt_*/result.json"))
            # Never choose a favorable result from multiple evaluation identities.
            if len({p.parent.parent for p in assessments}) != 1:
                raise ValueError(f"ambiguous or missing primary assessment: {folder.name}")
            primary_path = next((p for p in assessments if read_json(p)["evaluation_status"]
                                 in {"valid", "invalid_sample", "rubric_error"}), assessments[-1])
            primary = read_json(primary_path)
            quality = saved.get("trajectory_quality")
            if quality is None:
                raise ValueError("baseline archive has no quality result")
            baseline = settle_reward(dict(primary, trajectory_quality=quality), bundle, task, config)
            for field in REWARD_FIELDS:
                if baseline.get(field) != saved.get(field):
                    raise ValueError(f"baseline no longer reproduces saved {field}: {folder.name}")
            quality_stage = AsyncMock(return_value=copy.deepcopy(quality))
            start = time.perf_counter()
            with patch("harness_awm.judge_runtime.evaluate_input", AsyncMock(return_value=copy.deepcopy(primary))), \
                 patch("harness_awm.trajectory_quality.evaluate_trajectory", quality_stage), \
                 patch("harness_awm.backend._api_request_once", AsyncMock(side_effect=AssertionError("offline replay"))):
                actual = await evaluate_episode(folder, task, config,
                    Budget(folder / "replay-budget.sqlite", 0, 0), asyncio.Semaphore(8))
            replay_seconds = time.perf_counter() - start
            JudgeResult.model_validate(actual)
            if {k: v for k, v in actual.items() if k != "trajectory_quality"} != {
                    k: v for k, v in baseline.items() if k != "trajectory_quality"}:
                raise ValueError(f"settlement changed: {folder.name}")
            skipped = quality_stage.await_count == 0
            if skipped:
                reason = actual["trajectory_quality"]["reason"]
                zero = reason == "confirmed_zero_reward" and actual["reward_valid"] and actual["reward"] == 0
                invalid = (reason == "invalid_primary_assessment" and actual["reward"] is None
                    and not actual["reward_valid"] and not map_reward(primary,
                        approved=True, execution_valid=True, test_only=False)["reward_valid"])
                if not (zero or invalid):
                    raise ValueError("invalid skip")
            requests = list((folder / "trajectory_quality").rglob("request.json"))
            def mtime(path):
                return times[str(path.relative_to(tmp))]
            candidates.append({"candidate": folder.name, "skipped": skipped,
                "skip_reason": actual["trajectory_quality"].get("reason") if skipped else None,
                "reward": actual["reward"], "reward_valid": actual["reward_valid"],
                "quality_requests": len(requests), "saved_quality_requests": len(requests) if skipped else 0,
                "original_finish": mtime(folder / "episode_result.json"),
                "primary_finish": mtime(primary_path), "offline_replay_seconds": replay_seconds})
            old_rewards.append(baseline["reward"])
            new_rewards.append(actual["reward"])
        # This benchmark covers the null-as-zero training recipe used by these runs.
        old_training = [r if r is not None else 0.0 for r in old_rewards]
        new_training = [r if r is not None else 0.0 for r in new_rewards]
        if old_training != update["rewards"] or old_training != new_training:
            raise ValueError("training reward vector changed")
        advantages = population_advantages(new_training, std_floor=std_floor)
        if advantages != update["advantages"]:
            raise ValueError("training advantages changed")
        saved_seconds = projected_saving(candidates)
        wall = update["wall_seconds"]
        if saved_seconds > wall:
            raise ValueError("inconsistent archive timestamps")
        return {"group": root.name, "archive": str(archive.resolve()), "archive_sha256": file_hash(archive),
            "status": update["status"], "rewards": new_training, "advantages": advantages,
            "reward_and_advantage_match": True, "original_group_seconds": wall,
            "projected_saved_seconds": saved_seconds, "projected_group_seconds": wall - saved_seconds,
            "candidates": candidates}


async def benchmark(run):
    plan_path = next(p for p in (run / "training/plan.json", run / "records/plan.json") if p.exists())
    plan = read_json(plan_path)
    if plan.get("null_reward_policy") != "zero":
        raise ValueError("benchmark requires the null-as-zero recipe")
    config = RunConfig.model_validate(plan["run_config"])
    archives = sorted(p for p in (run / "archives").glob("group-*.tar.gz")
                      if p.with_suffix('.manifest.json').is_file())
    if not archives:
        raise ValueError("no completed archives")
    groups = []
    for archive in archives:
        groups.append(await replay_archive(archive, config, plan["advantage_std_floor"]))
        print(f"{run.name}/{archive.name}: rewards and advantages match", flush=True)
    old = sum(g["original_group_seconds"] for g in groups)
    new = sum(g["projected_group_seconds"] for g in groups)
    candidates = [c for g in groups for c in g["candidates"]]
    return {"run": str(run.resolve()), "groups": groups, "summary": {
        "groups": len(groups), "candidates": len(candidates),
        "optimizer_updates": sum(g["status"] == "updated" for g in groups),
        "skipped_candidates": sum(c["skipped"] for c in candidates),
        "quality_requests": sum(c["quality_requests"] for c in candidates),
        "saved_quality_requests": sum(c["saved_quality_requests"] for c in candidates),
        "original_group_seconds": old, "projected_group_seconds": new,
        "projected_saved_seconds": old - new, "projected_speedup": old / new,
        "projected_time_reduction_fraction": 1 - new / old,
        "all_rewards_and_advantages_match": True}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"method": "sealed-archive-production-path-replay-and-recorded-critical-path-projection",
        "live_api_calls": 0, "live_training_benchmark": False,
        "limitations": ["Judge results are frozen; no new judgments or training updates are performed.",
            "Projection retains each group's max candidate finish time; candidate savings are not summed.",
            "Assumes unchanged remaining request latencies and optimizer time; excludes startup, archives, weight sync and checkpoints outside recorded group time."],
        "runs": [asyncio.run(benchmark(run)) for run in args.run]}
    write_json(args.output, report)
    print(json.dumps([r["summary"] for r in report["runs"]], indent=2))


if __name__ == "__main__":
    main()
