"""Offline CST candidate comparison. Does not admit rewards or launch training.

Usage: PYTHONPATH=. python scripts/preview_cst_reward_v2.py --run RUN --output NEW_REPORT
Existing Judge decisions are reused; this is arithmetic replay, not evaluation.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from harness_awm.composite_reward import reward_from_components
from harness_awm.training.rewards import admitted_group, population_advantages


DRAFT_POLICY = "judge-v2-cst-v2-preview"
LAMBDA_S = 0.10
LAMBDA_T = 0.30
STD_FLOOR = 0.10


def preview_reward(result):
    """Validate v1 components before repricing them; never manufacture evidence."""
    parts = result.get("reward_components")
    old = reward_from_components(parts, result.get("business_label"), result.get("task_status"))
    if result.get("reward_valid") is not True or result.get("reward") != old:
        raise ValueError("invalid_source_reward")
    if parts["C"] == 0:
        return 0.0
    return round(parts["C"] * (1 - LAMBDA_S * (1 - parts["S"]) - LAMBDA_T * (1 - parts["T"])), 12)


def floored_advantages(rewards, std_floor=STD_FLOOR):
    if (type(std_floor) not in (int, float) or not math.isfinite(std_floor) or std_floor <= 0
            or len(rewards) < 2 or any(type(x) not in (int, float) or not math.isfinite(x) for x in rewards)):
        raise ValueError("invalid_preview_advantage_inputs")
    mean = statistics.mean(rewards)
    denominator = max(statistics.pstdev(rewards), std_floor) + 1e-6
    return [(r - mean) / denominator for r in rewards]


def metrics(rewards, *, floor):
    advantages = floored_advantages(rewards) if floor else population_advantages(rewards)
    return {"rewards": rewards, "mean": statistics.mean(rewards),
            "std": statistics.pstdev(rewards), "advantages": advantages,
            "advantage_rms": math.sqrt(statistics.mean(a * a for a in advantages))}


def replay(run):
    records, excluded, sources = [], [], []
    for update_path in sorted((Path(run) / "groups").glob("*/update.json")):
        update = json.loads(update_path.read_text())
        candidates = sorted(update_path.parent.glob("candidate-*/episode_result.json"))
        results = [json.loads(p.read_text()) for p in candidates]
        if update["status"] != "updated":
            excluded.append({"group": update["group"], "status": update["status"],
                "update_reason": update.get("reason"),
                "candidate_exclusions": [{"path": str(p.resolve()), "reason": r.get("reward_exclusion")}
                                         for p, r in zip(candidates, results) if r.get("reward_exclusion")]})
            continue
        # The saved update and episode results must agree before comparison.
        old = admitted_group(results, group_size=len(update["rewards"]))
        if old != update["rewards"]:
            raise ValueError(f"source_update_reward_mismatch:{update_path}")
        if any(r["task_instance_id"] != results[0]["task_instance_id"] for r in results):
            raise ValueError(f"mixed_task_group:{update_path}")
        new = [preview_reward(r) for r in results]
        sources.extend(results)
        records.append({"group": update["group"], "optimizer_step": update["optimizer_step"],
            "task_instance_id": results[0]["task_instance_id"], "source": str(update_path.resolve()),
            "all_confirmed_success": all(r["reward_components"]["C"] == 1 for r in results),
            "v1": metrics(old, floor=False), "v1_floor_only": metrics(old, floor=True),
            "v2_weights_only": metrics(new, floor=False), "v2_with_floor": metrics(new, floor=True)})
    if not records:
        raise ValueError("no_completed_update_groups")
    positive = [r for r in sources if r["reward_components"]["C"] > 0]
    return {"status": "offline_preview_pending_user_confirmation", "policy": DRAFT_POLICY,
        "training_ready": False, "source_run": str(Path(run).resolve()),
        "parameters": {"alpha": 0.3, "lambda_s": LAMBDA_S, "lambda_t": LAMBDA_T,
                       "std_floor": STD_FLOOR, "epsilon": 1e-6},
        "limitations": ["Frozen Judge decisions; no new rollouts or Judge requests.",
                        "Not a new policy's measured success rate or optimizer update.",
                        "No training admission or sealed-evidence revalidation is performed.",
                        "Judge failures remain excluded; no null-to-zero conversion."],
        "summary": {"updated_groups": len(records), "excluded_groups": len(excluded),
                    "candidates": len(sources), "old_rewards": dict(Counter(str(r["reward"]) for r in sources)),
                    "positive_tier_candidates": len(positive),
                    "positive_tier_without_soft_items": sum(not r["reward_components"]["soft_items"] for r in positive),
                    "all_confirmed_success_groups": sum(g["all_confirmed_success"] for g in records)},
        "groups": records, "excluded_groups": excluded}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = replay(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
