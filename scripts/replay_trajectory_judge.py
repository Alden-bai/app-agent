"""Rejudge sealed actor traces with Judge 2 only; no actor, environment or GPU.

Results are separate scoring artifacts, not retrospective optimizer updates or
new reward admissions. The first Judge and the original episodes stay unchanged.
"""
import argparse
import asyncio
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness_awm.backend import Budget
from harness_awm.composite_reward import components_for, reward_from_components
from harness_awm.contracts import RunConfig
from harness_awm.io import file_hash, read_json, write_json
from harness_awm.judge_inputs import verify_input
from harness_awm.protocol import estimate_tokens
from harness_awm.trajectory_quality import (
    actor_context, compact_quality_payload, evaluate_trajectory, quality_windows, request_messages,
)


def replay_inputs(episode):
    original = read_json(episode / "episode_result.json")
    expected_id = original.get("evaluation_id")
    if not expected_id:
        raise ValueError("missing_original_evaluation_id:" + str(episode))
    for path in sorted(episode.glob("evaluations_v2/*/attempt_*/result.json")):
        if path.parent.parent.name != expected_id:
            continue
        first = read_json(path)
        if (first.get("evaluation_valid") and first.get("evaluation_status") == "valid"
                and first.get("evaluation_id") == expected_id
                and first.get("business_label") == original.get("business_label")
                and first.get("business_label") != "fail"):
            bundle = read_json(path.with_name("input.json"))
            verify_input(bundle)
            return bundle, first, path
    raise ValueError("missing_valid_first_judge:" + str(episode))


async def run(args):
    run_root, output = Path(args.run).resolve(), Path(args.output).resolve()
    config = RunConfig.model_validate(read_json(run_root / "config.json"))
    output.mkdir(parents=True, exist_ok=True)
    budget = Budget(output / "budget.sqlite", args.max_requests, args.max_tokens)
    semaphore = asyncio.Semaphore(args.concurrency)
    if args.execute:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        if not os.environ.get(config.judge_api.api_key_env):
            raise ValueError("missing_api_key_environment_variable:" + config.judge_api.api_key_env)
        from harness_awm.judge_rate_limit import install
        if config.judge_rpm and config.judge_tpm:
            install(config, output / "rate.sqlite")
    selected, controls = [], 0
    for path in sorted((run_root / "training/groups").glob("*/candidate-*/episode_result.json")):
        old = read_json(path)
        quality = old.get("trajectory_quality") or {}
        if quality.get("status") == "error":
            selected.append((path, old, "failure"))
        elif quality.get("status") == "valid" and controls < args.controls:
            selected.append((path, old, "control"))
            controls += 1

    async def replay(path, old, kind):
        episode = path.parent
        name = episode.parent.name + "/" + episode.name
        row = {"episode": str(episode), "case": name, "kind": kind,
               "old_reward": old.get("reward"), "old_error": (old.get("trajectory_quality") or {}).get("error"),
               "episode_result_sha256": file_hash(path), "scoring_only": True}
        try:
            bundle, first, source = replay_inputs(episode)
            context = actor_context(episode)
            payload = compact_quality_payload(context, bundle["original"]["task"], first)
            row.update(first_judge_path=str(source), first_judge_sha256=file_hash(source),
                       first_judge_label=first["business_label"], turns=len(context["turns"]),
                       estimated_request_tokens=estimate_tokens(request_messages(payload)) + config.judge_v2.max_output_tokens,
                       windows=len(quality_windows(context, bundle["original"]["task"], first, config)))
            if args.execute:
                result = await evaluate_trajectory(bundle, first, config, budget, semaphore, output / name / "quality")
                row["quality"] = result
                if result["status"] == "valid":
                    components = components_for({**first, "trajectory_quality": result})
                    row.update(components=components, recomputed_reward=reward_from_components(
                        components, first["business_label"], first["task_status"]))
                else:
                    row["recomputed_reward"] = None
            else:
                row["status"] = "preflight_passed"
            if file_hash(path) != row["episode_result_sha256"] or file_hash(source) != row["first_judge_sha256"]:
                raise ValueError("original_result_changed_during_replay")
        except Exception as exc:
            row.update(error=f"{type(exc).__name__}: {exc}", recomputed_reward=None)
        write_json(output / name / ("replay.json" if args.execute else "preflight.json"), row)
        print(name, row.get("quality", {}).get("status", row.get("status", "error")),
              row.get("recomputed_reward"), flush=True)
        return row

    rows = await asyncio.gather(*(replay(*case) for case in selected))
    report = {"execute": args.execute, "source_run": str(run_root), "scoring_only": True,
              "cases": rows, "usage": budget.summary(),
              "recovered_failures": sum(r["kind"] == "failure" and r.get("recomputed_reward") is not None for r in rows),
              "failure_cases": sum(r["kind"] == "failure" for r in rows)}
    write_json(output / ("report.json" if args.execute else "preflight.json"), report)
    print("report", str(output), "recovered", report["recovered_failures"], "/", report["failure_cases"], flush=True)
    return 1 if any(r.get("error") or (args.execute and r.get("recomputed_reward") is None) for r in rows) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true", help="Call the configured remote Judge API; default is CPU preflight only")
    parser.add_argument("--controls", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-requests", type=int, default=150)
    parser.add_argument("--max-tokens", type=int, default=15000000)
    args = parser.parse_args()
    if args.controls < 0 or args.concurrency < 1:
        parser.error("controls must be nonnegative and concurrency positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
