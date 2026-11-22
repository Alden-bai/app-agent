"""Evaluate a local checkpoint on held-out AWM tasks using the GRPO runner/Judge."""
import argparse
import asyncio
import json
import os
import time
from importlib.metadata import version
from pathlib import Path

from ..backend import Budget
from ..contracts import ROOT, RunConfig
from ..io import digest, file_hash, now, read_json, write_json
from ..judge_inputs import sources_from_manifest
from ..judge_v2 import normalize_rubric
from ..runner import EpisodeRunner
from ..stages import dependencies, metrics, official_hashes
from ..tracking import SwanLabTracker, run_config
from .actor import TrainingActorBackend, STOP_POLICY, stop_token_ids
from .local import checkpoint_identity, load_checkpoint
from .tasks import SPLITS, load_tasks, select_tasks


def judge_budget_exhausted(folder):
    """A rejected reservation can exhaust budget before the numeric limit is hit."""
    folder = Path(folder)
    paths = list((folder / "rewards_v2").glob("*.json"))
    paths += list((folder / "evaluations_v2").glob("**/result.json"))
    return any("global_budget_exhausted" in str(read_json(p).get("errors", [])) for p in paths)


def preflight(tasks, config, split):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("local checkpoint evaluation currently requires one process")
    if config.purpose != "evaluation" or config.actor != "training":
        raise ValueError("local evaluation requires purpose=evaluation and actor=training")
    if split not in {"validation", "test"} or not tasks or any(t.split != SPLITS[split] for t in tasks):
        raise ValueError("evaluation requires the selected held-out split")
    if len({t.task_instance_id for t in tasks}) != len(tasks):
        raise ValueError("duplicate evaluation tasks")
    if any(limit is not None and limit < 1 for limit in (config.max_physical_requests, config.max_total_tokens)):
        raise ValueError("evaluation requires an explicit positive Judge request/token budget")
    if config.candidates_per_task != 1:
        raise ValueError("checkpoint evaluation reports pass@1; candidates_per_task must be 1")
    if not config.judge_inputs_manifest:
        raise ValueError("evaluation requires frozen Judge sources")
    for task in tasks:
        if task.protocol_id != config.protocol_id or not task.tool_schema_hash:
            raise ValueError("evaluation task protocol/catalog binding missing or mismatched")
        if file_hash(task.initial_state_ref) != task.initial_state_hash or digest(read_json(task.environment_ref)) != task.environment_code_hash:
            raise ValueError("evaluation runtime asset changed")
        sources = sources_from_manifest(task, config.judge_inputs_manifest)
        if sources["original"]["task_id"] != f"{task.scenario}:{task.native_task_id}" or sources["original"]["task"] != task.instruction:
            raise ValueError("evaluation Judge source/task mismatch")
        normalize_rubric(**sources)
    return {"checked_tasks": len(tasks), "split": split, "evaluation_started": False,
            "metric": "Judge-confirmed task success (pass@1), not Judge accuracy",
            "training_updates": False, "reward_admission_required": False}


async def evaluate(model, tokenizer, tasks, config, output, *, split, checkpoint,
                   task_manifest=None, sampling="greedy", runner_factory=EpisodeRunner):
    preflight(tasks, config, split)
    if sampling not in {"greedy", "policy"}:
        raise ValueError("unsupported evaluation sampling")
    from transformers import set_seed
    model.requires_grad_(False).eval()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    candidates = [{"rollout_id": f"t{i:05d}", "task_instance_id": t.task_instance_id, "scenario": t.scenario,
                   "path": f"episodes/t{i:05d}/a0"} for i, t in enumerate(tasks)]
    manifest = {"schema_version": "awm-run-v1", "stage": "evaluate", "run_id": output.name,
        "created_at": now(), "config": config.model_dump(), "tasks": [t.model_dump() for t in tasks],
        "candidates": candidates, "checkpoint": checkpoint, "task_manifest": task_manifest,
        "evaluation_split": split, "sampling": sampling, "seed": config.seed,
        "training_updates": False, "official_source_hashes": official_hashes(config),
        "harness_source_hashes": {str(p.relative_to(Path(__file__).parents[1])): file_hash(p)
                                  for p in Path(__file__).parents[1].rglob("*.py")},
        "dependencies": {**dependencies(), "inference": {
            name: version(name) for name in ("torch", "transformers")}},
        "inference": {"dtype": str(next(model.parameters()).dtype),
                      "device": str(next(model.parameters()).device), "enable_thinking": False,
                      "stop_policy": STOP_POLICY, "eos_token_ids": stop_token_ids(model, tokenizer)}}
    write_json(output / "manifest.json", manifest)
    write_json(output / "manifest.sha256.json", {"sha256": file_hash(output / "manifest.json")})
    budget = Budget(output / "budget.sqlite", config.max_physical_requests, config.max_total_tokens)
    runner = runner_factory(config, budget, asyncio.Semaphore(config.judge_concurrency))
    tracker = SwanLabTracker(output, config={**run_config(config, "evaluate"), "split": split, "sampling": sampling})
    failed, failures = True, 0
    started = time.monotonic()
    completed, stop_reason = 0, "all_tasks_attempted"
    try:
        # One model, sequential tasks: no competing KV caches or global RNG races.
        for index, (task, candidate) in enumerate(zip(tasks, candidates), 1):
            set_seed(int(digest([config.seed, task.task_instance_id])[:8], 16))
            actor = TrainingActorBackend(model, tokenizer, policy_version=checkpoint["sha256"],
                max_output_tokens=config.max_output_tokens, context_limit=config.context_limit,
                greedy=sampling == "greedy", record_policy=False)
            try:
                await runner.run(task, output / candidate["path"], candidate["rollout_id"], actor)
            except Exception as exc:
                failures += 1
                write_json(output / "failures" / (candidate["rollout_id"] + ".json"),
                           {"error": f"{type(exc).__name__}: {exc}"})
            tracker.log({"progress/completed": index,
                         "progress/failures": failures, "budget": budget.summary()}, step=index)
            completed = index
            if judge_budget_exhausted(output / candidate["path"]):
                stop_reason = "judge_budget_exhausted"
                break
        failed = False
    finally:
        try:
            report = metrics(output)
            report.update(evaluation_split=split, checkpoint_sha256=checkpoint["sha256"], sampling=sampling,
                isolation=config.isolation,
                training_updates=False, independent_judge_accuracy_measured=False,
                process_reward_implemented=False,
                wall_seconds=time.monotonic() - started, attempted_tasks=completed, stop_reason=stop_reason,
                interpretation="Judge-confirmed task success; provisional passes and evaluator faults are separate. No training rewards or updates.")
            write_json(output / "metrics.json", report)
            from ..evaluation_metrics import write_reports
            write_reports(output, report)
            tracker.log({"summary": report}, step=len(candidates) + 1)
        finally:
            tracker.finish(failed=failed or bool(failures) or stop_reason == "judge_budget_exhausted")
    return report


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--split", choices=("validation", "test"), required=True)
    p.add_argument("--task", action="append", help="Optional exact task ID; metrics describe only the selected subset")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--execute", action="store_true", help="Start inference/Judge calls; omitted means preflight only")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    p.add_argument("--sampling", choices=("greedy", "policy"), default="greedy")
    p.add_argument("--skills", type=Path, help="Frozen inference_skills.json; omitted preserves config")
    a = p.parse_args()
    config = RunConfig.model_validate(read_json(a.config))
    if a.skills:
        config = RunConfig.model_validate({**config.model_dump(), "skills": read_json(a.skills)})
    if config.skills is not None:
        from ..skillbank import SkillBank
        SkillBank(config.skills)  # Fail before model loading or any external requests.
    tasks, task_manifest, sources = load_tasks(a.tasks, a.split)
    tasks = select_tasks(tasks, a.task)
    if sources:
        config = config.model_copy(update={"judge_inputs_manifest": sources})
    checked = preflight(tasks, config, a.split)
    checkpoint = checkpoint_identity(a.model)
    if not a.execute:
        print(json.dumps({**checked, "checkpoint_sha256": checkpoint["sha256"]}, indent=2))
        return
    if not a.output or a.output.exists():
        p.error("--execute requires a new --output directory")
    model, tokenizer = load_checkpoint(a.model, device=a.device, dtype=a.dtype)
    report = asyncio.run(evaluate(model, tokenizer, tasks, config, a.output, split=a.split,
        checkpoint=checkpoint, task_manifest=task_manifest, sampling=a.sampling))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
