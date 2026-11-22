"""Matched baseline/skill inference on a fixed held-out subset; dry-run by default."""
import argparse
import asyncio
from collections import Counter
import json
import math
from pathlib import Path

from ..contracts import ROOT, RunConfig
from ..io import digest, file_hash, read_json, write_json
from ..skillbank import SkillBank
from .evaluate import evaluate, preflight
from .local import checkpoint_identity, load_checkpoint
from .tasks import load_tasks


def compare(baseline, skills, ids, *, split="validation"):
    """Paired outcomes with fixed denominator; nulls are explicit, never dropped silently."""
    transitions = Counter()
    usage = {arm: Counter() for arm in ("baseline", "skills")}
    totals = {arm: Counter() for arm in usage}
    pairs = []
    for ident in ids:
        outcomes = []
        for arm, records in (("baseline", baseline), ("skills", skills)):
            r = records.get(ident)
            label = "unscored"
            if r and r.get("evaluation_valid") is True:
                label = "success" if r.get("task_status") == "confirmed_pass" else "not_confirmed"
            outcomes.append(label)
            totals[arm][label] += 1
            if r:
                usage[arm]["recorded"] += 1
                usage[arm]["turns"] += r.get("turns", 0)
                usage[arm]["tool_calls"] += r.get("tool_calls", 0)
                metrics = r.get("execution_metrics", {})
                usage[arm]["reported_tokens"] += metrics.get("reported_tokens", 0)
                usage[arm]["estimated_tokens"] += metrics.get("estimated_tokens", 0)
                usage[arm]["unknown_usage_requests"] += metrics.get("unknown_usage_requests", 0)
        transitions[" -> ".join(outcomes)] += 1
        pairs.append({"task_instance_id": ident, "baseline": outcomes[0], "skills": outcomes[1]})
    gain = transitions["not_confirmed -> success"]
    loss = transitions["success -> not_confirmed"]
    n = gain + loss
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(gain, loss) + 1)) / 2**n) if n else 1.
    rates = {a: totals[a]["success"] / len(ids) for a in totals}
    return {"split": split, "planned_pairs": len(ids), "totals": {k: dict(v) for k, v in totals.items()},
            "success_over_planned": rates, "delta_percentage_points": 100 * (rates["skills"] - rates["baseline"]),
            "both_scored_gains": gain, "both_scored_losses": loss,
            "paired_exact_p_on_scored_discordances": p,
            "transitions": dict(transitions), "usage_totals": {k: dict(v) for k, v in usage.items()},
            "pairs": pairs, "training_updates": False,
            "interpretation": "Matched held-out comparison; unscored cases remain in denominator. No claim of improvement without real outcomes."}


def results(root):
    records = {}
    for p in Path(root).glob("episodes/*/a0/episode_result.json"):
        r = read_json(p)
        ident = r["task_instance_id"]
        if ident in records:
            raise ValueError("duplicate_evaluation_task")
        records[ident] = r
    return records


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--skills", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--limit", type=int, default=64, help="0 = all tasks in the selected split")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if a.output.exists() or a.limit < 0:
        p.error("require a new output directory and nonnegative limit")
    base = RunConfig.model_validate({**read_json(a.config), "skills": None})
    tasks, manifest, sources = load_tasks(a.tasks, a.split)
    tasks.sort(key=lambda t: digest([base.seed, t.task_instance_id]))
    if a.limit:
        tasks = tasks[:a.limit]
    base = RunConfig.model_validate({**base.model_dump(), "judge_inputs_manifest": sources or base.judge_inputs_manifest})
    augmented = RunConfig.model_validate({**base.model_dump(), "skills": read_json(a.skills)})
    preflight(tasks, base, a.split)
    bank = SkillBank(augmented.skills)
    checkpoint = checkpoint_identity(a.model)
    previews = []
    for task in tasks:
        catalog = Path(task.environment_ref).with_name("tools.json")
        tools = read_json(catalog)
        if isinstance(tools, dict):
            tools = tools["tools"]
        selected = bank.retrieve(task, tools)
        previews.append({"task_instance_id": task.task_instance_id,
                         "selected": selected, "workflow_selected": any(s["kind"] == "workflow" for s in selected)})
    a.output.mkdir(parents=True)
    write_json(a.output / "plan.json", {
        "schema_version": "awm-skill-comparison-v1", "split": a.split, "sampling": "greedy",
        "checkpoint": checkpoint, "task_manifest": manifest,
        "tasks": [t.task_instance_id for t in tasks], "baseline": base.model_dump(),
        "skills": augmented.model_dump(), "bank_sha256": bank.sha256,
        "source_hashes": {str(f.relative_to(ROOT)): file_hash(f) for f in sorted((ROOT / "harness_awm").rglob("*.py"))},
        "training_updates": False})
    write_json(a.output / "retrieval_preview.json", previews)
    status = {"stage": "prepared", "evaluation_run": False, "planned_tasks": len(tasks),
              "tasks_with_skills": sum(bool(r["selected"]) for r in previews),
              "tasks_with_workflows": sum(r["workflow_selected"] for r in previews)}
    write_json(a.output / "status.json", status)
    if not a.execute:
        print(json.dumps(status, indent=2))
        return
    try:
        model, tokenizer = load_checkpoint(a.model, device="cuda", dtype="bfloat16", trainable=False)
        async def run():
            for arm, config in (("baseline", base), ("skills", augmented)):
                await evaluate(model, tokenizer, tasks, config, a.output / arm,
                               split=a.split, checkpoint=checkpoint, task_manifest=manifest, sampling="greedy")
        asyncio.run(run())
        report = compare(results(a.output / "baseline"), results(a.output / "skills"),
                         [t.task_instance_id for t in tasks], split=a.split)
        write_json(a.output / "comparison.json", report)
        status.update(stage="complete", evaluation_run=True)
    except BaseException as exc:
        status.update(stage="incomplete", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(a.output / "status.json", status)


if __name__ == "__main__":
    main()
