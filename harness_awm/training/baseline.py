"""Prepare and run frozen Qwen held-out baselines, with an optional pilot."""
import argparse
import asyncio
import hashlib
import math
import os
import subprocess
import tempfile
from pathlib import Path

from ..contracts import ROOT, RunConfig
from ..io import digest, file_hash, now, read_json, write_json
from ..isolation import clean_env, command
from .evaluate import evaluate, preflight
from .local import checkpoint_identity, load_checkpoint
from .tasks import load_tasks, select_tasks


def task_id(task):
    return f"{task.scenario}:{task.native_task_id}"


def select_pilot(tasks, count=20, seed=20261106):
    if count < 1 or len(tasks) < count:
        raise ValueError("insufficient pilot tasks")
    ordered = sorted(tasks, key=lambda t: (hashlib.sha256(f"{seed}:{task_id(t)}".encode()).hexdigest(), task_id(t)))
    selected, scenarios = [], set()
    for task in ordered:
        if task.scenario not in scenarios:
            selected.append(task)
            scenarios.add(task.scenario)
            if len(selected) == count:
                return selected
    ids = {t.task_instance_id for t in selected}
    return (selected + [t for t in ordered if t.task_instance_id not in ids])[:count]


def pilot_gate(report):
    count = report["planned_candidates"]
    if (count < 1 or report["settled_candidates"] != count
            or report["evaluation"]["scored"] != count
            or report.get("stop_reason") != "all_tasks_attempted"):
        raise ValueError("pilot incomplete or has evaluation/infrastructure faults; inspect per_task.csv before full validation")


def projected_budget(report, total=589):
    pilot_gate(report)
    count, usage = report["planned_candidates"], report["budget"]
    requests, tokens = usage["physical_requests"], usage["tokens_accounted_or_reserved"]
    if requests < 1 or tokens < 1:
        raise ValueError("pilot has no real Judge usage")
    return {"max_physical_requests": math.ceil(requests / count * total * 1.5),
            "max_total_tokens": math.ceil(tokens / count * total * 1.5),
            "estimated_wall_seconds": report["wall_seconds"] / count * total,
            "pilot_tasks": count, "full_tasks": total, "budget_margin": 1.5,
            "token_basis": "accounted_or_reserved_including_retries_and_unknown_usage"}


def strict_probe(config, folder):
    """Exercise the actual worker isolation command; never silently downgrade."""
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    result = {"isolation": config.isolation, "passed": False}
    try:
        if config.isolation != "strict":
            raise ValueError("baseline requires strict isolation")
        with tempfile.TemporaryDirectory(prefix="awm-strict-outside-", dir=folder.parent) as outside:
            sentinel = Path(outside) / "sentinel"
            sentinel.write_text("outside")
            code = """import pathlib, sqlite3, sys
p = pathlib.Path(sys.argv[1])
for mode in ('r', 'w'):
    try:
        with p.open(mode):
            pass
    except (FileNotFoundError, PermissionError):
        pass
    else:
        raise RuntimeError('outside access allowed: ' + mode)
with sqlite3.connect('probe.sqlite') as db:
    db.execute('CREATE TABLE IF NOT EXISTS probe (value TEXT)')
    db.execute("INSERT INTO probe VALUES ('ok')")
print('strict_runtime_ok')
"""
            argv = command(config, [config.runtime_python, "-c", code, str(sentinel)], [folder])
            result["command"] = argv
            completed = subprocess.run(argv, env=clean_env(config.timezone), capture_output=True, text=True, timeout=30)
        result.update(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr,
                      passed=completed.returncode == 0 and completed.stdout.strip() == "strict_runtime_ok")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    write_json(folder / "result.json", result)
    return result



def isolation_probe(config, folder):
    if config.isolation == 'private-root':
        from ..isolation import private_root_command
        import sqlite3
        folder = Path(folder).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        report = {'isolation': 'private-root', 'passed': False}
        try:
            with tempfile.TemporaryDirectory(dir=folder.parent) as outside, tempfile.TemporaryDirectory(dir=folder) as jail:
                sentinel = Path(outside) / 'sentinel'
                sentinel.write_text('outside')
                cmd = private_root_command(config, jail, 'probe', sentinel)
                proc = subprocess.run(cmd, env=clean_env(config.timezone), capture_output=True, text=True,
                                      stdin=subprocess.DEVNULL, timeout=30, check=False)
                report.update(returncode=proc.returncode, stderr=proc.stderr[-2000:])
                if proc.returncode == 0 and proc.stdout.strip() == 'private_root_ok' and sentinel.read_text() == 'outside':
                    with sqlite3.connect(Path(jail) / 'probe.sqlite') as db:
                        report['passed'] = db.execute('SELECT value FROM probe').fetchall() == [(1,)]
        except (OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as exc:
            report['error'] = str(exc)
        write_json(folder / 'result.json', report)
        return report
    if config.isolation == "strict":
        return strict_probe(config, folder)
    if config.isolation != "landlock":
        raise ValueError("baseline requires strict or landlock isolation")
    from ..landlock import abi_version
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    result = {"isolation": "landlock", "passed": False,
              "pid_namespace": False, "network_namespace": False,
              "limitations": "No PID/network isolation; ABI < 3 does not restrict truncate; metadata operations are not fully restricted."}
    # A sibling sentinel is outside the worker allowlist; no real secrets used.
    with tempfile.TemporaryDirectory(prefix="awm-isolation-probe-") as outside:
        sentinel = Path(outside) / "sentinel"
        sentinel.write_text("outside")
        code = """import os, pathlib, sqlite3, subprocess, sys
p = pathlib.Path(sys.argv[1])
for mode in ('r', 'w'):
    try:
        with p.open(mode):
            pass
    except PermissionError:
        pass
    else:
        raise RuntimeError('outside access allowed: ' + mode)
with sqlite3.connect('probe.sqlite') as db:
    db.execute('CREATE TABLE IF NOT EXISTS probe (value TEXT)')
    db.execute("INSERT INTO probe VALUES ('ok')")
assert pathlib.Path(os.environ['TMPDIR']).is_dir()
child = subprocess.run([sys.executable, '-c', 'open(__import__("sys").argv[1]).read()', str(p)], capture_output=True)
assert child.returncode != 0 and b'PermissionError' in child.stderr
print('landlock_runtime_ok')
"""
        try:
            result["landlock_abi"] = abi_version()
            argv = command(config, [config.runtime_python, "-c", code, str(sentinel)], [folder])
            completed = subprocess.run(argv, env=clean_env(config.timezone), capture_output=True, text=True, timeout=30)
            result.update(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr,
                          passed=completed.returncode == 0 and completed.stdout.strip() == "landlock_runtime_ok")
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
    write_json(folder / "result.json", result)
    return result


def source_hashes():
    return {str(p.relative_to(ROOT)): file_hash(p) for p in (ROOT / "harness_awm").rglob("*.py")}


def prepare(output, tasks_path, model_path, runtime_python, template, *, include_test=False, skip_pilot=False, unlimited=False, isolation="strict"):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("prepare requires a new output directory")
    if skip_pilot and not unlimited:
        raise ValueError("skipping pilot requires an explicit unlimited budget; no usage estimate is available")
    raw = read_json(template)
    # Preserve a venv's interpreter symlink: resolving it loses sys.prefix.
    raw.update(runtime_python=str(Path(runtime_python).absolute()), seed=20261106,
               purpose="evaluation", actor="training", judge="api", isolation=isolation,
               judge_protocol="rubric_v2", protocol_id="awm-single-json-v2", reward_admissions=None,
               max_turns=32, max_tool_calls=32, context_limit=32768, max_output_tokens=2048,
               candidates_per_task=1, max_physical_requests=1000, max_total_tokens=10000000)
    if unlimited:
        raw.update(max_physical_requests=None, max_total_tokens=None)
    raw["judge_api"].update(base_url=os.environ.get("OPENAI_BASE_URL") or raw["judge_api"]["base_url"],
                            model=os.environ.get("OPENAI_MODEL") or raw["judge_api"]["model"])
    config = RunConfig.model_validate(raw)
    if config.judge_api.model != "gpt-6-luna":
        raise ValueError("baseline protocol requires gpt-6-luna; check local Judge configuration")
    tasks, task_manifest, sources = load_tasks(tasks_path, "validation")
    if len(tasks) != 589:
        raise ValueError("baseline requires all 589 frozen validation tasks")
    if sources:
        config = config.model_copy(update={"judge_inputs_manifest": sources})
    checked = preflight(tasks, config, "validation")
    if include_test:
        test_tasks, _, test_sources = load_tasks(tasks_path, "test")
        if len(test_tasks) != 588 or test_sources != sources:
            raise ValueError("baseline requires all 588 frozen test tasks with the same Judge sources")
        checked = {"validation": checked, "test": preflight(test_tasks, config, "test")}
    checkpoint = checkpoint_identity(model_path)
    lock = read_json(ROOT / "configs/training/qwen3.5-2b-model.lock.json")
    for asset in lock["files"]:
        if file_hash(Path(model_path) / asset["name"]) != asset["sha256"]:
            raise ValueError("base model lock mismatch: " + asset["name"])
    pilot = [] if skip_pilot else select_pilot(tasks, seed=config.seed)
    plan = {"schema_version": "awm-baseline-plan-v1", "created_at": now(),
            "tasks_path": str(Path(tasks_path).resolve()), "task_manifest": task_manifest,
            "checkpoint": checkpoint, "split": "validation", "full_task_count": len(tasks),
            "splits": ["validation", "test"] if include_test else ["validation"],
            "skip_pilot": skip_pilot, "unlimited_judge_budget": unlimited,
            "pilot_task_ids": [task_id(t) for t in pilot], "selection": "sha256(seed:task_id), distinct scenarios first",
            "sampling": "greedy", "dtype": "bfloat16", "device": "cuda", "enable_thinking": False,
            "config": config.model_dump(), "source_hashes": source_hashes(), "static_preflight": checked}
    output.mkdir(parents=True)
    write_json(output / "plan.json", plan)
    write_json(output / "plan.sha256.json", {"sha256": file_hash(output / "plan.json")})
    write_json(output / "pilot_config.json", config.model_dump())
    write_json(output / "pilot_tasks.json", {"seed": config.seed, "task_ids": plan["pilot_task_ids"]})
    probe = isolation_probe(config, output / "strict_probe")
    status = {"stage": "prepared" if probe["passed"] else f"blocked_{config.isolation}_isolation",
              "evaluation_started": False, "static_preflight": checked,
              "isolation": config.isolation, "isolation_probe": probe,
              "strict_isolation_passed": config.isolation == "strict" and probe["passed"], "training_updates": False}
    write_json(output / "status.json", status)
    return status


async def execute(output):
    output = Path(output).resolve()
    plan = read_json(output / "plan.json")
    if file_hash(output / "plan.json") != read_json(output / "plan.sha256.json")["sha256"]:
        raise ValueError("baseline plan changed")
    if source_hashes() != plan["source_hashes"]:
        raise ValueError("harness changed since preparation; prepare a new baseline plan")
    config = RunConfig.model_validate(plan["config"])
    tasks, manifest, sources = load_tasks(plan["tasks_path"], "validation")
    if manifest != plan["task_manifest"] or sources != config.judge_inputs_manifest:
        raise ValueError("runtime task bundle changed")
    preflight(tasks, config, "validation")
    checkpoint = checkpoint_identity(plan["checkpoint"]["path"])
    if checkpoint != plan["checkpoint"]:
        raise ValueError("base checkpoint changed")
    probe = isolation_probe(config, output / "strict_probe_execute")
    if not probe["passed"]:
        write_json(output / "status.json", {"stage": f"blocked_{config.isolation}_isolation", "evaluation_started": False})
        raise RuntimeError(f"{config.isolation} isolation unavailable; see strict_probe_execute/result.json")
    if not os.environ.get(config.judge_api.api_key_env, "").strip():
        raise ValueError("missing Judge API key")
    splits = plan.get("splits", ["validation"])
    if any((output / split).exists() for split in splits):
        raise ValueError("full validation output already exists; never overwrite or select best attempts")
    pilot_tasks = select_tasks(tasks, plan["pilot_task_ids"]) if plan["pilot_task_ids"] else []
    model, tokenizer = load_checkpoint(checkpoint["path"], device=plan["device"], dtype=plan["dtype"])
    pilot_path = output / "pilot"
    if plan.get("skip_pilot"):
        pilot = None
    elif pilot_path.exists():
        prior = read_json(pilot_path / "manifest.json")
        if (prior["checkpoint"] != checkpoint or prior["config"] != config.model_dump()
                or prior["harness_source_hashes"] != {str(Path(k).relative_to("harness_awm")): v for k, v in plan["source_hashes"].items()}
                or prior["tasks"] != [t.model_dump() for t in pilot_tasks] or prior["sampling"] != plan["sampling"]):
            raise ValueError("existing pilot protocol mismatch")
        pilot = read_json(pilot_path / "metrics.json")
    else:
        write_json(output / "status.json", {"stage": "pilot_running", "evaluation_started": True})
        pilot = await evaluate(model, tokenizer, pilot_tasks, config, pilot_path, split="validation",
                               checkpoint=checkpoint, task_manifest=manifest, sampling=plan["sampling"])
    try:
        if pilot is not None:
            pilot_gate(pilot)
        projection = ({"max_physical_requests": None, "max_total_tokens": None,
                       "mode": "unlimited_by_user_request"} if plan.get("unlimited_judge_budget")
                      else projected_budget(pilot, len(tasks)))
    except ValueError as exc:
        write_json(output / "status.json", {"stage": "pilot_needs_review", "error": str(exc)})
        raise
    write_json(output / "budget_projection.json", projection)
    full_config = config.model_copy(update={k: projection[k] for k in ("max_physical_requests", "max_total_tokens")})
    reports = {}
    for split in splits:
        split_tasks, split_manifest, split_sources = load_tasks(plan["tasks_path"], split)
        if split_manifest != manifest or split_sources != config.judge_inputs_manifest:
            raise ValueError("held-out runtime bundle changed")
        preflight(split_tasks, full_config, split)
        write_json(output / f"{split}_config.json", full_config.model_dump())
        write_json(output / "status.json", {"stage": f"{split}_running", "evaluation_started": True})
        report = await evaluate(model, tokenizer, split_tasks, full_config, output / split, split=split,
                                checkpoint=checkpoint, task_manifest=manifest, sampling=plan["sampling"])
        reports[split] = {"planned": len(split_tasks), "settled": report["settled_candidates"],
                          "scored": report["evaluation"]["scored"], "report": str(output / split / "report_zh.md")}
    complete = all(r["planned"] == r["settled"] for r in reports.values())
    prefix = "validation" if splits == ["validation"] else "evaluation"
    status = {"stage": f"{prefix}_finished" if complete else f"{prefix}_incomplete",
              "evaluation_started": True, "training_updates": False,
              "splits": reports}
    write_json(output / "status.json", status)
    return status


def main():
    import json
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prep = commands.add_parser("prepare", help="freeze protocol and probe isolation; no model or API calls")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--tasks", type=Path, required=True)
    prep.add_argument("--model", type=Path, required=True)
    prep.add_argument("--runtime-python", type=Path, required=True)
    prep.add_argument("--config", type=Path, default=ROOT / "configs/training/evaluate-local.example.json")
    prep.add_argument("--isolation", choices=("strict", "landlock"), default="strict",
                      help="explicit backend; landlock needs no namespaces and has weaker process/metadata isolation")
    prep.add_argument("--include-test", action="store_true", help="also evaluate the frozen 588-task test split")
    prep.add_argument("--skip-pilot", action="store_true", help="directly run full held-out splits")
    prep.add_argument("--unlimited-judge-budget", action="store_true", help="no cumulative API request/token cap; retain accounting")
    run = commands.add_parser("execute", help="run pilot and gated full validation, never train")
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.output, args.tasks, args.model, args.runtime_python, args.config,
                         include_test=args.include_test, skip_pilot=args.skip_pilot, unlimited=args.unlimited_judge_budget, isolation=args.isolation)
    else:
        result = asyncio.run(execute(args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
