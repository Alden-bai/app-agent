"""Continue a frozen baseline with disjoint GPU workers, preserving every prior outcome.

The handoff waits for ALL started episodes and external requests to finish.
It never interrupts a model turn or Judge request and never resamples old failures.
"""
import argparse
import asyncio
import fcntl
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness_awm.backend import Budget
from harness_awm.contracts import RunConfig
from harness_awm.io import file_hash, now, read_json, verify_seal, write_json
from harness_awm.stages import metrics, official_hashes
from harness_awm.training.baseline import isolation_probe, source_hashes
from harness_awm.training.evaluate import evaluate, preflight
from harness_awm.training.local import checkpoint_identity, load_checkpoint
from harness_awm.training.tasks import load_tasks


class BusyBoundary(RuntimeError):
    pass


def budget_paths(source, split, visited=None):
    """Each physical ledger appears once, including all previous continuations."""
    source = Path(source).resolve()
    visited = set() if visited is None else visited
    if source in visited:
        raise ValueError("cyclic continuation history")
    visited.add(source)
    paths = {p.resolve() for p in [source / split / "budget.sqlite",
             *source.glob(f"workers/w*/{split}/budget.sqlite")] if p.exists()}
    if (source / "plan.json").exists():
        previous = read_json(source / "plan.json").get("source_run")
        if previous:
            paths.update(budget_paths(previous, split, visited))
    return sorted(paths)


def finished_episodes(source, *, verify=False, allow_unstarted=False, deferred=None):
    """Return exact prior attempts only at a fully scored, no-request boundary."""
    source = Path(source)
    kept = {}
    for split in ("validation", "test"):
        run = source / split
        if not (run / "manifest.json").exists():
            continue
        manifest = read_json(run / "manifest.json")
        for budget in budget_paths(source, split):
            with sqlite3.connect(budget.resolve().as_uri() + "?mode=ro", uri=True) as db:
                if db.execute("SELECT count(*) FROM requests WHERE status='in_flight'").fetchone()[0]:
                    raise BusyBoundary("Judge request still in flight")
                if db.execute("SELECT count(*) FROM holds").fetchone()[0]:
                    raise BusyBoundary("outstanding budget reservation")
        for candidate in manifest["candidates"]:
            folder = run / candidate["path"]
            if not folder.exists():
                continue
            result_path = folder / "episode_result.json"
            if not result_path.exists() or not (folder / "evidence_seal.json").exists():
                event_path = folder / "events.jsonl"
                events = [json.loads(line) for line in event_path.read_text().splitlines()] if event_path.exists() else []
                if (allow_unstarted and not result_path.exists()
                        and not any(e["event_type"] in {"model_request", "model_response", "tool_started", "execution_closed"}
                                    for e in events)
                        and not list((folder / "model_turns").glob("*/input.json"))):
                    if deferred is not None:
                        deferred[candidate["task_instance_id"]] = str(folder.resolve())
                    continue
                raise BusyBoundary("model/environment episode still running")
            result = read_json(result_path)
            # Keep completed evaluator faults too: no selection by score or validity.
            if result.get("evaluation_status") not in {"valid", "evaluation_error", "rubric_error", "invalid_sample"}:
                raise BusyBoundary("episode Judge has not completed")
            if verify:
                verify_seal(folder)
            ident = candidate["task_instance_id"]
            if ident in kept or result.get("task_instance_id") != ident:
                raise ValueError("duplicate or mismatched prior task")
            kept[ident] = {"split": split, "path": str(folder.resolve()),
                           "result_sha256": file_hash(result_path),
                           "seal_sha256": file_hash(folder / "evidence_seal.json")}
    return kept


def safe_handoff(source, pid, timeout=600, expected_arguments=None):
    """Freeze, recheck, then terminate only after ALL requests have finished."""
    source = Path(source).resolve()
    proc = Path(f"/proc/{pid}")
    command = (proc / "cmdline").read_bytes()
    if expected_arguments:
        argv = command.split(b"\0")
        if not all(str(arg).encode() in argv for arg in expected_arguments):
            raise ValueError("PID is not the specified parallel worker")
    elif b"harness_awm.training.baseline" not in command or str(source).encode() not in command:
        raise ValueError("PID is not the specified baseline")
    start_identity = (proc / "stat").read_text().split()[21]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            finished_episodes(source, allow_unstarted=True)
        except BusyBoundary:
            time.sleep(.001)
            continue
        if (proc / "stat").read_text().split()[21] != start_identity:
            raise ValueError("PID identity changed")
        os.kill(pid, signal.SIGSTOP)
        try:
            for _ in range(100):
                if (proc / "stat").read_text().split()[2] in {"T", "t"}:
                    break
                time.sleep(.01)
            else:
                raise RuntimeError("process did not stop for boundary verification")
            try:
                deferred = {}
                kept = finished_episodes(source, verify=True, allow_unstarted=True, deferred=deferred)
            except BusyBoundary:
                continue
            record = {"at": now(), "pid": pid, "reason": "user_requested_parallel_throughput",
                      "in_flight_judge_requests": 0, "unfinished_started_episodes": 0,
                      "kept": kept, "actor_resampling": False,
                      "deferred_before_first_model_request": deferred}
            write_json(source / "parallel_handoff.json", record)
            # These children can only belong to environment startup: there are no
            # in-flight model/Judge requests at the verified boundary.
            children_path = proc / "task" / str(pid) / "children"
            children = [int(c) for c in children_path.read_text().split()]
            for child in children:
                try:
                    if os.getpgid(child) == child:
                        os.killpg(child, signal.SIGTERM)
                    else:
                        os.kill(child, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            os.kill(pid, signal.SIGTERM)
            break
        finally:
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
    else:
        raise TimeoutError("No safe handoff boundary; serial evaluation remains running")
    for _ in range(100):
        if not proc.exists() or (proc / "stat").read_text().split()[2] == "Z":
            return record
        time.sleep(.05)
    raise RuntimeError("serial process still alive; will not launch overlapping workers")


def parallel_handoff(source):
    """Drain each existing worker independently at its own safe task boundary."""
    source = Path(source).resolve()
    plan = read_json(source / "plan.json")
    if file_hash(source / "plan.json") != read_json(source / "plan.sha256.json")["sha256"]:
        raise ValueError("source plan changed")
    if file_hash(source / "driver_frozen.py") != plan["driver_sha256"]:
        raise ValueError("missing exact source driver snapshot")
    launch = read_json(source / "launch.json")
    pids = launch["worker_pids"]
    if len(pids) != plan["workers"]:
        raise ValueError("worker launch incomplete")

    def drain(item):
        index, pid = item
        folder = source / "workers" / f"w{index:02d}"
        proc = Path(f"/proc/{pid}")
        if not proc.exists() or (proc / "stat").read_text().split()[2] == "Z":
            finished_episodes(folder, verify=True)
            return {"worker": index, "already_exited": True}
        result = safe_handoff(folder, pid, expected_arguments=[str(Path(__file__).resolve()), "worker",
                              "--output", str(source), "--index", str(index)])
        print(json.dumps({"drained_worker": index, "kept": len(result["kept"])}), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=len(pids)) as executor:
        drained = list(executor.map(drain, enumerate(pids)))
    supervisor = Path(f"/proc/{launch['supervisor_pid']}")
    if supervisor.exists() and (supervisor / "stat").read_text().split()[2] != "Z":
        argv = (supervisor / "cmdline").read_bytes().split(b"\0")
        if b"run" not in argv or str(source).encode() not in argv:
            raise ValueError("supervisor identity changed")
        # Supervisor only aggregates files; every model worker is already stopped.
        os.kill(launch["supervisor_pid"], signal.SIGTERM)
        for _ in range(100):
            if not supervisor.exists() or (supervisor / "stat").read_text().split()[2] == "Z":
                break
            time.sleep(.05)
        else:
            raise RuntimeError("old supervisor did not exit")
    deferred = {}
    kept = finished_episodes(source, verify=True, allow_unstarted=True, deferred=deferred)
    record = {"at": now(), "reason": "user_requested_memory_target_90GB", "kept": kept,
              "in_flight_judge_requests": 0, "actor_resampling": False,
              "deferred_before_first_model_request": deferred, "drained_workers": len(drained)}
    write_json(source / "parallel_handoff.json", record)
    return {"kept": len(kept), "deferred_before_inference": len(deferred), "workers_drained": len(drained)}


def partition(tasks, kept, workers):
    if workers < 1:
        raise ValueError("workers must be positive")
    pending = [t for t in tasks if t.task_instance_id not in kept]
    return [[t.task_instance_id for t in pending[index::workers]] for index in range(workers)]


def prepare(source, output, workers):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("parallel output already exists")
    original = read_json(source / "plan.json")
    if file_hash(source / "plan.json") != read_json(source / "plan.sha256.json")["sha256"]:
        raise ValueError("source plan hash mismatch")
    if source_hashes() != original["source_hashes"]:
        raise ValueError("harness implementation changed")
    config = RunConfig.model_validate(original["config"])
    if config.max_physical_requests is not None or config.max_total_tokens is not None:
        raise ValueError("parallel controller requires explicitly authorized unlimited cumulative budget")
    for split in ("validation", "test"):
        prior_manifest = source / split / "manifest.json"
        if prior_manifest.exists():
            prior = read_json(prior_manifest)
            if file_hash(prior_manifest) != read_json(prior_manifest.with_name("manifest.sha256.json"))["sha256"]:
                raise ValueError("prior run manifest changed")
            if (prior["config"] != config.model_dump() or prior["checkpoint"] != original["checkpoint"]
                    or prior["sampling"] != original["sampling"] or prior["task_manifest"] != original["task_manifest"]):
                raise ValueError("prior run inference or Judge protocol differs")
    deferred = {}
    kept = finished_episodes(source, verify=True, allow_unstarted=True, deferred=deferred)
    handoff = read_json(source / "parallel_handoff.json")
    if kept != handoff["kept"] or deferred != handoff.get("deferred_before_first_model_request", {}):
        raise ValueError("source results changed after verified handoff")
    all_tasks, assignment, manifests = {}, {}, {}
    for split, count in (("validation", 589), ("test", 588)):
        tasks, manifest, sources = load_tasks(original["tasks_path"], split)
        if len(tasks) != count or manifest != original["task_manifest"] or sources != config.judge_inputs_manifest:
            raise ValueError("frozen task binding changed")
        preflight(tasks, config, split)
        all_tasks[split] = tasks
        assignment[split] = partition(tasks, kept, workers)
        manifests[split] = manifest
    valid_ids = {t.task_instance_id for tasks in all_tasks.values() for t in tasks}
    if not set(kept) <= valid_ids:
        raise ValueError("prior results contain unknown tasks")
    checkpoint = checkpoint_identity(original["checkpoint"]["path"])
    if checkpoint != original["checkpoint"]:
        raise ValueError("checkpoint changed")
    output.mkdir(parents=True)
    probe = isolation_probe(config, output / "isolation_probe")
    if not probe["passed"]:
        raise RuntimeError("worker isolation probe failed")
    plan = {**original, "schema_version": "awm-parallel-baseline-v1", "created_at": now(),
            "source_run": str(source), "workers": workers, "kept": kept,
            "cpu_threads_per_worker": 1 if workers > 8 else 2,
            "release_cuda_cache_between_tasks": True,
            "deferred_before_first_model_request": deferred,
            "assignment": assignment, "driver_sha256": file_hash(__file__),
            "source_plan_sha256": file_hash(source / "plan.json"),
            "execution_change": "independent model replicas; concurrent tasks; same per-task inference and Judge"}
    write_json(output / "plan.json", plan)
    write_json(output / "plan.sha256.json", {"sha256": file_hash(output / "plan.json")})
    for split, tasks in all_tasks.items():
        destinations = {}
        for worker, ids in enumerate(assignment[split]):
            for i, ident in enumerate(ids):
                destinations[ident] = str(output / "workers" / f"w{worker:02d}" / split / "episodes" / f"t{i:05d}" / "a0")
        candidates = []
        for i, task in enumerate(tasks):
            prior = kept.get(task.task_instance_id)
            if prior and prior["split"] != split:
                raise ValueError("prior split mismatch")
            candidates.append({"rollout_id": f"t{i:05d}", "task_instance_id": task.task_instance_id,
                               "scenario": task.scenario,
                               "path": prior["path"] if prior else destinations[task.task_instance_id],
                               "reused_prior_result": bool(prior)})
        manifest = {"schema_version": "awm-run-v1", "stage": "evaluate", "run_id": output.name + "-" + split,
                    "created_at": now(), "config": config.model_dump(), "tasks": [t.model_dump() for t in tasks],
                    "candidates": candidates, "checkpoint": checkpoint, "evaluation_split": split,
                    "task_manifest": manifests[split], "sampling": original["sampling"], "seed": config.seed,
                    "harness_source_hashes": source_hashes(), "official_source_hashes": official_hashes(config),
                    "training_updates": False, "workers": workers, "source_run": str(source),
                    "driver_sha256": plan["driver_sha256"], "reused_results": sum(c["reused_prior_result"] for c in candidates)}
        write_json(output / split / "manifest.json", manifest)
        write_json(output / split / "manifest.sha256.json", {"sha256": file_hash(output / split / "manifest.json")})
    write_json(output / "status.json", {"stage": "prepared", "workers": workers, "reused": len(kept)})
    write_json(source / "status.json", {"stage": "continued_in_parallel_run", "continuation": str(output),
                                      "preserved_results": len(kept), "training_updates": False})
    return plan


def validated_plan(output):
    output = Path(output)
    if file_hash(output / "plan.json") != read_json(output / "plan.sha256.json")["sha256"]:
        raise ValueError("parallel plan changed")
    plan = read_json(output / "plan.json")
    if source_hashes() != plan["source_hashes"] or file_hash(__file__) != plan["driver_sha256"]:
        raise ValueError("parallel code changed")
    for prior in plan["kept"].values():
        folder = Path(prior["path"])
        if file_hash(folder / "episode_result.json") != prior["result_sha256"]:
            raise ValueError("prior result changed after handoff")
    return plan


def worker_main(output, index):
    import torch
    plan = validated_plan(output)
    if not 0 <= index < plan["workers"]:
        raise ValueError("unknown worker")
    torch.set_num_threads(plan.get("cpu_threads_per_worker", 2))
    torch.set_num_interop_threads(1)
    config = RunConfig.model_validate(plan["config"])
    # Staggered launches keep CPU checkpoint loading below the container RAM limit.
    model, tokenizer = load_checkpoint(plan["checkpoint"]["path"], device=plan["device"], dtype=plan["dtype"])
    from harness_awm.runner import EpisodeRunner

    class WorkerRunner(EpisodeRunner):
        async def run(self, *args, **kwargs):
            try:
                return await super().run(*args, **kwargs)
            finally:
                # Release unused allocator blocks after scoring, not model weights
                # or active KV caches. This reduces peaks when many replicas share a GPU.
                if plan.get("release_cuda_cache_between_tasks") and plan["device"] == "cuda":
                    torch.cuda.empty_cache()
    for split in ("validation", "test"):
        tasks, manifest, sources = load_tasks(plan["tasks_path"], split)
        if manifest != plan["task_manifest"] or sources != config.judge_inputs_manifest:
            raise ValueError("runtime task bundle changed")
        by_id = {t.task_instance_id: t for t in tasks}
        selected = [by_id[ident] for ident in plan["assignment"][split][index]]
        if not selected:
            continue
        report = asyncio.run(evaluate(model, tokenizer, selected, config,
            Path(output) / "workers" / f"w{index:02d}" / split, split=split,
            checkpoint=plan["checkpoint"], task_manifest=manifest, sampling=plan["sampling"], runner_factory=WorkerRunner))
        print(json.dumps({"worker": index, "split": split, "scored": report["evaluation"]["scored"]}), flush=True)


def aggregate(output, plan, elapsed):
    summaries = {}
    for split in ("validation", "test"):
        report = metrics(Path(output) / split)
        budgets = budget_paths(output, split)
        total = {"physical_requests": 0, "tokens_accounted_or_reserved": 0,
                 "unknown_usage_requests": 0, "held_requests": 0, "held_tokens": 0}
        for path in budgets:
            if path.exists():
                for key, value in Budget(path, 0, 0).summary().items():
                    total[key] += value
        report.update(budget=total, evaluation_split=split, isolation=plan["config"]["isolation"],
                      parallel_wall_seconds=elapsed, workers=plan["workers"], training_updates=False,
                      sampling=plan["sampling"], checkpoint_sha256=plan["checkpoint"]["sha256"],
                      interpretation="Exact task-level union; prior failures retained. Never average shard success rates.")
        write_json(Path(output) / split / "metrics.json", report)
        from harness_awm.evaluation_metrics import write_reports
        write_reports(Path(output) / split, report)
        summaries[split] = {k: report["evaluation"][k] for k in
                            ("planned", "result_records", "scored", "confirmed_successes", "score_coverage")}
    return summaries


def supervise(output):
    output = Path(output).resolve()
    plan = validated_plan(output)
    started = time.monotonic()
    processes, logs = [], []
    with (output / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "launch.json").exists():
            raise ValueError("workers already launched; inspect results before any recovery")
        threads = str(plan.get("cpu_threads_per_worker", 2))
        env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", PYTHONUNBUFFERED="1",
                   OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, TOKENIZERS_PARALLELISM="false", SWANLAB_MODE="offline")
        write_json(output / "launch.json", {"supervisor_pid": os.getpid(), "started_at": now(), "worker_pids": []})
        try:
            for index in range(plan["workers"]):
                log = (output / f"worker_{index:02d}.log").open("xb")
                logs.append(log)
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker", "--output", str(output),
                                            "--index", str(index)], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
                processes.append(process)
                write_json(output / "launch.json", {"supervisor_pid": os.getpid(), "started_at": now(),
                                                    "worker_pids": [p.pid for p in processes]})
                time.sleep(2)
            while True:
                exits = [p.poll() for p in processes]
                summary = aggregate(output, plan, time.monotonic() - started)
                live = sum(code is None for code in exits)
                failed = [i for i, code in enumerate(exits) if code not in (None, 0)]
                stage = "running" if live else "finished" if not failed else "worker_failures"
                write_json(output / "status.json", {"stage": stage, "updated_at": now(), "workers": plan["workers"],
                    "live_workers": live, "worker_exit_codes": exits, "failed_workers": failed,
                    "isolation": plan["config"]["isolation"], "splits": summary, "reused_results": len(plan["kept"])})
                if not live:
                    return
                time.sleep(15)
        finally:
            for log in logs:
                log.close()


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("handoff", "scale-handoff", "prepare", "run", "worker"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.action == "handoff":
        print(json.dumps(safe_handoff(args.source, args.pid)))
    elif args.action == "scale-handoff":
        print(json.dumps(parallel_handoff(args.source)))
    elif args.action == "prepare":
        plan = prepare(args.source, args.output, args.workers)
        print(json.dumps({"workers": plan["workers"], "kept": len(plan["kept"]),
                          "pending": {k: sum(map(len, v)) for k, v in plan["assignment"].items()}}))
    elif args.action == "worker":
        worker_main(args.output, args.index)
    else:
        supervise(args.output)


if __name__ == "__main__":
    main()
