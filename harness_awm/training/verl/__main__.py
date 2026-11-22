"""Prepare/check/train AWM's pinned verl GRPO recipe (no implicit execution)."""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path

from ...contracts import ROOT, RunConfig, TaskSpec
from ...io import digest, file_hash, read_json, write_json
from ...reward_policy import VERSION
from ..grpo import preflight
from ..local import checkpoint_identity, checkpoint_filesystem
from ..tasks import load_tasks, select_tasks
from . import VERL_REVISION
from .coverage import CANDIDATES, POOL_SIZE, SAVE_EVERY, TASK_COUNT, selected_task_ids


def readiness(plan, *, runtime=False, require_rewards=True):
    errors = []
    config = RunConfig.model_validate(plan["run_config"])
    tasks = [TaskSpec.model_validate(row) for row in plan["tasks"]]
    if require_rewards:
        try:
            preflight(tasks, config, plan["group_size"])
        except (ValueError, OSError, KeyError) as exc:
            errors.append(f"reward/task preflight: {exc}")
    if not Path(config.runtime_python).is_file() or not os.access(config.runtime_python, os.X_OK):
        errors.append("environment runtime_python is missing or not executable: " + config.runtime_python)
    model_config = read_json(Path(plan["model"]) / "config.json")
    if (model_config.get("model_type") != "qwen3_5_text"
            or model_config.get("architectures") != ["Qwen3_5ForCausalLM"]):
        errors.append("model must be a text-only Qwen3_5ForCausalLM SFT checkpoint (qwen3_5_text)")
    if any(model_config.get(k, 0) != 0 for k in ("attention_dropout", "hidden_dropout", "classifier_dropout")):
        errors.append("on-policy recipe requires dropout=0")
    if require_rewards and not os.environ.get(config.judge_api.api_key_env, "").strip():
        errors.append("Judge credential environment variable is not set: " + config.judge_api.api_key_env)
    if runtime:
        try:
            verify_runtime(plan.get("runtime_profile", "pinned"))
        except (ValueError, ImportError, OSError) as exc:
            errors.append(f"verl runtime: {exc}")
        try:
            import torch
            if not torch.cuda.is_available() or torch.cuda.device_count() < plan["gpus"]:
                errors.append("requested CUDA devices unavailable in this Python runtime")
        except ImportError as exc:
            errors.append(str(exc))
        from ..baseline import isolation_probe
        with tempfile.TemporaryDirectory(prefix="awm-grpo-isolation-") as folder:
            probe = isolation_probe(config, folder)
        if not probe["passed"]:
            errors.append("isolation probe failed: " + str(probe))
    return {"ready": not errors, "errors": errors, "runtime_checked": runtime,
            "tasks": len(tasks), "backend": "verl", "revision": VERL_REVISION,
            "training_started": False, "judge_requests": 0}


def verify_runtime(profile="pinned"):
    import verl
    if profile not in {"pinned", "local-sdpa", "local-sdpa-cu128"}:
        raise ValueError("unknown verl runtime profile")
    filenames = {'pinned': 'verl-runtime.json', 'local-sdpa': 'verl-local-sdpa-runtime.json',
                 'local-sdpa-cu128': 'verl-nioflow-cu128-runtime.json'}
    lock = read_json(ROOT / 'requirements' / filenames[profile])
    if profile == 'local-sdpa-cu128':
        import torch
        from .compat import cuda_home
        cuda_home(profile)
        if torch.version.cuda != lock['cuda']:
            raise ValueError(f'expected CUDA 12.8 PyTorch, found {torch.version.cuda}')
        lock['source_sha256'] = read_json(ROOT / 'requirements' / lock['source_lock'])['source_sha256']
    for package, expected in lock["packages"].items():
        try:
            actual = metadata.version(package)
            if package != "verl" or profile == "pinned":
                actual = actual.split("+")[0]
        except metadata.PackageNotFoundError as exc:
            raise ValueError(f"missing dependency: {package}") from exc
        if actual != expected:
            raise ValueError(f"{package}: expected {expected}, found {actual}")
    for relative, expected in lock["source_sha256"].items():
        if file_hash(Path(verl.__file__).parent / relative) != expected:
            raise ValueError("verl source differs from pinned revision: " + relative)


def load_plan(folder):
    folder = Path(folder).resolve()
    plan = read_json(folder / "plan.json")
    if digest(plan) != read_json(folder / "plan.sha256.json")["sha256"]:
        raise ValueError("plan changed; prepare a new run")
    if plan["output"] != str(folder) or plan["verl_revision"] != VERL_REVISION:
        raise ValueError("moved plan or unsupported verl revision; prepare a new run")
    # Preparation already hashes every weight byte. Avoid a second multi-GB
    # network read when the exact file set, inode, size, mtime and ctime agree.
    # Any change, including an in-place edit with restored mtime, rechecks hashes.
    if (not plan.get("checkpoint_filesystem")
            or checkpoint_filesystem(plan["model"]) != plan["checkpoint_filesystem"]):
        if checkpoint_identity(plan["model"]) != plan["checkpoint"]:
            raise ValueError("checkpoint changed since preparation")
    for relative, expected in plan["harness_source_sha256"].items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError("harness changed since preparation; prepare a new run: " + relative)
    return plan


def resolve_prepare_defaults(args):
    full_coverage = getattr(args, "full_coverage", False)
    if full_coverage and (getattr(args, "target_updates", None) is not None
                          or getattr(args, "steps", None) is not None or getattr(args, "task", None)
                          or getattr(args, "group_offset", 0)):
        raise ValueError("full-coverage uses the frozen selection once without an update target or offset")
    if full_coverage and (getattr(args, "groups", None) not in (None, TASK_COUNT)
                          or getattr(args, "group_size", None) not in (None, CANDIDATES)
                          or getattr(args, "save_every", None) not in (None, SAVE_EVERY)):
        raise ValueError("full-coverage requires 2000 tasks, four candidates and a checkpoint every 500 tasks")
    if getattr(args, "steps", None) is not None and getattr(args, "target_updates", None) is not None:
        raise ValueError("steps and target-updates are mutually exclusive")
    defaults = {"group_size": 4, "groups": 1, "concurrency": 4, "gpus": 1,
                "learning_rate": 1e-6, "save_every": 10, "target_updates": None,
                "runtime_profile": "pinned", "rollout_correction": "strict",
                "null_reward_policy": "exclude", "recover_tool_errors": None,
                "rollout_tensor_parallel_size": 1, "optimizer_offload": True,
                "param_offload": False}
    preset = getattr(args, "preset", None)
    if preset is not None:
        if preset not in {"single-gpu-800", "a100x8-27b"}:
            raise ValueError("unknown_GRPO_preset")
        defaults.update(read_json(ROOT / "configs/training" / f"grpo-{preset}.json"))
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    if getattr(args, "steps", None) is not None:
        args.groups = args.steps
        args.target_updates = None
    if full_coverage:
        args.groups, args.group_size, args.save_every = TASK_COUNT, CANDIDATES, SAVE_EVERY
        args.target_updates = None
        args.shuffle_tasks = True
    if args.null_reward_policy not in {"exclude", "zero"}:
        raise ValueError("unknown_null_reward_policy")
    return args


def prepare(args):
    args = resolve_prepare_defaults(args)
    target = getattr(args, "target_updates", None)
    offset = getattr(args, "group_offset", 0)
    if offset < 0:
        raise ValueError("group-offset must be nonnegative")
    if target is not None and not 1 <= target <= args.groups:
        raise ValueError("target-updates must be between 1 and groups")
    if (args.group_size < 2 or args.groups < 1 or args.gpus < 1
            or args.concurrency < 1 or args.save_every < 1
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not 0 < args.gpu_memory_utilization < 1):
        raise ValueError("invalid sizes/rate")
    if args.rollout_tensor_parallel_size < 1 or args.gpus % args.rollout_tensor_parallel_size:
        raise ValueError("rollout tensor parallel size must divide gpus")
    if getattr(args, "pilot_guards", False) and args.gpus != 1:
        raise ValueError("pilot-guards use single-GPU accounting; use the multi-GPU delivery preflight")
    allocator_gb = getattr(args, "allocator_gb", 72)
    gpu_stop_gb = getattr(args, "gpu_stop_gb", 78)
    if not (0 < allocator_gb < gpu_stop_gb < 1000):
        raise ValueError("require 0 < allocator-gb < gpu-stop-gb < 1000")
    config = RunConfig.model_validate(read_json(args.config))
    if args.recover_tool_errors is not None:
        config = config.model_copy(update={"recover_tool_errors": args.recover_tool_errors})
    if (config.recover_tool_errors or args.null_reward_policy == "zero") and config.protocol_id != "awm-single-json-v2":
        raise ValueError("null penalty and tool recovery require the v2 rollout protocol")
    tasks, manifest, sources = load_tasks(args.tasks, "grpo_train")
    tasks = select_tasks(tasks, args.task)
    full_coverage = getattr(args, "full_coverage", False)
    if full_coverage:
        from ...reward_values import COMPOSITE_VERSION
        if manifest.get("counts", {}).get("grpo_train") != POOL_SIZE:
            raise ValueError("expected the frozen training pool")
        if config.judge_protocol != "rubric_v2" or config.purpose == "evaluation" or config.reward_policy != COMPOSITE_VERSION:
            raise ValueError("the selected experiment requires the training rubric and composite quality reward")
        order = selected_task_ids((task.task_instance_id for task in tasks), config.seed)
        by_id = {task.task_instance_id: task for task in tasks}
        tasks = [by_id[key] for key in order]
    elif getattr(args, "shuffle_tasks", False):
        import random
        random.Random(config.seed).shuffle(tasks)
    if not tasks:
        raise ValueError("no training tasks")
    if sources:
        config = config.model_copy(update={"judge_inputs_manifest": sources})
    changes = {}
    for field in ("official_root", "runtime_python", "judge_inputs_manifest", "reward_admissions"):
        if getattr(config, field):
            path = Path(getattr(config, field))
            changes[field] = str(path.absolute() if field == "runtime_python" else path.resolve())
    config = config.model_copy(update=changes)
    model = args.model.resolve()
    filesystem = checkpoint_filesystem(model)
    identity = checkpoint_identity(model)
    if checkpoint_filesystem(model) != filesystem:
        raise ValueError("checkpoint changed during preparation")
    effective_utilization = args.gpu_memory_utilization
    gpu_budget = None
    if getattr(args, "pilot_guards", False):
        from .pilot import rollout_fraction
        total_mib = float(subprocess.check_output(['nvidia-smi', '--id=0',
            '--query-gpu=memory.total', '--format=csv,noheader,nounits'], text=True).strip())
        actor_gb = sum(record['size'] for name, record in filesystem['files'].items()
                       if name.endswith(('.safetensors', '.bin'))) / 10**9
        gpu_budget = {'total_gb': total_mib * 1024**2 / 10**9, 'actor_gb': actor_gb,
                      'target_gb': allocator_gb, 'overhead_gb': 2.0,
                      'requested_utilization': args.gpu_memory_utilization}
        effective_utilization = rollout_fraction(args.gpu_memory_utilization,
            **{k: v for k, v in gpu_budget.items() if k != 'requested_utilization'})
    # Read special-token metadata without loading tensors or importing transformers.
    generation_path = model / "generation_config.json"
    raw_config = read_json(model / "config.json")
    generation = read_json(generation_path) if generation_path.exists() else raw_config.get("text_config", raw_config)
    token_config = read_json(model / "tokenizer_config.json")
    eos = generation.get("eos_token_id")
    eos = list(eos) if isinstance(eos, list) else [eos]
    token = token_config.get("eos_token")
    token = token.get("content") if isinstance(token, dict) else token
    for key, entry in token_config.get("added_tokens_decoder", {}).items():
        if entry.get("content") == token:
            eos.append(int(key))
    # Transformers 5 can store added-token IDs only in tokenizer.json.
    tokenizer_json = model / "tokenizer.json"
    if tokenizer_json.exists():
        for entry in read_json(tokenizer_json).get("added_tokens", []):
            if entry.get("content") == token:
                eos.append(entry["id"])
    if not eos or any(type(t) is not int or t < 0 for t in eos):
        raise ValueError("missing_or_invalid_stop_tokens")
    output = args.output.resolve()
    plan = {"schema_version": "awm-verl-grpo-v1", "verl_revision": VERL_REVISION,
        "output": str(output), "model": str(model), "checkpoint": identity,
        "checkpoint_filesystem": filesystem,
        "run_config": config.model_dump(), "tasks": [t.model_dump() for t in tasks],
        "task_manifest": manifest,
        "task_order": "seeded-shuffle" if getattr(args, "shuffle_tasks", False) else "source",
        "group_size": args.group_size, "groups": args.groups,
        "group_offset": offset,
        "concurrency": min(args.concurrency, args.group_size), "gpus": args.gpus,
        "rollout_tensor_parallel_size": args.rollout_tensor_parallel_size,
        "optimizer_offload": args.optimizer_offload, "param_offload": args.param_offload,
        "learning_rate": args.learning_rate, "gpu_memory_utilization": effective_utilization,
        "gpu_budget": gpu_budget,
        "save_unit": "groups" if full_coverage or getattr(args, "steps", None) is not None else "updates",
        "save_every": args.save_every, "clip": 0.2, "beta": 0.0, "eos_ids": sorted(set(eos)),
        "runtime_profile": getattr(args, "runtime_profile", "pinned"),
        "target_updates": target,
        "full_coverage": full_coverage,
        "preset": getattr(args, "preset", None),
        "null_reward_policy": args.null_reward_policy,
        "training_reward_policy": "null-as-zero-v1" if args.null_reward_policy == "zero" else "admitted-only-v1",
        "pilot_guards": getattr(args, "pilot_guards", False),
        "allocator_gb": allocator_gb, "gpu_stop_gb": gpu_stop_gb,
        "stress_probe": getattr(args, "stress_probe", False),
        "advantage_policy": "population-std-floor-v1", "advantage_std_floor": 0.10,
        "rollout_correction": getattr(args, "rollout_correction", "strict"),
        "checkpoint_dir": str(Path(getattr(args, "checkpoint_dir", None) or output / "checkpoints").resolve()),
        "reward_policy": config.reward_policy, "std": "population", "loss_normalization": "trajectory_token_mean",
        "harness_source_sha256": {str(p.relative_to(ROOT)): file_hash(p)
                                  for p in sorted((ROOT / "harness_awm").rglob("*.py"))}}
    report = readiness(plan)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    write_json(output / "plan.sha256.json", {"sha256": digest(plan)})
    write_json(output / "preflight.json", report)
    return report


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="write a new run plan; never load a model or call Judge")
    for name in ("config", "tasks", "model", "output"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--task", action="append")
    prep.add_argument("--shuffle-tasks", action="store_true", help="Freeze a seed-shuffled order over the selected task pool")
    prep.add_argument("--preset", choices=("single-gpu-800", "a100x8-27b"))
    prep.add_argument("--group-size", type=int)
    prep.add_argument("--groups", type=int)
    prep.add_argument("--full-coverage", action="store_true",
                      help="Run the frozen 2000-task selection once and save every 500 completed tasks")
    prep.add_argument("--steps", type=int, help="Stop after this many rollout groups (including zero-advantage groups); save-every also counts groups")
    prep.add_argument("--group-offset", type=int, default=0,
                      help="starting group number for fresh sampling; does not restore training state")
    prep.add_argument("--concurrency", type=int)
    prep.add_argument("--gpus", type=int)
    prep.add_argument("--rollout-tensor-parallel-size", type=int)
    prep.add_argument("--optimizer-offload", action=argparse.BooleanOptionalAction, default=None)
    prep.add_argument("--param-offload", action=argparse.BooleanOptionalAction, default=None)
    prep.add_argument("--learning-rate", type=float)
    prep.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    prep.add_argument("--save-every", type=int)
    prep.add_argument("--runtime-profile", choices=("pinned", "local-sdpa", "local-sdpa-cu128"))
    prep.add_argument("--target-updates", type=int)
    prep.add_argument("--null-reward-policy", choices=("exclude", "zero"))
    prep.add_argument("--recover-tool-errors", action=argparse.BooleanOptionalAction, default=None)
    prep.add_argument("--pilot-guards", action="store_true")
    prep.add_argument("--stress-probe", action="store_true")
    prep.add_argument("--allocator-gb", type=float, default=72)
    prep.add_argument("--gpu-stop-gb", type=float, default=78)
    prep.add_argument("--checkpoint-dir", type=Path)
    prep.add_argument("--rollout-correction", choices=("strict", "token-is-v1"),
                      help="Explicit same-version inference/learner importance correction; raw sampling logprobs remain intact")
    check = sub.add_parser("check", help="check plan and runtime; never start training")
    check.add_argument("--run", type=Path, required=True)
    check.add_argument("--skip-runtime", action="store_true")
    train = sub.add_parser("train", help="start GPU training and real budgeted Judge calls")
    train.add_argument("--run", type=Path, required=True)
    probe = sub.add_parser("probe", help="exercise GPU rollout, weight sync and logprobs without Judge or updates")
    probe.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    try:
        if args.command == "prepare":
            report = prepare(args)
        else:
            plan = load_plan(args.run)
            report = readiness(plan, runtime=not getattr(args, "skip_runtime", False),
                               require_rewards=args.command != "probe")
            if args.command in {"train", "probe"}:
                if not report["ready"]:
                    raise ValueError("preflight failed: " + "; ".join(report["errors"]))
                from .config import compose_config
                from omegaconf import OmegaConf
                config = compose_config(plan)
                from omegaconf import open_dict
                with open_dict(config):
                    config.awm.probe_only = args.command == "probe"
                os.environ["AWM_VERL_TEXT_MODEL"] = "1"
                os.environ["AWM_VERL_RUNTIME_PROFILE"] = plan.get("runtime_profile", "pinned")
                from .compat import worker_setup
                worker_setup()
                import ray
                from verl.trainer.main_ppo import run_ppo
                from .runtime import AWMTaskRunner
                # Atomic single-use claim prevents replay of old-policy groups.
                with (args.run / ("probe_started.json" if args.command == "probe" else "started.json")).open("x") as f:
                    json.dump({"pid": os.getpid(), "plan_sha256": digest(plan)}, f)
                OmegaConf.save(config, args.run / "verl-config.yaml")
                try:
                    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(AWMTaskRunner))
                finally:
                    ray.shutdown()
                if args.command == "train" and not read_json(args.run / "summary.json")["target_met"]:
                    raise ValueError("maximum groups reached without target updates; acceptance incomplete")
                return
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.command == "check" and not report["ready"]:
            sys.exit(1)
    except (ValueError, OSError, ImportError) as exc:
        p.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
