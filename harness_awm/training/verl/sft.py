"""Prepare/check/train SFT through verl, using the existing encoded AWM dataset."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from ...contracts import ROOT
from ...io import digest, file_hash, read_json, write_json
from ..local import checkpoint_identity
from . import VERL_REVISION
from .sft_compat import verify_sft_runtime
from .sft_data import verified_rows


def generation_metadata(model, text_config):
    path = Path(model) / "generation_config.json"
    if path.exists():
        return read_json(path)
    return {k: text_config[k] for k in ("eos_token_id", "bos_token_id", "pad_token_id") if k in text_config}


def compose_sft_config(plan):
    import verl
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict
    with initialize_config_dir(config_dir=str(Path(verl.__file__).parent / "trainer/config"), version_base=None):
        config = compose(config_name="sft_trainer_engine")
    updates = {
        "model.path": plan["model"], "model.hf_config_path": str(Path(plan["output"]) / "text_config"),
        "model.tokenizer_path": plan["model"], "model.use_remove_padding": False,
        "model.use_fused_kernels": False, "model.enable_gradient_checkpointing": True,
        "model.override_config": {"attn_implementation": "sdpa", "use_cache": False},
        "engine.strategy": "fsdp2", "engine.model_dtype": "fp32", "engine.dtype": "bfloat16",
        "engine.use_torch_compile": False, "engine.seed": plan["seed"],
        "engine.optimizer_offload": plan.get("optimizer_offload", False),
        "engine.param_offload": plan.get("param_offload", False),
        "data.train_batch_size": plan["global_batch_size"], "data.micro_batch_size_per_gpu": 1,
        "data.use_dynamic_bsz": plan.get("max_tokens_per_gpu", 0) > 0,
        "data.max_token_len_per_gpu": plan.get("max_tokens_per_gpu", 0) or plan["max_length"], "data.pad_mode": "no_padding", "data.num_workers": 0,
        "data.max_length": plan["max_length"], "data.val_files": None,
        "optim.lr": plan["learning_rate"], "optim.weight_decay": 0.0,
        "optim.lr_scheduler_type": "cosine", "optim.lr_warmup_steps_ratio": plan["warmup_ratio"],
        "checkpoint.save_contents": ["model", "optimizer", "extra", "hf_model"],
        "trainer.default_local_dir": str(Path(plan["output"]) / "checkpoints"),
        "trainer.project_name": os.environ.get("SWANLAB_PROJECT", "awm-agent"),
        "trainer.experiment_name": Path(plan["output"]).name,
        "trainer.total_epochs": plan["epochs"], "trainer.total_training_steps": plan["steps"],
        "trainer.seed": plan["seed"], "trainer.logger": ["console"],
        "trainer.n_gpus_per_node": plan["gpus"], "trainer.nnodes": 1,
        "trainer.save_freq": "after_each_epoch", "trainer.test_freq": -1,
        "trainer.resume_mode": "disable",
    }
    with open_dict(config):
        for key, value in updates.items():
            OmegaConf.update(config, key, value, merge=False)
        config.awm = {"data": plan["data"], "gpu_memory_limit_gib": plan.get("gpu_memory_limit_gib", 0)}
        mode = os.environ.get("SWANLAB_MODE", "")
        if mode != "disabled" and (mode == "offline" or os.environ.get("SWANLAB_API_KEY")):
            config.trainer.logger = ["console", "swanlab"]
    return config


def prepare(args):
    if (args.gpus < 1 or args.global_batch_size < 1 or args.global_batch_size % args.gpus
            or args.epochs < 1 or args.max_length < 2 or args.max_steps == 0 or args.max_steps < -1
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not 0 <= args.warmup_ratio < 1):
        raise ValueError("invalid SFT settings; global batch must be divisible by gpus")
    model, data, output = args.model.resolve(), args.data.resolve(), args.output.resolve()
    samples, report = verified_rows(data, model, args.max_length)
    budget = getattr(args, "max_tokens_per_gpu", 0)
    memory_limit = getattr(args, "gpu_memory_limit_gib", 0)
    if not math.isfinite(memory_limit) or memory_limit < 0:
        raise ValueError("invalid GPU memory limit")
    if budget and args.gpus != 1:
        raise ValueError("strict SFT token budgeting currently requires one GPU")
    if budget < 0 or (budget and budget < max(len(s["input_ids"]) for s in samples)):
        raise ValueError("token budget must fit the longest sample")
    raw = read_json(model / "config.json")
    if raw.get("model_type") not in {"qwen3_5", "qwen3_5_text"}:
        raise ValueError("SFT recipe supports Qwen3.5 dense text parameters")
    text_config = dict(raw.get("text_config", raw))
    text_config.update(model_type="qwen3_5_text", architectures=["Qwen3_5ForCausalLM"])
    if text_config.get("attention_dropout", 0) != 0:
        raise ValueError("recipe requires attention_dropout=0")
    epoch_steps = math.ceil(len(samples) / args.global_batch_size)
    steps = min(args.max_steps, epoch_steps * args.epochs) if args.max_steps > 0 else epoch_steps * args.epochs
    plan = {"schema_version": "awm-verl-sft-v1", "verl_revision": VERL_REVISION,
        "model": str(model), "data": str(data), "output": str(output),
        "checkpoint": checkpoint_identity(model), "data_sha256": report["train_sha256"],
        "runtime_lock_sha256": file_hash(ROOT / "requirements/verl-sft-runtime.json"),
        "report_sha256": file_hash(data / "report.json"), "text_config": text_config,
        "global_batch_size": args.global_batch_size, "gpus": args.gpus, "epochs": args.epochs,
        "steps": steps, "max_length": args.max_length, "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio, "seed": args.seed, "samples": len(samples),
        "max_tokens_per_gpu": getattr(args, "max_tokens_per_gpu", 0),
        "optimizer_offload": getattr(args, "optimizer_offload", False),
        "param_offload": getattr(args, "param_offload", False),
        "gpu_memory_limit_gib": memory_limit,
        "target_tokens": sum(sum(x != -100 for x in s["labels"][1:]) for s in samples),
        "harness_source_sha256": {str(p.relative_to(ROOT)): file_hash(p)
                                  for p in sorted((ROOT / "harness_awm").rglob("*.py"))}}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    write_json(output / "plan.sha256.json", {"sha256": digest(plan)})
    write_json(output / "text_config/config.json", text_config)
    generation = generation_metadata(model, text_config)
    write_json(output / "text_config/generation_config.json", generation)
    return {"prepared": True, "training_started": False, "samples": len(samples),
            "optimizer_steps": steps, "supervised_tokens": plan["target_tokens"]}


def load_plan(folder):
    folder = Path(folder).resolve()
    plan = read_json(folder / "plan.json")
    if (digest(plan) != read_json(folder / "plan.sha256.json")["sha256"]
            or plan["output"] != str(folder) or plan["verl_revision"] != VERL_REVISION):
        raise ValueError("changed or moved SFT plan; prepare again")
    if plan.get("runtime_lock_sha256") != file_hash(ROOT / "requirements/verl-sft-runtime.json"):
        raise ValueError("SFT runtime lock changed; prepare again")
    if checkpoint_identity(plan["model"]) != plan["checkpoint"]:
        raise ValueError("checkpoint changed")
    if (file_hash(Path(plan["data"]) / "train.jsonl") != plan["data_sha256"]
            or file_hash(Path(plan["data"]) / "report.json") != plan["report_sha256"]):
        raise ValueError("SFT data changed")
    if read_json(folder / "text_config/config.json") != plan["text_config"]:
        raise ValueError("text config changed")
    if read_json(folder / "text_config/generation_config.json") != generation_metadata(plan["model"], plan["text_config"]):
        raise ValueError("generation config changed")
    for relative, expected in plan["harness_source_sha256"].items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError("harness changed; prepare again: " + relative)
    return plan


def check(plan):
    errors = []
    try:
        verify_sft_runtime()
    except (ImportError, ValueError, OSError) as exc:
        errors.append(str(exc))
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() < plan["gpus"]:
            errors.append("requested CUDA devices unavailable in this Python runtime")
    except ImportError as exc:
        errors.append(str(exc))
    return {"ready": not errors, "errors": errors, "training_started": False, "backend": "verl-sft"}


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for name in ("model", "data", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--global-batch-size", type=int, default=32)
    p.add_argument("--max-tokens-per-gpu", type=int, default=0)
    p.add_argument("--gpu-memory-limit-gib", type=float, default=0)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--optimizer-offload", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--param-offload", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--max-length", type=int, default=32768)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--warmup-ratio", type=float, default=.03)
    p.add_argument("--seed", type=int, default=20261106)
    for name in ("check", "train", "worker"):
        sub.add_parser(name).add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            report = prepare(args)
        else:
            plan = load_plan(args.run)
            if args.command == "worker":
                if "LOCAL_RANK" not in os.environ or not (args.run / "started.json").exists():
                    raise ValueError("worker must be launched by the train command")
                from .sft_runtime import run_worker
                run_worker(compose_sft_config(plan))
                return
            report = check(plan)
            if args.command == "train":
                if not report["ready"]:
                    raise ValueError("preflight failed: " + "; ".join(report["errors"]))
                from omegaconf import OmegaConf
                config = compose_sft_config(plan)
                os.environ.setdefault("SWANLAB_LOG_DIR", str(args.run.resolve() / "swanlog"))
                with (args.run / "started.json").open("x") as f:
                    json.dump({"pid": os.getpid(), "plan_sha256": digest(plan)}, f)
                OmegaConf.save(config, args.run / "verl-config.yaml")
                result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                    f"--nproc_per_node={plan['gpus']}", "-m", "--", "harness_awm.training.verl.sft", "worker",
                    "--run", str(args.run.resolve())])
                write_json(args.run / "summary.json", {"backend": "verl-sft", "failed": result.returncode != 0,
                    "exit_code": result.returncode, "planned_steps": plan["steps"], "samples": plan["samples"]})
                sys.exit(result.returncode)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.command == "check" and not report["ready"]:
            sys.exit(1)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
