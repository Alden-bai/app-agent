"""Read-only environment checks; --distributed also tests NCCL and BF16 matmul."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import importlib.util


def supports_capability(capability, architectures):
    # CUDA cubins can run on a newer minor revision of the same major family
    # (e.g. an sm_86 wheel supports the 4090's sm_89). PTX can be JIT-compiled.
    major, minor = capability
    for arch in architectures:
        kind, _, code = arch.partition("_")
        if not code.isdigit():
            continue
        target = (int(code[:-1]), int(code[-1]))
        if kind == "sm" and target[0] == major and target[1] <= minor:
            return True
        if kind == "compute" and target <= capability:
            return True
    return False


def check(require_cuda=False, model=None, distributed=False):
    import torch
    import transformers
    from trl import SFTTrainer, GRPOTrainer
    from accelerate.commands.config.config_args import load_config_from_file
    from .data import sha
    problems = []
    result = {"python": platform.python_version(), "versions": {
        p: importlib.metadata.version(p) for p in ("torch", "transformers", "trl", "accelerate", "datasets")},
        "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
        "cuda_devices": [], "bwrap_installed": bool(shutil.which("bwrap")),
        "linear_attention_kernel_installed": importlib.util.find_spec("fla") is not None,
        "live_grpo_ready": False,
        "grpo_implementation": "harness_awm.training.grpo: turn-wise, population std, single process",
        "live_grpo_blockers": ["Run task-specific GRPO preflight with calibrated reward admissions",
                               "Real environment/Judge and target GPU acceptance are not established by doctor",
                               "Multi-process GRPO is not supported by the turn-wise engine"],
    }
    if platform.python_version_tuple()[:2] != ('3', '12'):
        problems.append("reproducible environment expects Python 3.12")
    for path in Path("configs/training").glob("fsdp2-*.yaml"):
        config = load_config_from_file(str(path))
        if config.distributed_type != 'FSDP' or config.fsdp_config.get('fsdp_version') != 2:
            problems.append(f"invalid FSDP configuration: {path}")
    if require_cuda and not torch.cuda.is_available():
        problems.append("CUDA required but unavailable")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(i)
            result["cuda_devices"].append({"name": torch.cuda.get_device_name(i),
                "memory_gib": round(torch.cuda.get_device_properties(i).total_memory/2**30, 1), "capability": cap})
            if not supports_capability(cap, torch.cuda.get_arch_list()):
                problems.append(f"GPU {i} architecture missing from PyTorch wheel")
        result["compiled_architectures"] = torch.cuda.get_arch_list()
    if model:
        model = Path(model)
        lock = json.loads(Path("configs/training/qwen3.5-2b-model.lock.json").read_text())
        missing = [r["name"] for r in lock["files"] if not (model/r["name"]).is_file()]
        result["missing_model_assets"] = missing
        problems += [f"missing model asset: {n}" for n in missing]
        for row in lock["files"]:
            if row["name"] not in missing and sha(model/row["name"]) != row["sha256"]:
                problems.append(f"model checksum mismatch: {row['name']}")
        result["model_assets_verified"] = not missing and not any('checksum' in p for p in problems)
    if distributed:
        if not torch.cuda.is_available():
            problems.append("NCCL test requires GPUs")
        else:
            import torch.distributed as dist
            from datetime import timedelta
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            dist.init_process_group("nccl", timeout=timedelta(seconds=60))
            try:
                x = torch.ones(1, device="cuda")
                dist.all_reduce(x)
                a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                (a @ a).float().square().mean().backward()
                torch.cuda.synchronize()
                if x.item() != dist.get_world_size() or not torch.isfinite(a.grad).all():
                    problems.append("NCCL/BF16 computation failed")
                result["distributed_world_size"] = dist.get_world_size()
            finally:
                dist.destroy_process_group()
    result["errors"] = problems
    result["passed"] = not problems
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--model", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    result = check(a.require_cuda, a.model, a.distributed)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
