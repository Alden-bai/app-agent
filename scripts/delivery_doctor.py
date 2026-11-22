"""Target-host A100 checks. No model weights loaded and no Judge requests."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--sft-only", action="store_true", help="hardware and environment checks for offline SFT")
    p.add_argument("--config", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    import torch
    if a.distributed:
        import torch.distributed as dist
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=180))
        try:
            value = torch.tensor(float(rank + 1), device=f"cuda:{rank}")
            dist.all_reduce(value)
            if dist.get_world_size() != 8 or value.item() != 36:
                raise ValueError("eight-rank NCCL all-reduce failed")
            if rank == 0:
                print('{"nccl_8_ranks": "passed"}')
        finally:
            dist.destroy_process_group()
        return
    errors, warnings, cards = [], [], []
    if sys.version_info[:2] != (3, 12):
        errors.append("Python 3.12 required")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        errors.append("Exactly eight visible CUDA devices are required")
    else:
        for i in range(8):
            dev = torch.cuda.get_device_properties(i)
            cards.append({"index": i, "name": dev.name, "memory_gib": dev.total_memory / 2**30})
            if "A100" not in dev.name or dev.total_memory < 75 * 2**30 or (dev.major, dev.minor) != (8, 0):
                errors.append(f"GPU {i} is not an A100 80GB")
        if "sm_80" not in torch.cuda.get_arch_list():
            errors.append("PyTorch wheel lacks sm_80")
    for label, code in [("environment", "import awm, mcp, sqlalchemy")]:
        if not a.config:
            p.error("--config required")
        from harness_awm.contracts import RunConfig
        cfg = RunConfig.model_validate(json.loads(a.config.read_text()))
        try:
            subprocess.run([cfg.runtime_python, "-c", code], check=True, capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as e:
            errors.append(f"{label} interpreter unavailable: {type(e).__name__}")
    isolation = {"status": "not_required_for_offline_sft"}
    if not a.sft_only:
        from harness_awm.training.baseline import isolation_probe
        with tempfile.TemporaryDirectory(prefix="awm-delivery-isolation-", dir=a.config.parent) as folder:
            isolation = isolation_probe(cfg, folder)
        if not isolation["passed"]:
            errors.append(f"{cfg.isolation} isolation check failed")
    memory = {line.split(':')[0]: int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:', 'MemAvailable:'))}
    available = memory["MemAvailable"]
    cgroup = Path('/sys/fs/cgroup')
    if (cgroup / 'memory.max').exists():
        limit = (cgroup / 'memory.max').read_text().strip()
        if limit != 'max':
            available = min(available, max(0, int(limit) - int((cgroup / 'memory.current').read_text())))
    memory['EffectiveAvailable'] = available
    if available < 384 * 2**30:
        errors.append("Need at least 384 GiB available host RAM for conservative FP32/Adam CPU-offload recipe; recommend >=512 GiB physical RAM")
    try:
        driver = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).splitlines()[0]
        if torch.version.cuda and int(torch.version.cuda.split('.')[0]) >= 13 and int(driver.split('.')[0]) < 580:
            errors.append('CUDA 13 GRPO environment requires NVIDIA driver >=580')
    except (OSError, subprocess.SubprocessError, IndexError):
        errors.append('Cannot query NVIDIA driver')
    if shutil.disk_usage('/dev/shm').free < 2 * 2**30:
        warnings.append('Ray object store requests 2 GiB; increase container --shm-size before training')
    disk = shutil.disk_usage(a.config.parent).free
    if disk < 3 * 2**40:
        warnings.append("Available checkpoint storage below 3 TiB. The 2,000-task experiment retains four SFT and four GRPO checkpoints. Check the shared volume quota against actual checkpoint sizes.")
    report = {"ready": not errors, "errors": errors, "warnings": warnings, "gpus": cards,
              "torch": torch.__version__, "cuda": torch.version.cuda, "host_memory_bytes": memory,
              "free_disk_bytes": disk, "isolation": isolation, "judge_requests": 0,
              "gpu_training_validated": False}
    if a.output:
        a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
