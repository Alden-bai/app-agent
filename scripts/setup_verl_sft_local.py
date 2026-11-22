"""Create a small SFT venv reusing this host's already installed CUDA wheels."""
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]
paths = [ROOT / relative for relative in (
    "runs/qwen35_base_setup/venv/lib/python3.12/site-packages",
    ".venv-verl-check/lib/python3.12/site-packages",
    ".venv-training/lib/python3.12/site-packages",
    "runs/sft_kernel_deps", "runs/liger_benchmark_deps")]
if sys.version_info[:2] != (3, 12) or not all(p.is_dir() for p in paths):
    raise SystemExit("Requires Python 3.12 and this host's existing CUDA/verl/kernel directories")
environment = ROOT / ".venv-verl-sft"
if not environment.exists():
    venv.create(environment, with_pip=True)
(environment / "lib/python3.12/site-packages/awm-sft-runtime.pth").write_text(
    "\n".join(str(p) for p in [*paths, ROOT]) + "\n")
subprocess.run([str(environment / "bin/python"), "-c",
    "from harness_awm.training.verl.sft_compat import verify_sft_runtime; "
    "verify_sft_runtime(); import torch; assert torch.cuda.is_available(); "
    "print('veRL SFT CUDA runtime verified:', torch.__version__)"], check=True, cwd=ROOT)
