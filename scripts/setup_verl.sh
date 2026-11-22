#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
VERL_PYTHON="${VERL_PYTHON:-python3.12}"
"$VERL_PYTHON" - <<'PY'
import shutil, sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit('Use Python 3.12 for this pinned verl environment.')
free = shutil.disk_usage('.').free / 2**30
if free < 15:
    raise SystemExit(f'Only {free:.1f} GiB free; provide at least 15 GiB for the separate CUDA environment and builds.')
PY
"$VERL_PYTHON" -m venv .venv-verl
.venv-verl/bin/python -m pip install --upgrade pip
.venv-verl/bin/python -m pip install -r requirements/verl.in
.venv-verl/bin/python -m pip install ninja packaging psutil setuptools wheel
MAX_JOBS="${MAX_JOBS:-4}" .venv-verl/bin/python -m pip install --no-build-isolation 'flash-attn==2.8.3'
.venv-verl/bin/python -m pip install -e '.[test]'
.venv-verl/bin/python -m pip check
.venv-verl/bin/python - <<'PY'
from harness_awm.training.verl.__main__ import verify_runtime
verify_runtime()
import torch
if not torch.cuda.is_available():
    raise SystemExit('Installed, but CUDA is unavailable; GPU training is not validated.')
print('Pinned verl runtime installed. Run the AWM plan check before training.')
PY
