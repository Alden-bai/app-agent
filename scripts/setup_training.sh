#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
backend="${1:-cpu}"
case "$backend" in
  cpu) lock=requirements/training-cpu.lock.txt ;;
  cu128) lock=requirements/training-cu128.lock.txt ;;
  cu128-kernels) lock=requirements/training-cu128-kernels.lock.txt; backend=cu128 ;;
  *) echo 'Usage: bash scripts/setup_training.sh cpu|cu128|cu128-kernels [wheelhouse-directory]' >&2; exit 2 ;;
esac
export UV_CACHE_DIR="${UV_CACHE_DIR:-${TMPDIR:-/tmp}/awm-training-uv-cache}"
training_env="${AWM_TRAINING_ENV:-$PWD/.venv-training}"
if [[ -n "${2:-}" ]]; then
  export UV_OFFLINE=1
  wheel_lock="$2/$(basename "$lock")"
  if [[ ! -f "$wheel_lock" ]]; then
    echo "Missing verified wheelhouse lock: $wheel_lock" >&2
    exit 2
  fi
  lock="$wheel_lock"
fi
if [[ ! -x "$training_env/bin/python" ]]; then
  uv venv --python 3.12 "$training_env"
fi
if [[ -n "${2:-}" ]]; then
  uv pip sync --python "$training_env/bin/python" --no-index --find-links "$2" "$lock"
else
  uv pip sync --python "$training_env/bin/python" --torch-backend "$backend" "$lock"
fi
# Use the already-locked runtime dependencies; build backend comes from uv/cache.
uv pip install --python "$training_env/bin/python" --no-deps --no-build-isolation -e . -e ./agent-world-model
uv pip check --python "$training_env/bin/python"
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 "$training_env/bin/python" -m harness_awm.training.doctor
