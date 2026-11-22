#!/usr/bin/env bash
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "$0")/../.." && pwd -P)
cd -- "$source_dir"
env_root=${AWM_ENV_ROOT:-$source_dir}
export AWM_ENV_ROOT=$env_root
export ENV_PYTHON=${ENV_PYTHON:-$env_root/.venv-delivery-env/bin/python}
export SFT_PYTHON=${SFT_PYTHON:-$env_root/.venv-delivery-sft/bin/python}
export GRPO_PYTHON=${GRPO_PYTHON:-$env_root/.venv-delivery-grpo/bin/python}
export AWM_SOURCE_DIR=$source_dir AWM_NIOFLOW=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$source_dir:$source_dir/agent-world-model${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_DIR=${MODEL_DIR:-$source_dir/models/Qwen3.5-27B}
export AWM_RUN_ROOT=${AWM_RUN_ROOT:-$(dirname -- "$source_dir")/awm-runs/nioflow-cu128-grpo2000-v2}
export SFT_WORK=${SFT_WORK:-$AWM_RUN_ROOT/sft} ONLINE_WORK=${ONLINE_WORK:-$AWM_RUN_ROOT/online}
if [[ ! -x "$ENV_PYTHON" ]]; then
    echo "缺少 CPU 解释器 $ENV_PYTHON，请将 AWM_ENV_ROOT 指向三套已安装环境的父目录。" >&2
    exit 2
fi
exec "$ENV_PYTHON" delivery/nioflow/launch.py "$@"
