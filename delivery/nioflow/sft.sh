#!/usr/bin/env bash
# Run as the foreground command of one single-node, eight-A100 NIOFlow Job.
set -euo pipefail

source_dir=${AWM_SOURCE_DIR:?}
model=${MODEL_DIR:?}
work=${SFT_WORK:?}
cd -- "$source_dir"
py="${SFT_PYTHON:?}"
delivery="scripts/a100_delivery.py"
if [[ ! -e "$work" ]]; then
    "$ENV_PYTHON" "$delivery" init --sft-only --env-python "$ENV_PYTHON" --work "$work"
fi
if [[ ! -x "$py" || ! -f "$delivery" || ! -f "$work/config.json" || ! -f "$work/sft-only" || ! -r "$model/config.json" ]]; then
    echo "请核对交付目录 $PWD、SFT_WORK=$work 和 MODEL_DIR=$model 的 Job 挂载。" >&2
    exit 2
fi
export PATH="$(dirname -- "$py"):$PATH" PYTHONUNBUFFERED=1
exec 9>>"$work/start-sft.lock"
if ! flock -n 9; then
    echo "已有 Job 在使用 $work，或此挂载不支持文件锁。" >&2
    exit 1
fi
log=$(mktemp "$work/start-sft-XXXXXXXX.log")
echo "Job log: $log"
run_job() {
    echo "Job=${NIOFLOW_JOB_ID:-unknown} host=$(hostname) work=$work model=$model"
    "$py" "$delivery" doctor --work "$work"
    "$py" "$delivery" distributed-check --work "$work"

    # A failed smoke may have already created longest-smoke-data. Keep its
    # output for troubleshooting and use a fresh offline workdir for the retry.
    local smoke="$work/sft-smoke" smoke_work="$work"
    if [[ ! -f "$smoke/summary.json" || ! -f "$smoke/progress.json" ]] &&
       [[ -e "$smoke" || -e "$work/longest-smoke-data" ]]; then
        smoke_work=$(mktemp -d "$work/sft-retry-XXXXXXXX")/work
        "$ENV_PYTHON" "$delivery" init --sft-only --env-python "$ENV_PYTHON" --work "$smoke_work"
        smoke="$smoke_work/sft-smoke"
    fi
    if [[ ! -e "$smoke" ]]; then
        "$py" "$delivery" sft-smoke --work "$smoke_work" --model "$model"
    fi
    check_run "$smoke" 1 1e-5
    local reload="$smoke/reload-$(basename "$log" .log).json"
    "$py" scripts/verify_delivery_checkpoint.py --checkpoint "$smoke/checkpoints/global_step_1" --output "$reload"
    echo "SFT_SMOKE_PASSED checkpoint=$smoke/checkpoints/global_step_1"

    local lr out
    for lr in 1e-5 2e-5; do
        out="$work/sft-lr$lr"
        if [[ ! -e "$out" ]]; then
            "$py" "$delivery" sft --work "$work" --model "$model" --learning-rate "$lr" --epochs 2 --output "$out"
        fi
        check_run "$out" 2 "$lr"
    done
    echo "SFT_COMPLETE outputs=$work/sft-lr1e-5,$work/sft-lr2e-5"
}

check_run() {
    "$py" - "$1" "$model" "$2" "$3" <<'PY'
import json
import sys
from pathlib import Path

folder, model = map(Path, sys.argv[1:3])
epochs, lr = int(sys.argv[3]), float(sys.argv[4])

def read(name):
    return json.loads((folder / name).read_text())

try:
    plan, summary, progress = (read(name) for name in ("plan.json", "summary.json", "progress.json"))
    steps = plan["steps"]
    expected_samples = 8 if epochs == 1 else json.loads(
        Path("data/training_qwen35_27b_v1/report.json").read_text()
    )["supervised_turns"]
    expected_batch = 8 if epochs == 1 else 64
    if epochs == 2 and plan['data_sha256'] != json.loads(Path('data/training_qwen35_27b_v1/report.json').read_text())['train_sha256']:
        raise ValueError('SFT 训练集哈希不匹配')
    steps_per_epoch = (expected_samples + expected_batch - 1) // expected_batch
    if (Path(plan["model"]).resolve() != model.resolve() or plan["epochs"] != epochs
            or plan["learning_rate"] != lr or plan["gpus"] != 8 or plan["samples"] != expected_samples
            or plan["global_batch_size"] != expected_batch or steps != steps_per_epoch * epochs
            or summary["failed"] is not False or summary["exit_code"] != 0
            or summary["planned_steps"] != steps or progress["step"] != steps):
        raise ValueError("训练参数、退出状态或实际完成步数与计划不符")
    for epoch in range(1, epochs + 1):
        ckpt = folder / "checkpoints" / f"global_step_{steps_per_epoch * epoch}"
        for rank in range(8):
            for kind in ("model", "optim", "extra_state"):
                if (ckpt / f"{kind}_world_size_8_rank_{rank}.pt").stat().st_size == 0:
                    raise ValueError(f"checkpoint 分片为空 {ckpt}")
        if not (ckpt / "huggingface/config.json").is_file():
            raise ValueError(f"缺少 HF 导出 {ckpt}")
        print(f"CHECKPOINT {ckpt}", flush=True)
except (OSError, KeyError, ValueError) as exc:
    sys.exit(f"{folder} 未通过训练结果验收，请保留该目录和日志。{exc}")
PY
}

run_job 2>&1 | tee -a "$log"
