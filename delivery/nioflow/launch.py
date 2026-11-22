"""Foreground entry for the receiver's NIOFlow allocation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def run(*args, env=None):
    subprocess.run(list(map(str, args)), check=True, cwd=ROOT, env=env)


def load_judge():
    from dotenv import dotenv_values
    default = ROOT / '.env'
    if not default.is_file():
        default = Path(os.environ['AWM_ENV_ROOT']) / '.env'
    envfile = Path(os.environ.get('AWM_JUDGE_ENV_FILE', default))
    if 'AWM_JUDGE_ENV_FILE' in os.environ and not envfile.is_file():
        raise FileNotFoundError(envfile)
    if envfile.is_file():
        values = dotenv_values(envfile)
        key_name = os.environ.get('AWM_JUDGE_KEY_ENV') or values.get('AWM_JUDGE_KEY_ENV') or 'OPENAI_API_KEY'
        for name in ['AWM_JUDGE_BASE_URL', 'AWM_JUDGE_MODEL', 'AWM_JUDGE_KEY_ENV', 'OPENAI_BASE_URL', 'OPENAI_MODEL', key_name]:
            if values.get(name):
                os.environ.setdefault(name, values[name])
    for target, variable in [('AWM_JUDGE_BASE_URL', 'OPENAI_BASE_URL'), ('AWM_JUDGE_MODEL', 'OPENAI_MODEL')]:
        if os.environ.get(variable):
            os.environ.setdefault(target, os.environ[variable])


def rebuild(work, env_py, sft_py):
    from harness_awm.io import file_hash
    split = ROOT / 'data/training_split_20261106_v1'
    manifest = json.loads((split / 'manifest.json').read_text())
    expected = json.loads((ROOT / 'data/training_qwen35_27b_v1/report.json').read_text())['train_sha256']
    out = work / 'data-rebuild'

    def compare(folder):
        for name, sha in manifest['files'].items():
            if name.endswith('.jsonl') and file_hash(folder / 'split' / name) != sha:
                raise ValueError('data replay mismatch: ' + name)
        if file_hash(folder / 'encoded/train.jsonl') != expected:
            raise ValueError('SFT encoded replay mismatch')

    if out.exists():
        compare(out)
    else:
        with tempfile.TemporaryDirectory(dir=work, prefix='rebuild-') as temporary:
            pending = Path(temporary) / 'result'
            run(env_py, ROOT / 'scripts/prepare_training_split.py', '--root', ROOT, '--replay-frozen', '--output', pending / 'split')
            run(sft_py, '-m', 'harness_awm.training.data', '--split', pending / 'split', '--root', ROOT,
                '--tokenizer', ROOT / 'assets/qwen35-27b', '--output', pending / 'encoded')
            compare(pending)
            pending.rename(out)
    print('DATA_REPLAY_PASSED', expected, out, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['all', 'cpu-check', 'rebuild-data', 'check', 'sft', 'online'], default='all', nargs='?')
    args = p.parse_args()
    if sys.version_info[:2] != (3, 12):
        p.error('CPU 环境需要 Python 3.12，请检查 ENV_PYTHON')
    work = Path(os.environ['AWM_RUN_ROOT']).resolve()
    if work.is_relative_to(ROOT) or ROOT.is_relative_to(work):
        p.error('AWM_RUN_ROOT 需要使用独立的实验目录')
    work.mkdir(parents=True, exist_ok=True)
    env_py = Path(os.environ['ENV_PYTHON']).absolute()
    sft_py = Path(os.environ['SFT_PYTHON']).absolute()
    grpo_py = Path(os.environ['GRPO_PYTHON']).absolute()
    with (work / '.launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print('JOB', os.environ.get('NIOFLOW_JOB_ID', 'local'), 'WORK', work, flush=True)
        if (ROOT / 'DELIVERY_MANIFEST.json').is_file():
            run(env_py, ROOT / 'scripts/verify_training_delivery.py', ROOT)
        run(env_py, ROOT / 'scripts/check_delivery_data.py', '--root', ROOT, '--output', work / 'data-audit.json')
        if args.stage == 'cpu-check':
            with tempfile.TemporaryDirectory(dir=work) as temp:
                cfg = Path(temp) / 'worker'
                run(env_py, ROOT / 'scripts/a100_delivery.py', 'init', '--work', cfg, '--env-python', env_py)
                run(env_py, ROOT / 'scripts/nioflow_worker_smoke.py', '--config', cfg / 'config.json',
                    '--output', work / 'worker-smoke.json')
                run(env_py, ROOT / 'scripts/prepare_delivery_skills.py', '--output', Path(temp) / 'skills.json')
            print('CPU_CHECK_PASSED', work, flush=True)
            return
        if args.stage in {'all', 'rebuild-data'}:
            rebuild(work, env_py, sft_py)
            if args.stage == 'rebuild-data':
                return
        sys.path.insert(0, str(ROOT / 'scripts'))
        from a100_delivery import validate_model_assets
        model = Path(os.environ['MODEL_DIR']).resolve()
        validate_model_assets(model)
        index = model / 'model.safetensors.index.json'
        shards = set(json.loads(index.read_text())['weight_map'].values()) if index.is_file() else {'model.safetensors'}
        if not shards or any(Path(shard).name != shard or not (model / shard).is_file() or
                             not (model / shard).stat().st_size for shard in shards):
            raise ValueError(f'基模权重分片缺失 {model}')
        if args.stage != 'sft':
            load_judge()
        if args.stage in {'all', 'sft', 'check'}:
            run(sft_py, '-c', "import torch; from harness_awm.training.verl.sft_compat import verify_sft_runtime; "
                "verify_sft_runtime(); assert torch.version.cuda=='12.8', torch.version.cuda")
        if args.stage == 'check':
            with tempfile.TemporaryDirectory(dir=work) as temp:
                offline = Path(temp) / 'sft'
                entry = ROOT / 'scripts/a100_delivery.py'
                run(env_py, entry, 'init', '--sft-only', '--env-python', env_py, '--work', offline)
                run(sft_py, entry, 'doctor', '--work', offline)
                run(sft_py, entry, 'distributed-check', '--work', offline)
        if args.stage in {'all', 'check'}:
            try:
                run(env_py, HERE / 'online.py', '--preflight-only')
            except subprocess.CalledProcessError as exc:
                if args.stage == 'check':
                    raise
                print('ONLINE_BLOCKED', exc.returncode, '继续执行独立的 SFT 阶段', flush=True)
                run('bash', HERE / 'sft.sh')
                raise SystemExit(exc.returncode)
            if args.stage == 'check':
                return
        if args.stage in {'all', 'sft'}:
            run('bash', HERE / 'sft.sh')
        if args.stage in {'all', 'online'}:
            run(env_py, HERE / 'online.py')


if __name__ == '__main__':
    main()
