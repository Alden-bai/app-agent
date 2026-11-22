"""Run the two-epoch SFT extension with durable logs and completion checks."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness_awm.io import now, read_json, write_json, file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--microbatch-tokens', type=int, default=131072)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / 'launch.json').exists():
            raise RuntimeError('run already launched; inspect state before any restart')
        if read_json(a.source / 'trainer_state.json')['global_step'] != 210:
            raise ValueError('expected original checkpoint-210')
        if shutil.disk_usage(out).free < 55 * 2**30:
            raise RuntimeError('need 55 GiB free for two full checkpoints')
        if os.statvfs(out).f_favail < 500:
            raise RuntimeError('insufficient free inodes on output mount')
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('GPU already in use')
        train = out / 'training'
        command = [sys.executable, '-m', 'harness_awm.training.sft',
            '--model', str(ROOT / 'models/Qwen3.5-2B'), '--resume', str(a.source),
            '--data', str(ROOT / 'data/training_qwen35_2b_v1'), '--output', str(train),
            '--epochs', '4', '--max-steps', '420', '--batch-size', '64',
            '--gradient-accumulation', '1', '--microbatch-tokens', str(a.microbatch_tokens),
            '--learning-rate', '5e-6', '--lr-stage-start-step', '210',
            '--lr-stage-end-step', '420', '--lr-stage-warmup-steps', '7',
            '--save-strategy', 'epoch', '--save-total-limit', '2', '--skip-final-save']
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'runs/sft_kernel_deps'),
            SWANLAB_MODE='online', SWANLAB_EXPERIMENT_NAME='sft-continue-210-to-420-lr5e-6',
            OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', PYTORCH_ALLOC_CONF='expandable_segments:True',
            HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
            PYTHONUNBUFFERED='1')
        write_json(out / 'plan.json', {'created_at': now(), 'source_checkpoint': str(a.source),
            'source_state_sha256': file_hash(a.source / 'trainer_state.json'),
            'source_data_sha256': read_json(ROOT / 'data/training_qwen35_2b_v1/report.json')['train_sha256'],
            'additional_epochs': 2, 'expected_checkpoints': [315, 420], 'command': command,
            'learning_rate': {'peak': 5e-6, 'warmup_steps': 7, 'start_step': 210, 'end_step': 420},
            'source_hashes': {name: file_hash(ROOT / name) for name in (
                'harness_awm/training/sft.py', 'harness_awm/training/continuation.py',
                'harness_awm/training/adaptive.py', 'scripts/run_sft_continuation.py')},
            'evaluation_started': False})
        with (out / 'train.log').open('xb') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            write_json(out / 'launch.json', {'supervisor_pid': os.getpid(), 'training_pid': child.pid,
                                             'started_at': now()})
            peak = 0
            while True:
                code = child.poll()
                try:
                    gpu = subprocess.check_output(['nvidia-smi',
                        '--query-gpu=memory.used,memory.total,utilization.gpu',
                        '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
                    peak = max(peak, int(gpu.split(',')[0]))
                except (subprocess.SubprocessError, ValueError):
                    gpu = None
                status = {'updated_at': now(), 'stage': 'running' if code is None else 'checking',
                    'training_pid': child.pid, 'exit_code': code, 'gpu_mib_total_util': gpu,
                    'peak_observed_gpu_mib': peak, 'free_disk_gib': shutil.disk_usage(out).free / 2**30}
                if (train / 'progress.json').exists():
                    status['progress'] = read_json(train / 'progress.json')
                write_json(out / 'status.json', status)
                with (out / 'gpu.jsonl').open('a') as metrics:
                    metrics.write(json.dumps({'at': now(), 'gpu': gpu}) + '\n')
                if code is not None:
                    break
                time.sleep(10)
        try:
            if code != 0:
                raise RuntimeError(f'training exited with code {code}; inspect train.log')
            for step, epoch in ((315, 3), (420, 4)):
                folder = train / f'checkpoint-{step}'
                state = read_json(folder / 'trainer_state.json')
                if state['global_step'] != step or abs(state['epoch'] - epoch) > .001:
                    raise RuntimeError(f'invalid checkpoint state: {folder}')
                for name in ('model.safetensors', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth',
                             'config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
                    if not (folder / name).is_file() or (folder / name).stat().st_size == 0:
                        raise RuntimeError(f'missing or empty checkpoint artifact: {folder / name}')
            status.update(stage='completed', completed_at=now(),
                          checkpoints=[str(train / f'checkpoint-{step}') for step in (315, 420)])
        except Exception as exc:
            status.update(stage='failed', error=str(exc))
        write_json(out / 'status.json', status)
        if status['stage'] != 'completed':
            raise RuntimeError(status['error'])


if __name__ == '__main__':
    main()
