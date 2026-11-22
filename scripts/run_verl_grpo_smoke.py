"""Run the bounded local GRPO acceptance pipeline in a fresh directory."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys

from harness_awm.contracts import ROOT
from harness_awm.io import now, read_json, write_json
from harness_awm.training.local import checkpoint_identity


def existing_parent(path):
    while not path.exists():
        path = path.parent
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint-dir', type=Path, required=True)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.78,
                   help='vLLM fraction; 0.78 targets about 85 GiB total on this 96 GB host')
    p.add_argument('--selection', type=Path, default=ROOT / 'runs/sft_best_test_20261110_v1/selection.json')
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (ROOT / 'runs/verl_grpo_smoke.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selection = read_json(args.selection)
        model = selection['checkpoint']['path']
        if checkpoint_identity(model) != selection['checkpoint']:
            raise ValueError('selected SFT identity changed')
        for path, minimum in [(output, 2 * 1024**3), (args.checkpoint_dir, 40 * 1024**3)]:
            parent = existing_parent(path)
            if shutil.disk_usage(parent).free < minimum or os.statvfs(parent).f_favail < 2000:
                raise ValueError('insufficient storage or inodes: ' + str(path))
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
        def run(stage, argv):
            write_json(output / 'status.json', {'stage': stage, 'status': 'running', 'at': now(), 'pid': os.getpid()})
            with (output / (stage + '.log')).open('x') as log:
                subprocess.run([sys.executable, *map(str, argv)], cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
        def prepare(name, config):
            argv = ['-m', 'harness_awm.training.verl', 'prepare', '--config', config,
                '--tasks', ROOT / 'data/training_runtime_20261106_v1', '--model', model,
                '--output', output / name, '--groups', '24', '--group-size', '4', '--concurrency', '2',
                '--gpus', '1', '--runtime-profile', 'local-sdpa', '--target-updates', '3',
                '--gpu-memory-utilization', str(args.gpu_memory_utilization),
                '--checkpoint-dir', args.checkpoint_dir, '--save-every', '24']
            for task in ['e_commerce_marketplace_10:0', 'fleet_management_6:1', 'membership_management_2:0']:
                argv.extend(['--task', task])
            run('prepare-' + name, argv)
        try:
            prepare('probe', ROOT / 'configs/training/grpo-landlock-local.json')
            run('gpu-probe', ['-m', 'harness_awm.training.verl', 'probe', '--run', output / 'probe'])
            run('calibration', [ROOT / 'scripts/calibrate_verl_grpo.py', '--config',
                ROOT / 'configs/training/grpo-landlock-local.json', '--tasks', ROOT / 'data/training_runtime_20261106_v1',
                '--output', output / 'calibration'])
            prepare('training', output / 'calibration/config.json')
            run('check', ['-m', 'harness_awm.training.verl', 'check', '--run', output / 'training'])
            try:
                run('train', ['-m', 'harness_awm.training.verl', 'train', '--run', output / 'training'])
            except subprocess.CalledProcessError:
                summary_path = output / 'training/summary.json'
                if summary_path.exists():
                    summary = read_json(summary_path)
                    if not summary['failed'] and not summary['target_met']:
                        run('reload-incomplete', [ROOT / 'scripts/verify_verl_grpo_checkpoint.py',
                            '--run', output / 'training', '--inspect-incomplete'])
                        raise ValueError('sampling limit reached without target updates; checkpoint inspected, acceptance incomplete')
                raise
            run('reload', [ROOT / 'scripts/verify_verl_grpo_checkpoint.py', '--run', output / 'training'])
            write_json(output / 'status.json', {'status': 'complete', 'stage': 'accepted', 'at': now(),
                                              'acceptance': str(output / 'training/acceptance.json')})
        except BaseException as exc:
            write_json(output / 'failure.json', {'status': 'failed', 'at': now(), 'error': str(exc),
                                               'last_status': read_json(output / 'status.json')})
            raise


if __name__ == '__main__':
    main()
