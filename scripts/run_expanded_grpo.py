"""Calibrate a frozen multi-step train pool, then launch a monitored GRPO pilot."""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from harness_awm.contracts import ROOT
from harness_awm.io import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--controls', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--groups', type=int, default=48)
    parser.add_argument('--group-size', type=int, default=4)
    parser.add_argument('--group-offset', type=int, default=0)
    parser.add_argument('--target-updates', type=int, default=8)
    parser.add_argument('--save-every', type=int, default=4)
    parser.add_argument('--calibration-workers', type=int, default=3)
    parser.add_argument('--min-tasks', type=int, default=4)
    parser.add_argument('--resume-calibration', action='store_true')
    parser.add_argument('--training-name', default='training')
    parser.add_argument('--rollout-correction', choices=('strict', 'token-is-v1'), default='strict')
    args = parser.parse_args()
    run = args.run.resolve()
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    state = {'pid': os.getpid(), 'stage': 'calibration', 'status': 'running',
             'requested_tasks': list(read_json(args.controls)), 'calibrations': {}}
    def status(**kw):
        state.update(kw, updated_at=time.time())
        write_json(run / 'status.json', state)
    def command(name, argv):
        with (run / (name + '.log')).open('x') as log:
            result = subprocess.run([sys.executable, *map(str, argv)], cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        return result.returncode
    with (ROOT / 'runs/verl_grpo_smoke.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            status()
            def calibrate(task):
                override_path = run / 'calibration_overrides.json'
                override = read_json(override_path).get(task, {}) if override_path.exists() else {}
                folder = Path(override.get('folder', run / 'calibration' / task.replace(':', '-')))
                control_path = Path(override.get('controls', args.controls)).resolve()
                if folder.exists() and args.resume_calibration:
                    from harness_awm.contracts import RunConfig
                    from harness_awm.io import digest
                    from harness_awm.judge_runtime import config_hash
                    expected = read_json(folder / 'expectations.json')
                    if (expected['judge_config_hash'] != config_hash(RunConfig.model_validate(read_json(run / 'config.json')))
                            or digest(expected['specs']) != digest(read_json(control_path))
                            or [t['scenario'] + ':' + t['native_task_id'] for t in expected['tasks']] != [task]):
                        raise ValueError('Cannot resume calibration with changed task, Judge or controls')
                    while not (folder / 'summary.json').exists():
                        alive = False
                        for procfile in Path('/proc').glob('[0-9]*/cmdline'):
                            try:
                                raw = procfile.read_bytes()
                            except (OSError, ProcessLookupError):
                                continue
                            if b'calibrate_verl_grpo.py' in raw and str(folder).encode() in raw:
                                alive = True
                                break
                        if not alive:
                            return task, folder, 1
                        time.sleep(5)
                    return task, folder, 0 if read_json(folder / 'summary.json')['passed'] else 1
                rc = command(('calibration-override-' if override else 'calibration-') + task.replace(':', '-'), [ROOT / 'scripts/calibrate_verl_grpo.py',
                    '--config', run / 'config.json', '--tasks', ROOT / 'data/training_runtime_20261106_v1',
                    '--controls', control_path, '--task', task, '--output', folder])
                return task, folder, rc
            entries, accepted = {}, []
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.calibration_workers) as pool:
                futures = [pool.submit(calibrate, task) for task in state['requested_tasks']]
                for future in concurrent.futures.as_completed(futures):
                    task, folder, rc = future.result()
                    summary = read_json(folder / 'summary.json') if (folder / 'summary.json').exists() else {}
                    state['calibrations'][task] = {'exit_code': rc, 'passed': summary.get('passed', False),
                                                  'budget': summary.get('budget'), 'path': str(folder)}
                    if rc == 0 and summary.get('passed'):
                        entries.update(read_json(folder / 'reward_admissions.json')['tasks'])
                        accepted.append(task)
                    status(admitted=len(entries))
            # Deterministic order, prioritizing longer dependency chains before simple updates.
            accepted = [key for key in state['requested_tasks'] if key in accepted]
            if len(accepted) < args.min_tasks:
                raise RuntimeError(f'Only {len(accepted)} tasks passed calibration; require {args.min_tasks}; inspect retained results')
            write_json(run / 'reward_admissions.json', {'schema_version': 'awm-reward-admissions-v2', 'tasks': entries})
            config = read_json(run / 'config.json')
            config['reward_admissions'] = str(run / 'reward_admissions.json')
            write_json(run / 'training-config.json', config)
            status(stage='waiting_for_gpu', training_tasks=accepted)
            while True:
                used = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
                if used < 1024:
                    break
                time.sleep(10)
            train = run / args.training_name
            argv = ['-m', 'harness_awm.training.verl', 'prepare', '--config', run / 'training-config.json',
                '--tasks', ROOT / 'data/training_runtime_20261106_v1', '--model', args.model.resolve(), '--output', train,
                '--groups', args.groups, '--group-offset', args.group_offset, '--group-size', args.group_size, '--concurrency', 4, '--gpus', 1,
                '--runtime-profile', 'local-sdpa', '--target-updates', args.target_updates,
                '--gpu-memory-utilization', .78, '--checkpoint-dir', args.checkpoint_dir.resolve(), '--save-every', args.save_every,
                '--rollout-correction', args.rollout_correction]
            for task in accepted:
                argv += ['--task', task]
            status(stage='prepare')
            if command(args.training_name + '-prepare', argv):
                raise RuntimeError('training preparation failed')
            if command(args.training_name + '-check', ['-m', 'harness_awm.training.verl', 'check', '--run', train]):
                raise RuntimeError('training preflight failed')
            status(stage='training')
            with (run / (args.training_name + '.log')).open('x') as log, (run / 'gpu.jsonl').open('a') as gpu:
                proc = subprocess.Popen([sys.executable, '-u', '-m', 'harness_awm.training.verl', 'train', '--run', str(train)],
                                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                status(training_pid=proc.pid)
                while proc.poll() is None:
                    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                                   '--format=csv,noheader,nounits'], text=True).strip()
                    used, utilization = map(int, raw.split(','))
                    record = {'at': time.time(), 'used_mib': used, 'used_gib': used/1024, 'utilization': utilization}
                    gpu.write(json.dumps(record)+'\n'); gpu.flush()
                    status(gpu=record, progress=read_json(train/'progress.json') if (train/'progress.json').exists() else None)
                    time.sleep(5)
                rc = proc.returncode
            status(training_exit_code=rc)
            if rc:
                raise RuntimeError('training exited nonzero; inspect training.log and summary.json')
            status(stage='checkpoint_verification')
            if command(args.training_name + '-verify', [ROOT/'scripts/verify_verl_grpo_checkpoint.py', '--run', train]):
                raise RuntimeError('checkpoint verification failed')
            status(stage='complete', status='complete', acceptance=str(train/'acceptance.json'))
        except BaseException as exc:
            status(status='needs_attention', error=str(exc))
            raise


if __name__ == '__main__':
    main()
