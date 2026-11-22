"""Calibrate and launch monitored GRPO training with bounded updates and GPU guards."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from harness_awm.contracts import ROOT
from harness_awm.io import read_json, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--scratch', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--preset', choices=('single-gpu-800',))
    p.add_argument('--train-timeout', type=int, default=7200)
    p.add_argument('--kernel-cache', type=Path, default=Path(sys.prefix).parent / 'kernel-cache')
    p.add_argument('--groups', type=int, default=8)
    p.add_argument('--group-size', type=int, default=4)
    p.add_argument('--target-updates', type=int, default=1)
    p.add_argument('--save-every', type=int, default=1)
    p.add_argument('--reuse-calibration', type=Path)
    p.add_argument('--require-reused-calibration', action='store_true')
    p.add_argument('--reuse-probe', type=Path)
    p.add_argument('--gpu-memory-utilization', type=float, default=.60)
    p.add_argument('--allocator-gb', type=float, default=72)
    p.add_argument('--gpu-stop-gb', type=float, default=78)
    args = p.parse_args()
    if not (0 < args.allocator_gb < args.gpu_stop_gb < 1000 and 0 < args.gpu_memory_utilization < 1):
        p.error('invalid GPU memory limits')
    if args.train_timeout <= 0:
        p.error('train-timeout must be positive')
    if not (0 < args.target_updates <= args.groups and args.save_every > 0):
        p.error('require 0 < target-updates <= groups and save-every > 0')
    run, scratch = args.run.resolve(), args.scratch.resolve()
    run.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    kernel_cache = args.kernel_cache.resolve()
    (kernel_cache / 'tmp').mkdir(parents=True, exist_ok=True)
    state = {'status': 'running', 'pid': os.getpid(), 'target_updates': args.target_updates, 'max_groups': args.groups}
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1',
               TOKENIZERS_PARALLELISM='false', SWANLAB_MODE='disabled',
               TMPDIR=str(kernel_cache / 'tmp'),
               TRITON_CACHE_DIR=str(kernel_cache / 'triton'),
               TORCHINDUCTOR_CACHE_DIR=str(kernel_cache / 'torchinductor'),
               VLLM_CACHE_ROOT=str(kernel_cache / 'vllm'))
    def status(**fields):
        state.update(fields, updated_at=time.time())
        write_json(run / 'status.json', state)
    def command(name, argv, gpu=False, timeout=7200):
        status(stage=name)
        with (run / (name + '.log')).open('x') as log:
            proc = subprocess.Popen([sys.executable, *map(str, argv)], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status(child_pid=proc.pid)
            started, peak, last_status = time.monotonic(), 0, 0
            try:
                with (run / 'gpu.jsonl').open('a') as records:
                    while proc.poll() is None:
                        if time.monotonic() - started > timeout:
                            raise RuntimeError('pilot_stage_time_limit:' + name)
                        if gpu:
                            raw = subprocess.check_output(['nvidia-smi', '--id=0',
                                '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
                            used, util = map(int, raw.strip().split(','))
                            peak = max(peak, used)
                            row = {'at': time.time(), 'stage': name, 'used_mib': used,
                                   'used_gb': used * 1024**2 / 10**9, 'utilization': util}
                            records.write(json.dumps(row) + '\n'); records.flush()
                            if row['used_gb'] >= args.gpu_stop_gb:
                                raise RuntimeError(f'pilot_gpu_stop_{args.gpu_stop_gb}GB:' + str(row))
                            if time.monotonic() - last_status >= 10:
                                status(gpu=row, peak_gb=peak * 1024**2 / 10**9)
                                last_status = time.monotonic()
                        time.sleep(1)
                if proc.returncode:
                    raise RuntimeError(f'{name} exited {proc.returncode}; inspect {name}.log')
            except BaseException:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                raise
    with (ROOT / 'runs/verl_grpo_smoke.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            from harness_awm.training.verl.pilot import resource_check
            resource_check(run, free_gib=75)
            config = read_json(run / 'config.json')
            controls = read_json(run / 'controls.json')
            accepted, admissions = [], {}
            def calibrate(task):
                name = 'calibration-' + task.replace(':', '-')
                folder = scratch / name
                if folder != run / name:
                    (run / name).symlink_to(folder, target_is_directory=True)
                if args.reuse_calibration:
                    from harness_awm.contracts import RunConfig
                    from harness_awm.training.tasks import load_tasks, select_tasks
                    from harness_awm.training.grpo import preflight
                    old = args.reuse_calibration.resolve()
                    old_config = RunConfig.model_validate({**config, 'reward_admissions': str(old / 'reward_admissions.json')})
                    task_specs, _, sources = load_tasks(ROOT / 'data/training_runtime_20261106_v1', 'grpo_train')
                    if sources:
                        old_config = old_config.model_copy(update={'judge_inputs_manifest': sources})
                    selected = select_tasks(task_specs, [task])
                    try:
                        preflight(selected, old_config, args.group_size)
                    except (ValueError, OSError, KeyError) as exc:
                        status(calibration_reuse_failure={task: str(exc)})
                        if args.require_reused_calibration:
                            raise RuntimeError('required_calibration_reuse_failed:' + task) from exc
                    else:
                        prior = read_json(old / 'reward_admissions.json')['tasks']
                        write_json(run / (name + '-reuse.json'), {'source': str(old), 'task': task, 'verified': True})
                        return task, {t.task_instance_id: prior[t.task_instance_id] for t in selected}
                try:
                    command(name, [ROOT / 'scripts/calibrate_verl_grpo.py', '--config', run / 'config.json',
                        '--tasks', ROOT / 'data/training_runtime_20261106_v1', '--controls', run / 'controls.json',
                        '--task', task, '--output', folder], timeout=3600)
                except RuntimeError:
                    if not (folder / 'summary.json').exists():
                        raise
                summary = read_json(folder / 'summary.json')
                write_json(run / (name + '-summary.json'), summary)
                from harness_awm.training.verl.pilot import archive_group
                archive_group(folder, run / 'archives' / (name + '.tar.gz'))
                if summary.get('passed'):
                    return task, read_json(folder / 'reward_admissions.json')['tasks']
                return task, {}
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=3) as pool:
                for future in as_completed([pool.submit(calibrate, task) for task in controls]):
                    task, entries = future.result()
                    if entries:
                        accepted.append(task)
                        admissions.update(entries)
                    status(admitted_tasks=accepted)
            accepted.sort(key=lambda task: list(controls).index(task))
            if len(accepted) < 2:
                raise RuntimeError('pilot_requires_at_least_two_freshly_admitted_tasks')
            write_json(run / 'reward_admissions.json', {'schema_version': 'awm-reward-admissions-v2', 'tasks': admissions})
            config['reward_admissions'] = str(run / 'reward_admissions.json')
            write_json(run / 'training-config.json', config)
            train = scratch / 'training'
            (run / 'training').symlink_to(train, target_is_directory=True)
            argv = ['-m', 'harness_awm.training.verl', 'prepare', '--config', run / 'training-config.json',
                '--tasks', ROOT / 'data/training_runtime_20261106_v1', '--model', args.model,
                '--output', train, '--groups', args.groups, '--target-updates', args.target_updates, '--group-size', args.group_size,
                '--concurrency', 4, '--learning-rate', '1e-6', '--gpu-memory-utilization', args.gpu_memory_utilization,
                '--allocator-gb', args.allocator_gb, '--gpu-stop-gb', args.gpu_stop_gb,
                '--runtime-profile', 'local-sdpa', '--rollout-correction', 'token-is-v1',
                '--checkpoint-dir', run / 'checkpoints', '--save-every', args.save_every, '--pilot-guards', '--stress-probe']
            if args.preset:
                argv += ['--preset', args.preset]
            for task in accepted:
                argv += ['--task', task]
            command('prepare', argv)
            command('check', ['-m', 'harness_awm.training.verl', 'check', '--run', train])
            if args.reuse_probe:
                from harness_awm.training.verl.__main__ import load_plan
                from harness_awm.io import file_hash
                prior, current = load_plan(args.reuse_probe), load_plan(train)
                keys = ('checkpoint', 'model', 'gpus', 'group_size', 'concurrency', 'learning_rate',
                        'gpu_memory_utilization', 'runtime_profile', 'rollout_correction',
                        'allocator_gb', 'gpu_stop_gb', 'pilot_guards', 'stress_probe', 'harness_source_sha256')
                if any(prior.get(key) != current.get(key) for key in keys):
                    raise ValueError('prior GPU probe configuration differs')
                ignored = {'reward_admissions', 'judge_rate_db'}
                if ({k: v for k, v in prior['run_config'].items() if k not in ignored}
                        != {k: v for k, v in current['run_config'].items() if k not in ignored}):
                    raise ValueError('prior GPU probe runtime configuration differs')
                proof = args.reuse_probe / 'gpu_probe/report.json'
                report = read_json(proof)
                if not (report.get('passed') and report.get('weight_sync')
                        and report.get('optimizer_steps') == 1 and report.get('judge_requests') == 0
                        and report.get('sequence_tokens', 0) >= current['run_config']['context_limit'] - 32):
                    raise ValueError('prior GPU stress proof is incomplete')
                write_json(run / 'stress-probe-reuse.json', {'source': str(proof.resolve()),
                    'sha256': file_hash(proof), 'configuration_matched': True, 'report': report})
            else:
                command('stress-probe', ['-m', 'harness_awm.training.verl', 'probe', '--run', train], gpu=True)
            command('train', ['-m', 'harness_awm.training.verl', 'train', '--run', train], gpu=True, timeout=args.train_timeout)
            command('verify', [ROOT / 'scripts/verify_verl_grpo_checkpoint.py', '--run', train], gpu=True)
            status(status='complete', stage='complete', acceptance=str(train / 'acceptance.json'),
                   next_gate='review_training_and_checkpoint_acceptance')
        except BaseException as exc:
            status(status='needs_attention', error=str(exc))
            raise
        finally:
            train = scratch / 'training'
            if train.exists():
                # collect_group writes group.json only after every candidate has finished.
                # Preserve complete failed groups too (API guards can stop before normal archival).
                from harness_awm.training.verl.pilot import archive_group
                for group in sorted((train / 'groups').glob('group-*')):
                    archive = run / 'archives' / (group.name + '.tar.gz')
                    if (group / 'group.json').exists() and not archive.exists():
                        archive_group(group, archive)
                for path in train.glob('*.json'):
                    write_json(run / 'records' / path.name, read_json(path))


if __name__ == '__main__':
    main()
