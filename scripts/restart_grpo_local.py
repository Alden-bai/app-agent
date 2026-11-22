"""Restart a bounded GRPO pilot using verified local evidence and local runtime I/O."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from harness_awm.contracts import ROOT, RunConfig
from harness_awm.io import file_hash, read_json, verify_seal, write_json


def stage_admissions(source, destination):
    """Relocate proofs while keeping sealed evidence bytes and original attestations."""
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = read_json(source)
    provenance = []
    for index, (task_id, entry) in enumerate(manifest['tasks'].items()):
        original = Path(entry['calibration_ref'])
        if file_hash(original) != entry['calibration_sha256']:
            raise ValueError('source calibration hash mismatch: ' + task_id)
        proof = read_json(original)
        relocated = {}
        for control in proof['controls'] + proof.get('trajectory_controls', []):
            folder = Path(control['path']).resolve()
            if str(folder) not in relocated:
                target = destination / f'task-{index:03d}' / f'control-{len(relocated):02d}'
                hashes = read_json(folder / 'evidence_seal.json')
                for relative, expected in hashes.items():
                    src = (folder / relative).resolve()
                    dst = target / relative
                    if not src.is_relative_to(folder) or not dst.resolve().is_relative_to(target.resolve()):
                        raise ValueError('unsafe evidence path')
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
                    if file_hash(dst) != expected:
                        raise ValueError('source evidence hash mismatch: ' + relative)
                shutil.copyfile(folder / 'evidence_seal.json', target / 'evidence_seal.json')
                verify_seal(target)
                relocated[str(folder)] = str(target.resolve())
            control['path'] = relocated[str(folder)]
        target_proof = destination / f'task-{index:03d}' / 'calibration.json'
        write_json(target_proof, proof)
        provenance.append({'task': task_id, 'original': str(original),
            'original_sha256': entry['calibration_sha256'], 'relocation': relocated,
            'local': str(target_proof.resolve()), 'local_sha256': file_hash(target_proof)})
        entry.update(calibration_ref=str(target_proof.resolve()), calibration_sha256=file_hash(target_proof))
    write_json(destination / 'reward_admissions.json', manifest)
    write_json(destination / 'relocation.json', {'source_manifest': str(source.resolve()),
        'source_sha256': file_hash(source), 'sealed_bytes_unchanged': True, 'tasks': provenance})
    return destination / 'reward_admissions.json'


async def environment_probe(config, tasks, controls, output):
    from harness_awm.backend import Budget, ScriptedBackend
    from harness_awm.runner import EpisodeRunner
    import importlib.util
    spec = importlib.util.spec_from_file_location('awm_calibration_controls', ROOT / 'scripts/calibrate_verl_grpo.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actions, state = module.actions, module.state
    output.mkdir(parents=True, exist_ok=False)
    scripted = config.model_copy(update={'actor': 'scripted'})
    budget = Budget(output / 'budget.sqlite', config.max_physical_requests, config.max_total_tokens)
    runner = EpisodeRunner(scripted, budget, asyncio.Semaphore(4))
    lags = []
    finished = False
    async def heartbeat():
        while not finished:
            before = time.monotonic()
            await asyncio.sleep(.05)
            lags.append(max(0, time.monotonic() - before - .05))
    async def candidate(index):
        # Exercise the exact scenario that previously lost 7/8 rewards.
        task = tasks[0]
        spec = controls[f'{task.scenario}:{task.native_task_id}']
        folder = output / f'candidate-{index:03d}'
        start = time.monotonic()
        result = await runner.run(task, folder, f'local-probe/{index}',
            ScriptedBackend(actions(spec, 'complete')), evaluate_after=False)
        verify_seal(folder)
        ok = (result['execution_integrity'] == 'valid'
              and result['termination_reason'] == 'finish' and all(state(folder / 'final.db', spec['sql'])))
        return {'candidate': index, 'passed': ok, 'seconds': time.monotonic() - start,
                'termination_reason': result['termination_reason'], 'error': result.get('error')}
    beat = asyncio.create_task(heartbeat())
    try:
        results = await asyncio.gather(*(candidate(i) for i in range(4)))
    finally:
        finished = True
        await beat
    report = {'passed': all(r['passed'] for r in results), 'concurrency': 4, 'results': results,
              'max_event_loop_lag_seconds': max(lags, default=0), 'judge_requests': 0,
              'test_only': True, 'budget': budget.summary()}
    write_json(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    args = parser.parse_args()
    run, scratch, old = args.run.resolve(), args.scratch.resolve(), args.source_run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    scratch.mkdir(parents=True, exist_ok=False)
    from harness_awm.training.tasks import load_tasks, select_tasks
    from harness_awm.training.grpo import preflight
    from harness_awm.training.verl.pilot import archive_group
    from harness_awm.judge_runtime import config_hash
    manifest_path = stage_admissions(old / 'reward_admissions.json', scratch / 'admissions')
    prior = read_json(old / 'training/plan.json')
    keys = [t['scenario'] + ':' + t['native_task_id'] for t in prior['tasks']]
    controls = {key: read_json(old / 'controls.json')[key] for key in keys}
    config = RunConfig.model_validate({**read_json(old / 'config.json'),
        'reward_admissions': str(manifest_path), 'judge_rate_db': str(scratch / 'judge-rate.sqlite')})
    tasks, _, sources = load_tasks(ROOT / 'data/training_runtime_20261106_v1', 'grpo_train')
    tasks = select_tasks(tasks, keys)
    config = config.model_copy(update={'judge_inputs_manifest': sources})
    preflight(tasks, config, 8)
    report = asyncio.run(environment_probe(config, tasks, controls, scratch / 'environment-probe'))
    write_json(run / 'environment-probe.json', report)
    print(json.dumps(report), flush=True)
    if not report['passed']:
        raise RuntimeError('local environment probe failed; training not started')
    archive_group(scratch / 'admissions', run / 'archives/admissions.tar.gz')
    archive_group(scratch / 'environment-probe', run / 'archives/environment-probe.tar.gz')
    write_json(run / 'config.json', config.model_dump())
    write_json(run / 'controls.json', controls)
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    if not os.environ.get(config.judge_api.api_key_env):
        raise ValueError('Judge credential unavailable')
    argv = [sys.executable, '-u', str(ROOT / 'scripts/run_guarded_grpo_pilot.py'),
        '--run', str(run), '--scratch', str(scratch), '--model', prior['model'],
        '--reuse-calibration', str(manifest_path.parent), '--require-reused-calibration',
        '--reuse-probe', str(old / 'training'), '--gpu-memory-utilization', '.82',
        '--allocator-gb', '85', '--gpu-stop-gb', '90']
    with (run / 'supervisor.log').open('x') as log:
        proc = subprocess.Popen(argv, cwd=ROOT, env=dict(os.environ, PYTHONUNBUFFERED='1'),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(run / 'launch.json', {'pid': proc.pid, 'argv': argv, 'source_run': str(old),
        'judge_config_hash': config_hash(config), 'local_runtime': str(scratch),
        'new_calibration_requests': 0, 'environment_probe_passed': True})
    print(json.dumps({'pid': proc.pid, 'run': str(run), 'scratch': str(scratch)}), flush=True)


if __name__ == '__main__':
    main()
