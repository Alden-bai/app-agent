"""Resume transport failures against immutable existing trajectories, never resample decisions."""
from __future__ import annotations

import copy
from pathlib import Path
from uuid import uuid4

from .assets import validate_assets
from .common import digest, now, read_json, save_json
from .runner import preflight
from .verification import judge_sql


async def resume_judge(run_dir, config, *, tasks=None, limit=None):
    root = Path(run_dir).resolve()
    manifest = read_json(root / 'manifest.json')
    validate_assets(root, manifest)
    states = read_json(root / 'task_states.json')
    known = {f"{t['scenario']}:{t['task_idx']}": t for t in manifest['tasks']}
    if tasks and set(tasks) - set(known):
        raise ValueError('Requested task outside frozen sample')
    pending = []
    for key, state in states.items():
        if tasks and key not in tasks:
            continue
        for index, relative in enumerate(state['results']):
            result = read_json(root / relative)
            mode = result.get('modes', {}).get('sql', {})
            if result.get('status') != 'valid' or mode.get('status') not in (
                    'llm_call_failure', 'llm_unavailable', 'not_tested', 'llm_pending'):
                continue
            extraction = result.get('sql_extraction', {})
            if extraction.get('status') != 'valid':
                continue
            folder = root / result['path']
            # This path always refers to the original MCP case, even after recovery.
            previous_requests = folder / 'llm_requests.json'
            if previous_requests.exists():
                logs = read_json(previous_requests)
                if logs:
                    request = logs[0]['request']
                    if request['model'] != config.model:
                        raise ValueError('Recovery model differs from original request; use a separate run')
            # Preserve the endpoint and transport configuration used by the original attempt.
            frozen_config = result.get('llm_config')
            if frozen_config is None:
                baseline = root / 'judge_baseline_config.json'
                if baseline.exists():
                    frozen_config = read_json(baseline)
                else:
                    raise ValueError('Missing original model/endpoint baseline for recovery')
            if frozen_config != config.public():
                raise ValueError('Recovery configuration differs from original run')
            pending.append((key, index, relative, result, folder, extraction))
    pending = pending[:limit] if limit is not None else pending
    if not pending:
        print('No eligible Judge transport failures to resume.', flush=True)
        return states
    checked = await preflight(config)
    save_json(root / 'preflight.json', checked)
    if checked['llm_status'] != 'reachable':
        raise RuntimeError('LLM preflight failed; no recovery Judge requests submitted')
    circuit = {}
    for key, index, relative, original, folder, extraction in pending:
        if circuit.get('blocked'):
            break
        recovery = folder / 'judge_recovery' / (now().replace(':', '') + '-' + uuid4().hex[:8])
        recovery.mkdir(parents=True)
        trajectory = read_json(folder / 'trajectory.json')
        verdict = await judge_sql(config, known[key], extraction['details'], trajectory['trajectory'],
                                  recovery, circuit=circuit)
        result = copy.deepcopy(original)
        result['modes']['sql'] = verdict
        result['llm_config'] = config.public()
        result['recovery'] = {'previous_evaluation': relative, 'at': now(),
                              'kind': 'transport_only', 'trajectory_sha256': digest(folder / 'trajectory.json'),
                              'sql_extraction_sha256': digest(extraction)}
        target = recovery / 'evaluation.json'
        save_json(target, result)
        states[key]['results'][index] = str(target.relative_to(root))
        states[key]['updated_at'] = now()
        save_json(root / 'tasks' / states[key]['scenario'] / str(states[key]['task_idx']) / 'state.json', states[key])
        save_json(root / 'task_states.json', states)
        print(f"{key} {result['case']}: sql={verdict.get('accepted')}({verdict['status']})", flush=True)
    return states
