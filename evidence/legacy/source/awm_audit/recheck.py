"""Versioned oracle corrections on recorded tool executions; no paid resampling."""
from __future__ import annotations

import copy
from pathlib import Path

from .assets import validate_assets
from .common import backup, code_hash, digest, now, read_json, save_json
from .oracle import evaluate, load_spec
from .verification import execute_verifier, verifier_entry


def recheck(run_dir, config, *, tasks):
    if not tasks:
        raise ValueError('Recheck requires explicit --task and a reviewed rule correction')
    root = Path(run_dir).resolve()
    manifest = read_json(root / 'manifest.json')
    validate_assets(root, manifest)
    states = read_json(root / 'task_states.json')
    if set(tasks) - set(states):
        raise ValueError('Task outside frozen sample')
    for key in tasks:
        state = copy.deepcopy(states[key])
        initial = root / 'initials' / (state['scenario'] + '.db')
        spec = load_spec(state['scenario'], state['task_idx'], initial)
        if spec.status != 'ready':
            raise ValueError('Recheck only applies to ready revised rules')
        frozen_hash = digest(spec.frozen())
        recheck_hash = digest({'rule': frozen_hash, 'implementation': code_hash(), 'initial': digest(initial), 'config': config.public()})
        rule_path = root / 'tasks' / state['scenario'] / str(state['task_idx']) / 'rules' / (frozen_hash + '.json')
        save_json(rule_path, {'task': {k: state[k] for k in ('scenario', 'task_idx', 'task')},
                             'rule_hash': frozen_hash, 'initial_sha256': digest(initial),
                             'basis': 'reviewed independent rule amendment; existing episode retained', 'spec': spec.frozen()})
        paths = []
        for previous in state['results']:
            original = read_json(root / previous)
            folder = root / original['path']
            result = copy.deepcopy(original)
            target = folder / 'oracle_recheck' / recheck_hash / 'evaluation.json'
            if target.exists():
                result = read_json(target)
            elif original.get('failure'):
                # Infrastructure errors remain errors even if a partial state happens to pass.
                paths.append(previous)
                continue
            else:
                work = target.parent
                payload = read_json(folder / 'trajectory.json')
                oracle = evaluate(spec, folder / 'initial.db', folder / 'final.db', payload['final_answer'], payload['trajectory'])
                result['oracle'] = oracle
                correct_label = oracle.get('status') == 'valid' and oracle['correct'] == original['intended_label']
                result['status'] = ('valid' if correct_label else 'oracle_unknown' if oracle['status'] != 'valid'
                                    else 'reference_failed' if original['intended_label'] else 'negative_not_wrong')
                if correct_label:
                    for mode in ('code', 'sql'):
                        if mode in result.get('modes', {}):
                            continue  # Retain the original decision on the same immutable trajectory.
                        work.mkdir(parents=True, exist_ok=True)
                        for name in ('initial.db', 'final.db'):
                            backup(folder / name, work / name)
                        save_json(work / 'trajectory.json', payload)
                        try:
                            entry = verifier_entry(root, state, mode)
                        except ValueError as error:
                            result['modes'][mode] = {'status': 'asset_error', 'accepted': None, 'error': str(error)}
                            continue
                        verdict = execute_verifier(entry, mode, work)
                        if mode == 'sql' and verdict['status'] == 'valid':
                            result['sql_extraction'] = verdict
                            result['sql_evidence_review'] = {'status': 'needs_review'}
                            verdict = {'status': 'llm_pending', 'accepted': None}
                        result['modes'][mode] = verdict
                result['rule_hash'] = frozen_hash
                result['llm_config'] = config.public()
                result['oracle_recheck'] = {'previous_evaluation': previous, 'at': now(),
                                           'reason': 'corrected independent rule; no new MCP execution or Judge resampling'}
                save_json(target, result)
            paths.append(str(target.relative_to(root)))
            print(f"{key} {result['case']}: {result['status']}", flush=True)
        state['results'] = paths
        state['rule_hash'] = frozen_hash
        state['rule_path'] = str(rule_path.relative_to(root))
        state['updated_at'] = now()
        state['status'] = 'completed_by_tools' if any(read_json(root / p).get('status') == 'valid' and
                              read_json(root / p).get('oracle', {}).get('correct') is True for p in paths) else 'needs_review'
        if state['status'] == 'completed_by_tools':
            state.pop('reason', None)
        elif spec.notes:
            state['reason'] = spec.notes
            state['evidence'] = spec.evidence
        save_json(root / 'tasks' / state['scenario'] / str(state['task_idx']) / 'state.json', state)
        states[key] = state
        save_json(root / 'task_states.json', states)
    return states
