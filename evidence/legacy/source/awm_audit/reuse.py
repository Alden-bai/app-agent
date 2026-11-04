"""Explicit reuse of immutable executions after reviewed recipe amendments."""
import sqlite3
from pathlib import Path

from .common import digest, read_json
from .oracle import Rule, TaskSpec, evaluate


def execution_assets_match(root, state, folder, relative, previous, modes):
    from .assets import scenario_asset
    from .verification import verifier_entry
    if not (folder / 'environment.jsonl').exists():
        return False
    import json
    old_env = json.loads((folder / 'environment.jsonl').read_text())
    if digest(old_env) != digest(scenario_asset(root, 'gen_envs.jsonl', state['scenario'])):
        return False
    for mode in modes:
        try:
            entry = verifier_entry(root, state, mode)
        except ValueError as error:
            if previous['modes'][mode].get('status') != 'asset_error' or previous['modes'][mode].get('error') != str(error):
                return False
            continue
        candidates = [root / relative, folder / 'evaluation.json']
        request = next((p.parent / ('verifier_' + mode) / 'request.json' for p in candidates
                        if (p.parent / ('verifier_' + mode) / 'request.json').exists()), None)
        if request is None or digest(read_json(request)['entry']) != digest(entry):
            return False
    return True


def database_content_hash(path):
    # SQLite backup changes file headers; compare schema/data, not container bytes.
    with sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro', uri=True) as db:
        return digest(sorted(db.iterdump()))


def reuse_case(run_dir, taskdir, state, spec, case, initial, config, modes, fingerprint):
    groups = [state] if state else []
    groups += [read_json(p) for p in sorted((taskdir / 'state_history').glob('*.json'),
                                          key=lambda p:p.stat().st_mtime)]
    sources = []
    seen = set()
    for group in groups:
        rule_path = group.get('rule_path')
        if not rule_path or not (run_dir / rule_path).exists():
            continue
        frozen = read_json(run_dir / rule_path).get('spec')
        if not frozen:
            continue
        previous_spec = TaskSpec(**{**frozen, 'rules': [Rule(**r) for r in frozen['rules']]})
        recipes = {c['name']: c for c in previous_spec.cases()}
        recipe = recipes.get(case['name'])
        if not recipe or digest([recipe['calls'], recipe['answer']]) != digest([case['calls'], case['answer']]):
            continue
        for relative in group.get('results', []):
            if relative in seen:
                continue
            seen.add(relative)
            previous = read_json(run_dir / relative)
            if previous['case'] == case['name'] and previous.get('status') == 'valid':
                sources.append((relative, previous))
    # A preserved earliest judgment always takes precedence over a later repeat.
    sources.sort(key=lambda pair: (run_dir / pair[0]).stat().st_mtime)
    for relative, previous in sources:
        folder = run_dir / previous['path']
        payload = read_json(folder / 'trajectory.json')
        if payload.get('failure') or not all(m in previous.get('modes', {}) for m in modes):
            continue
        if not execution_assets_match(run_dir, state, folder, relative, previous, modes):
            continue
        if database_content_hash(folder / 'initial.db') != database_content_hash(initial):
            continue
        baseline = previous.get('llm_config') or read_json(run_dir / 'judge_baseline_config.json')
        if 'sql' in modes and baseline != config.public():
            continue
        checked = evaluate(spec, folder / 'initial.db', folder / 'final.db', payload['final_answer'], payload['trajectory'])
        if checked['status'] == 'valid' and checked['correct'] == case['expected']:
            return {**previous, 'fingerprint': fingerprint, 'oracle': checked, 'source_evaluation': relative,
                    'reuse_reason': 'identical frozen recipe and initial data; revised oracle confirms label; original judgments retained'}
    return None
