"""Verify raw evidence references and reproduce supported admission checks."""
from __future__ import annotations

import json
from pathlib import Path

from awm_audit.common import digest, read_json
from .certification import GATES, coverage, decide, five_slots_pass, sql_review_valid


def _inside(run, path):
    path = Path(path)
    if not path.is_absolute():
        path = run / path
    path = path.resolve()
    if not path.is_relative_to(run):
        raise ValueError('Evidence escapes release directory: ' + str(path))
    return path


def seal_task(run, task_folder, environment_folder):
    run = Path(run).resolve()
    return {str(p.resolve()): digest(p) for folder in (task_folder, environment_folder)
            for p in sorted(Path(folder).rglob('*')) if p.is_file()
            and _inside(run, p) == p.resolve()}


def validate_certificate(run, manifest, cert):
    run = Path(run).resolve()
    if cert.get('freeze_sha256') != manifest['freeze_sha256']:
        raise ValueError('Certificate version differs from frozen release')
    inventory = cert.get('evidence_inventory', {})
    if not inventory:
        raise ValueError('Certificate has no raw evidence inventory')
    resolved = {}
    for name, sha in inventory.items():
        path = _inside(run, name)
        if not path.is_file() or digest(path) != sha:
            raise ValueError('Certificate raw evidence missing or changed: ' + str(path))
        resolved[str(path)] = sha
    for name in cert.get('evidence', []):
        if str(_inside(run, name)) not in resolved:
            raise ValueError('Unsealed certificate evidence reference: ' + name)
    for defect in cert.get('defects', []):
        for name in defect.get('evidence', []):
            if str(_inside(run, name)) not in resolved:
                raise ValueError('Unsealed defect evidence reference: ' + name)
    task = next((t for t in manifest['tasks'] if t['ledger_id'] == cert['ledger_id']), None)
    if task is None or cert.get('task_sha256') != task['task_sha256']:
        raise ValueError('Certificate task binding differs from original ledger')
    # Recompute the decision before validating it; a stored decision string
    # cannot suppress checks after a gate has been accidentally changed.
    decision, _ = decide(cert)
    if decision != 'keep':
        return True
    if not all(cert.get('gates', {}).get(g) is True for g in GATES):
        raise ValueError('A required keep gate is missing')
    if (task['scenario'], task['task_idx']) != ('e_commerce_10', 7):
        raise ValueError('No independently implemented certificate revalidator for this task')
    from . import reference_ecommerce as plugin
    from .provenance import resolve
    from awm_audit.assets import scenario_asset
    from awm_audit.verification import verifier_entry
    folder = run / 'evidence/tasks' / task['ledger_id'].replace(':', '_')
    spec = read_json(folder / 'task_spec.json')
    if digest(spec) != cert.get('spec_sha256'):
        raise ValueError('Frozen requirement/case specification changed')
    cases = cert.get('cases', [])
    if len(cases) != len(spec['cases']) or {c['name'] for c in cases} != {c['name'] for c in spec['cases']}:
        raise ValueError('A frozen case was omitted or duplicated')
    if not coverage(spec, cases)['passed'] or not five_slots_pass(cases):
        raise ValueError('Coverage or five-slot admission does not reproduce')
    registry = plugin.public_contracts(scenario_asset(run, 'gen_spec.jsonl', task['scenario']))
    entry = verifier_entry(run, task, 'sql')
    expected_cases = {c['name']: c for c in spec['cases']}
    for case in cases:
        root = _inside(run, case['evidence'])
        recipe = expected_cases[case['name']]
        trace = read_json(root / 'trajectory.json')
        if trace.get('failure') or not case.get('reachable'):
            raise ValueError('Keep case is not a real reachable execution')
        independent = plugin.evaluate(spec, root / 'initial.db', root / 'final.db',
                                      trace['final_answer'], trace['trajectory'])
        if independent != case['independent'] or independent != read_json(root / 'independent.json'):
            raise ValueError('Independent rules do not reproduce the stored label')
        if independent.get('status') != 'valid' or independent.get('correct') is not recipe['expected']:
            raise ValueError('Frozen independent case label is disputed')
        # Regenerate parameter provenance from task, public contract and actual
        # prior tool return objects; executor database access is never available.
        returns, provenance = [], []
        aliases = read_json(root / 'tool_aliases.json')
        tools = {t['name']: t for t in read_json(root / 'tools.json')}
        events = [e for e in trace['trajectory'] if any(t.get('name') == 'call_tool' for t in e.get('tool_calls', []))]
        if len(events) != len(recipe['calls']):
            raise ValueError('Real tool call count differs from frozen recipe')
        for index, (step, event) in enumerate(zip(recipe['calls'], events)):
            args = resolve(step['arguments'], task['task'], returns, provenance, public_contracts=registry)
            call = next(c for c in event['tool_calls'] if c.get('name') == 'call_tool')
            actual_name = call['arguments']['tool_name']
            expected_name = aliases.get(step['tool'], step['tool'])
            if actual_name != expected_name and not actual_name.startswith(expected_name + '_'):
                raise ValueError('Unexpected exposed tool in recorded execution')
            schema = tools[actual_name]['inputSchema']
            for wrapper in ('body', 'payload'):
                if wrapper in args and wrapper not in schema.get('properties', {}) and isinstance(args[wrapper], dict):
                    body = args.pop(wrapper); args.update(body)
            args = {k: v for k, v in args.items() if v is not None or k in schema.get('required', [])}
            if args != json.loads(call['arguments']['arguments']):
                raise ValueError('Recorded arguments do not reproduce from allowed sources')
            response = read_json(root / f'response_{index}.json')
            plain = response.get('structuredContent')
            if plain is None:
                texts = [c['text'] for c in response['content'] if c['type'] == 'text']
                if len(texts) == 1:
                    try: plain = json.loads(texts[0])
                    except ValueError: plain = texts[0]
                else: plain = texts
            if plain != json.loads(event['tool_response']['content']):
                raise ValueError('Trajectory tool return differs from raw MCP response')
            returns.append(plain)
        if provenance != read_json(root / 'argument_provenance.json'):
            raise ValueError('Argument provenance trace changed')
        extraction = read_json(root / 'extraction.json')
        review = plugin.sql_review(entry, extraction.get('evidence'), root / 'initial.db', root / 'final.db')
        if extraction.get('status') != 'valid' or not sql_review_valid(review):
            raise ValueError('Complete independent SQL review does not reproduce')
        if review != read_json(root / 'sql_review.json'):
            raise ValueError('SQL evidence review drift')
        for slot in case['judge_slots']:
            if type(slot.get('slot')) is not int or slot['slot'] not in range(5):
                raise ValueError('Invalid frozen slot position')
            slot_folder = root / 'judge' / str(slot['slot'])
            if slot != read_json(slot_folder / 'result.json'):
                raise ValueError('Judge slot changed')
            requests = read_json(slot_folder / 'requests.json')
            physical = [r for r in requests if 'request_id' in r]
            responses = [r for r in physical if 'response' in r]
            if len(physical) > 3 or len(responses) != 1 or responses[0] is not physical[-1]:
                raise ValueError('Judge slot resampled or raw response missing')
            if responses[0]['response']['choices'][0].get('finish_reason') != 'stop':
                raise ValueError('Truncated/invalid output was counted as valid')
    return True
