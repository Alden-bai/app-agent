import sqlite3

from awm_audit.common import backup, digest, read_json, save_json
from awm_audit.config import LLMConfig
from awm_audit.oracle import TaskSpec, call, mutate, row_rule
from awm_audit.reuse import database_content_hash, reuse_case


def test_content_reuse_preserves_initial_failure_and_rejects_model_or_plan_change(tmp_path, monkeypatch):
    monkeypatch.setattr('awm_audit.reuse.execution_assets_match', lambda *a: True)
    config = LLMConfig('https://example.invalid', 'fixture', 'fake')
    initial = tmp_path / 'initial.db'
    with sqlite3.connect(initial) as db:
        db.executescript('CREATE TABLE items(id INTEGER PRIMARY KEY,quantity INTEGER);INSERT INTO items VALUES(1,1)')
    folder = tmp_path / 'old'
    backup(initial, folder / 'initial.db')
    backup(initial, folder / 'final.db')
    assert database_content_hash(initial) == database_content_hash(folder / 'initial.db')
    with sqlite3.connect(folder / 'final.db') as db:
        db.execute('UPDATE items SET quantity=2')
    spec = TaskSpec([call('set', quantity=2)], [row_rule('items', {'id': 1, 'quantity': 2})])
    save_json(tmp_path / 'rule.json', {'spec': spec.frozen()})
    save_json(folder / 'trajectory.json', {'trajectory': [], 'final_answer': 'Done', 'failure': None})
    previous = {'case': 'positive', 'status': 'valid', 'path': 'old', 'llm_config': config.public(),
                'modes': {'sql': {'status': 'llm_invalid_output', 'accepted': None}}}
    save_json(folder / 'evaluation.json', previous)
    state = {'rule_path': 'rule.json', 'results': ['old/evaluation.json']}
    result = reuse_case(tmp_path, tmp_path / 'task', state, spec, spec.cases()[0], initial, config, ['sql'], 'new')
    assert result['modes']['sql']['status'] == 'llm_invalid_output'
    assert result['source_evaluation'] == 'old/evaluation.json'
    assert reuse_case(tmp_path, tmp_path / 'task', state, spec, spec.cases()[0], initial,
                      LLMConfig(config.base_url, 'other', 'fake'), ['sql'], 'new') is None
    changed = {**spec.cases()[0], 'calls': [call('set', quantity=3)]}
    assert reuse_case(tmp_path, tmp_path / 'task', state, spec, changed, initial, config, ['sql'], 'new') is None
    with sqlite3.connect(initial) as db:
        db.execute('UPDATE items SET quantity=999')
    assert reuse_case(tmp_path, tmp_path / 'task', state, spec, spec.cases()[0], initial, config, ['sql'], 'new') is None


def test_explicit_reuse_rejects_tool_or_verifier_asset_drift(tmp_path, monkeypatch):
    from awm_audit.reuse import execution_assets_match
    from awm_audit import assets, verification
    environment = {'scenario': 'fixture', 'full_code': 'old tool'}
    entry = {'scenario': 'fixture', 'task_idx': 0, 'task': 'Set quantity'}
    folder = tmp_path / 'case'
    folder.mkdir()
    import json
    (folder / 'environment.jsonl').write_text(json.dumps(environment) + '\n')
    save_json(folder / 'verifier_code/request.json', {'entry': entry})
    monkeypatch.setattr(assets, 'scenario_asset', lambda *a: environment)
    monkeypatch.setattr(verification, 'verifier_entry', lambda *a: entry)
    previous = {'modes': {'code': {'status': 'valid'}}}
    assert execution_assets_match(tmp_path, {'scenario': 'fixture'}, folder, 'case/evaluation.json', previous, ['code'])
    environment['full_code'] = 'changed tool'
    assert not execution_assets_match(tmp_path, {'scenario': 'fixture'}, folder, 'case/evaluation.json', previous, ['code'])
    (folder / 'environment.jsonl').write_text(json.dumps(environment) + '\n')
    entry['task'] = 'Different task'
    assert not execution_assets_match(tmp_path, {'scenario': 'fixture'}, folder, 'case/evaluation.json', previous, ['code'])
