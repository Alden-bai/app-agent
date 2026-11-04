import asyncio
from types import SimpleNamespace

from awm_audit import recovery
from awm_audit.common import digest, read_json, save_json
from awm_audit.config import LLMConfig


def test_recovery_preserves_valid_and_invalid_decisions_and_trajectory(tmp_path, monkeypatch):
    config = LLMConfig('https://example.invalid', 'fixture', 'fake')
    task = {'scenario': 'fixture', 'task_idx': 0, 'task': 'Task'}
    save_json(tmp_path / 'manifest.json', {'tasks': [task]})
    save_json(tmp_path / 'judge_baseline_config.json', config.public())
    monkeypatch.setattr(recovery, 'validate_assets', lambda *a: None)
    paths = []
    for i, status in enumerate(['valid', 'llm_invalid_output', 'llm_call_failure']):
        folder = tmp_path / str(i)
        save_json(folder / 'trajectory.json', {'trajectory': [{'real': i}], 'final_answer': 'answer'})
        result = {'status': 'valid', 'path': str(i), 'case': str(i), 'task': 'fixture:0',
                  'modes': {'sql': {'status': status, 'accepted': status == 'valid'}},
                  'sql_extraction': {'status': 'valid', 'details': {'fact': i}}}
        save_json(folder / 'evaluation.json', result)
        paths.append(f'{i}/evaluation.json')
    save_json(tmp_path / 'task_states.json', {'fixture:0': {**task, 'results': paths}})
    async def preflight(config):
        return {'llm_status': 'reachable'}
    seen = []
    async def judge(config, task, extraction, trajectory, folder, **kwargs):
        seen.append((extraction, trajectory))
        return {'status': 'valid', 'accepted': False}
    monkeypatch.setattr(recovery, 'preflight', preflight)
    monkeypatch.setattr(recovery, 'judge_sql', judge)
    before = digest(tmp_path / '2/evaluation.json')
    asyncio.run(recovery.resume_judge(tmp_path, config))
    state = read_json(tmp_path / 'task_states.json')['fixture:0']
    assert state['results'][:2] == paths[:2]
    assert state['results'][2] != paths[2]
    assert digest(tmp_path / '2/evaluation.json') == before
    assert seen == [({'fact': 2}, [{'real': 2}])]
    asyncio.run(recovery.resume_judge(tmp_path, config))
    assert len(seen) == 1


def test_auth_circuit_skips_client_construction(tmp_path, monkeypatch):
    from awm_audit import verification
    def unexpected(*args):
        raise AssertionError('No client allowed when circuit is open')
    monkeypatch.setattr(verification, 'LoggedClient', unexpected)
    result = asyncio.run(verification.judge_sql(None, {}, {}, [], tmp_path, circuit={'blocked': {'status_code': 402}}))
    assert result['status'] == 'llm_unavailable'
    assert result['accepted'] is None


def test_oracle_recheck_preserves_old_decision_and_exposes_new_pending_judge(tmp_path, monkeypatch):
    import sqlite3
    from awm_audit import recheck as module
    from awm_audit.oracle import TaskSpec, row_rule
    task = {'scenario': 'fixture', 'task_idx': 0, 'task': 'Upload proof'}
    initial = tmp_path / 'initials/fixture.db'
    initial.parent.mkdir()
    with sqlite3.connect(initial) as db:
        db.executescript('CREATE TABLE proof(id INTEGER PRIMARY KEY,photos INTEGER);INSERT INTO proof VALUES(1,0)')
    folder = tmp_path / 'case'
    from awm_audit.common import backup
    backup(initial, folder / 'initial.db')
    backup(initial, folder / 'final.db')
    with sqlite3.connect(folder / 'final.db') as db:
        db.execute('UPDATE proof SET photos=3')
    save_json(folder / 'trajectory.json', {'trajectory': [], 'final_answer': 'Done'})
    original = {'status': 'reference_failed', 'intended_label': True, 'case': 'positive', 'path': 'case',
                'task': 'fixture:0', 'modes': {}}
    save_json(folder / 'evaluation.json', original)
    save_json(tmp_path / 'manifest.json', {'tasks': [task]})
    save_json(tmp_path / 'task_states.json', {'fixture:0': {**task, 'status': 'needs_review', 'results': ['case/evaluation.json']}})
    monkeypatch.setattr(module, 'validate_assets', lambda *a: None)
    monkeypatch.setattr(module, 'load_spec', lambda *a: TaskSpec([], [row_rule('proof', {'id': 1, 'photos': 3})]))
    monkeypatch.setattr(module, 'verifier_entry', lambda *a: {})
    monkeypatch.setattr(module, 'execute_verifier', lambda entry, mode, folder: {'status': 'valid', 'accepted': mode == 'code', 'details': {'rows': []}})
    module.recheck(tmp_path, LLMConfig('https://example.invalid', 'fixture', 'fake'), tasks=['fixture:0'])
    state = read_json(tmp_path / 'task_states.json')['fixture:0']
    result = read_json(tmp_path / state['results'][0])
    assert state['status'] == 'completed_by_tools'
    assert result['modes']['code']['accepted'] is True
    assert result['modes']['sql']['status'] == 'llm_pending'
    assert read_json(folder / 'evaluation.json') == original
    assert result['oracle_recheck']['previous_evaluation'] == 'case/evaluation.json'
