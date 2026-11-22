import asyncio

from harness_awm.io import read_json, write_json
from scripts import judge_today_skills as judge


def test_preparation_failure_is_reported_and_other_tasks_finish(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    output = tmp_path / 'output'
    output.mkdir()
    candidates = [
        dict(rollout_id=f't{i}', task_instance_id=f'task{i}', path=f'episode{i}',
             complete=True, skill_group='workflow', hashes={})
        for i in range(2)
    ]
    write_json(source / 'judge_sources.json', {
        c['task_instance_id']: {'original': {}, 'rubric': {}} for c in candidates
    })
    plan = dict(run=str(source), input_hashes={}, config={}, eligible=2,
                not_before='2000-01-01T00:00:00+00:00', candidates=candidates)
    monkeypatch.setattr('dotenv.dotenv_values', lambda _: {'DEEPSEEK_API_KEY': 'offline-test'})
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'offline-test')

    def prepare(folder, original, rubric):
        if folder.name == 'episode0':
            raise ValueError('broken evidence')
        return {}

    async def evaluate(*args):
        return dict(evaluation_valid=True, evaluation_status='valid',
                    task_status='confirmed_pass', items=[], errors=[])

    async def forbidden_api(*args, **kwargs):
        raise AssertionError('Regression test must never call an API')

    monkeypatch.setattr(judge, 'prepare_input', prepare)
    monkeypatch.setattr(judge.judge_runtime, 'evaluate_input', evaluate)
    monkeypatch.setattr(judge.judge_runtime, 'api_request', forbidden_api)
    monkeypatch.setattr(judge, '_api_request_once', forbidden_api)
    asyncio.run(judge.execute(output, plan))

    rows = read_json(output / 'per_task.json')
    assert rows[0]['outcome'] == '无法判定'
    assert rows[0]['errors'] == ['ValueError']
    assert rows[1]['outcome'] == '正确'
    status = read_json(output / 'summary.json')
    assert status['completed'] == 2
    assert status['stage'] == 'needs_attention'
    assert status['budget']['physical_requests'] == 0
