"""Exercise unchanged evidence validation and review through the new transport."""
import asyncio
import copy
from pathlib import Path

import pytest

from harness_awm.backend import Budget
from harness_awm.contracts import RunConfig
from harness_awm.io import write_json
from scripts import sft_optimized_judge_20261110 as opt
from test_judge_v2 import bundle, assessment, Responder
from test_sft_optimized_judge_20261110 import Gate, response


@pytest.mark.parametrize('state,expected', [('pass', 'confirmed_pass'), ('fail', 'failed'),
                                         ('unknown', 'provisional_pass')])
@pytest.mark.parametrize('truncate', [False, True])
def test_transport_keeps_all_three_decisions_and_mandatory_review(
        bundle, tmp_path, monkeypatch, state, expected, truncate):
    runtime = opt.r.recovery_judge()
    answer = assessment((state, 'pass'), refs=['E00002'])
    config = RunConfig(judge='api', judge_protocol='rubric_v2',
                       judge_v2={'query_rounds': 12, 'review_all_decisions': True})
    original = Responder([copy.deepcopy(answer), copy.deepcopy(answer)])
    monkeypatch.setattr(runtime, 'api_request', original)
    baseline = asyncio.run(runtime.evaluate_input(
        bundle, config, Budget(tmp_path / 'baseline.sqlite', 32, None),
        asyncio.Semaphore(1), tmp_path / 'baseline'))
    new_responder = Responder([copy.deepcopy(answer), copy.deepcopy(answer)])
    wires = []
    async def api(config, wire, budget, key, stage, folder, **kw):
        wires.append(copy.deepcopy(wire))
        if truncate and len(wires) == 1:
            rid = budget.reserve(key, stage, wire['messages'], wire['max_completion_tokens'])
            budget.finish(rid, {'total_tokens': 100})
            raw = response('length', '{truncated untrusted JSON')
            write_json(Path(folder) / 'request.json', wire)
            write_json(Path(folder) / 'response.json', raw)
            return raw
        return await new_responder(config, wire, budget, key, stage, folder)
    async def paced(config, kwargs, budget, key, stage, folder, **kw):
        return await opt.request_with_fallback(api, Gate(), config, kwargs, budget, key, stage, folder, **kw)
    monkeypatch.setattr(runtime, 'api_request', paced)
    budget = Budget(tmp_path / 'optimized.sqlite', 32, None)
    optimized = asyncio.run(runtime.evaluate_input(
        bundle, config, budget, asyncio.Semaphore(1), tmp_path / 'optimized'))
    assert optimized['evaluation_status'] == baseline['evaluation_status'] == 'valid'
    assert optimized['task_status'] == baseline['task_status'] == expected
    assert optimized['business_label'] == baseline['business_label']
    assert new_responder.calls == original.calls == 2  # includes mandatory review
    assert budget.summary()['physical_requests'] == 2 + int(truncate)
    assert wires[0]['max_completion_tokens'] == 4096
    if truncate:
        assert wires[1]['max_completion_tokens'] == 8192
        assert wires[0]['messages'] == wires[1]['messages']


def test_invalid_output_cannot_bypass_original_validator(bundle, tmp_path, monkeypatch):
    runtime = opt.r.recovery_judge()
    invalid = assessment(refs=['nonexistent_evidence'])
    responder = Responder([invalid, invalid, invalid])
    async def paced(config, kwargs, budget, key, stage, folder, **kw):
        return await opt.request_with_fallback(responder, Gate(), config, kwargs,
                                              budget, key, stage, folder, **kw)
    monkeypatch.setattr(runtime, 'api_request', paced)
    result = asyncio.run(runtime.evaluate_input(
        bundle, RunConfig(judge='api', judge_protocol='rubric_v2'),
        Budget(tmp_path / 'budget.sqlite', 32, None), asyncio.Semaphore(1), tmp_path / 'results'))
    assert result['evaluation_status'] == 'evaluation_error'
    assert not result['evaluation_valid']
    assert responder.calls == 3


def test_live_scheduler_retries_once_and_holds_unresolved_without_selecting(tmp_path, monkeypatch):
    from collections import Counter
    from types import SimpleNamespace
    from harness_awm.io import read_json
    run = tmp_path / 'run'
    (run / 'judge_jobs').mkdir(parents=True)
    monkeypatch.setattr(opt, 'RUN', run)
    monkeypatch.setattr(opt, 'DRAIN', False)
    records, attempts, status_events = {}, Counter(), []
    rows = [('model', {'task_instance_id': str(i)}, tmp_path / str(i), run / 'judge_jobs' / str(i))
            for i in range(17)]
    def inventory():
        pending, held = [], []
        for row in rows:
            key = str(row[3])
            record = records.get(key)
            state = opt.disposition(record, [{}] * attempts[key], 0)
            if state in ('pending', 'retry_pending'):
                pending.append(row)
            elif state != 'valid':
                held.append({'job': key, 'reason': state})
        return pending, dict(records), held, {'total': 17, 'original_scored': 0, 'untrusted': 0}
    async def assess(row, *args):
        job = row[3]
        key = str(job)
        attempts[key] += 1
        invalid = row[1]['task_instance_id'] == '3' or (row[1]['task_instance_id'] == '0' and attempts[key] == 1)
        record = {'episode': str(job / 'episode'), 'score': None if invalid else 0,
                  'errors': ['query_round_budget_exhausted'] if invalid else []}
        write_json(job / 'attempts.json', [{}] * attempts[key])
        write_json(job / 'episode/episode_result.json', {'evaluation_status': 'evaluation_error' if invalid else 'valid'})
        records[key] = record
        return record
    gate = SimpleNamespace(target=16, refresh=lambda: None,
                           snapshot=lambda: {'target_concurrency': 16, 'cooldown_until': 0})
    runtime = SimpleNamespace(api_request=None, config_hash=lambda config: 'original')
    monkeypatch.setattr(opt, 'inventory', inventory)
    monkeypatch.setattr(opt, 'assess', assess)
    monkeypatch.setattr(opt.adaptive, 'AdaptiveGate', lambda path: gate)
    monkeypatch.setattr(opt.r, 'recovery_judge', lambda: runtime)
    monkeypatch.setattr(opt.r.recovery, 'collect_repairs', lambda: [])
    monkeypatch.setattr(opt.r, 'status', lambda *a, **kw: status_events.append((a, kw)))
    monkeypatch.setattr(opt.base.shutil, 'disk_usage', lambda path: SimpleNamespace(free=10 * 1024**3))
    result = asyncio.run(opt.drive({'config': {}}, {'implementation_sha256': 'fixed'}))
    assert result == 'held'
    assert attempts[str(rows[0][3])] == attempts[str(rows[3][3])] == 2
    assert all(attempts[str(row[3])] == 1 for row in rows if row[1]['task_instance_id'] not in ('0', '3'))
    assert status_events[-1][1]['completed'] == 16
    assert status_events[-1][1]['remaining'] == 1
    assert status_events[-1][0] == ('judge', 'needs_attention')
    assert read_json(run / 'optimized_pilot.json')['stage'] == 'complete'
    assert read_json(run / 'optimized_held.json')['jobs'][0]['reason'] == 'held_assessment_retry_limit'
