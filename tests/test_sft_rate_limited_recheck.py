"""Offline regression checks for the fixed 429 cohort recheck transport."""

import asyncio
import copy
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from harness_awm.backend import Budget
from harness_awm.contracts import RunConfig
from harness_awm.io import file_hash, read_json, seal, write_json
from scripts import sft_rate_limited_recheck_20261110 as recheck


class ProviderError(Exception):
    def __init__(self, status=429, body=None, retry_after=None):
        super().__init__(str(body))
        self.status_code = status
        self.body = body or {}
        self.response = SimpleNamespace(headers={'retry-after': retry_after})


class Gate:
    def __init__(self, *args):
        self.acquired, self.finished, self.limited_until = [], [], []

    async def acquire(self, upper, check=lambda: None):
        check()
        self.acquired.append(upper)
        return len(self.acquired)

    def finish(self, event, usage=None):
        self.finished.append((event, usage))

    def limited(self, until):
        self.limited_until.append(until)

    def cooldown(self, until):
        self.limited_until.append(until)

    def snapshot(self):
        return {'last_60s_requests': len(self.acquired)}


def response(reason='stop', text='valid complete answer'):
    return {'choices': [{'finish_reason': reason, 'message': {'content': text}}],
            'usage': {'total_tokens': 100}}


def request():
    return {'messages': [{'role': 'system', 'content': 'Unchanged assessment rules'},
                         {'role': 'user', 'content': 'Original sealed evidence'}],
            'temperature': 0, 'max_completion_tokens': 8192}


@pytest.mark.parametrize('code', [
    'insufficient_quota', 'billing_hard_limit', 'billing_not_active',
    'organization_spend_limit_exceeded', 'project_spend_limit_exceeded',
    'organization_usage_limit_exceeded',
])
def test_quota_exhaustion_pauses_without_retry(tmp_path, monkeypatch, code):
    calls = []

    async def api(*args, **kwargs):
        calls.append(args)
        raise ProviderError(body={'code': code})

    monkeypatch.setattr(recheck.random, 'uniform', lambda *_: 0)
    gate = Gate()
    transport = recheck.Transport(api, gate, tmp_path)
    budget = Budget(tmp_path / 'budget.sqlite', 32, None)
    with pytest.raises(recheck.TransportPaused):
        asyncio.run(transport.request(RunConfig().judge_api, request(), budget,
                                      'logical', 'assessment', tmp_path / 'request'))
    assert len(calls) == len(gate.acquired) == 1
    assert read_json(tmp_path / 'transport_stop.json')['policy'] == 'pause'


def test_retry_after_accepts_http_date_and_keeps_longer_provider_wait():
    now = datetime(2026, 11, 10, tzinfo=timezone.utc).timestamp()
    later = datetime.fromtimestamp(now + 125, timezone.utc)
    error = ProviderError(retry_after=format_datetime(later, usegmt=True))
    assert recheck.retry_seconds(error, 0, now=now) == 125
    assert recheck.retry_seconds(ProviderError(retry_after='17.5'), 0, now=now) == 17.5
    assert recheck.retry_seconds(ProviderError(body='try again in 1m30s'), 0, now=now) == 90
    assert recheck.retry_seconds(ProviderError(retry_after='invalid'), 1, now=now) == 8


@pytest.mark.parametrize('failures,expected_calls,paused', [(2, 3, False), (9, 4, True)])
def test_transient_retries_keep_input_and_stop_after_four_attempts(
        tmp_path, monkeypatch, failures, expected_calls, paused):
    wires = []

    async def api(config, wire, budget, key, stage, folder, **kwargs):
        wires.append(copy.deepcopy(wire))
        if len(wires) <= failures:
            raise ProviderError(body={'code': 'rate_limit_exceeded'})
        result = response()
        write_json(Path(folder) / 'response.json', result)
        return result

    monkeypatch.setattr(recheck.random, 'uniform', lambda *_: 0)
    wire, gate = request(), Gate()
    transport = recheck.Transport(api, gate, tmp_path)
    budget = Budget(tmp_path / 'budget.sqlite', 32, None)
    call = transport.request(RunConfig().judge_api, wire, budget,
                             'logical', 'assessment', tmp_path / 'request')
    if paused:
        with pytest.raises(recheck.TransportPaused):
            asyncio.run(call)
    else:
        assert asyncio.run(call) == response()
    assert len(wires) == len(gate.acquired) == expected_calls
    assert all(w == wires[0] for w in wires)
    assert len(gate.limited_until) == min(failures, expected_calls)
    assert bool(transport.stopped) is paused


def test_gate_keeps_reservations_monotonic_and_persists_spacing(tmp_path, monkeypatch):
    monkeypatch.setattr(recheck.time, 'time', lambda: 1000.0)
    gate = recheck.SmoothGate(tmp_path / 'gate.sqlite')
    event = asyncio.run(gate.acquire(20000))
    gate.finish(event, 100)
    with sqlite3.connect(gate.path) as db:
        assert db.execute('SELECT tokens FROM events WHERE id=?', (event,)).fetchone()[0] == 20000
    restarted = recheck.SmoothGate(gate.path)
    assert restarted.delay(1001, 100) == pytest.approx(20000 * 60 / 85000 - 1)
    gate.finish(event, 21000)
    gate.finish(event)
    with sqlite3.connect(gate.path) as db:
        assert db.execute('SELECT tokens,status FROM events WHERE id=?', (event,)).fetchone() == (21000, 'unknown_reserved')
    gate.limited(1030)
    lowered = gate.tpm
    gate.limited(1020)
    assert gate.tpm == lowered  # One backoff for one concurrent 429 wave.
    assert gate.delay(1001, 100) >= 29
    assert gate.snapshot()['cooldown_until'] == 1030


def test_larger_request_waits_for_empty_window_and_obeys_safe_ceiling(tmp_path, monkeypatch):
    monkeypatch.setattr(recheck.time, 'time', lambda: 1000.0)
    gate = recheck.SmoothGate(tmp_path / 'gate.sqlite')
    with sqlite3.connect(gate.path) as db:
        db.execute('UPDATE pacing SET tpm=40000 WHERE id=1')
        db.execute("INSERT INTO events(started,tokens,status) VALUES(990,1,'response')")
    assert gate.delay(1000, 80000) >= 50
    assert gate.delay(1051, 80000) == 0
    with pytest.raises(ValueError, match='single_request_above_safe_provider_allowance'):
        gate.delay(1051, 95001)
    with sqlite3.connect(gate.path) as db:
        db.execute('DELETE FROM events')
    asyncio.run(gate.acquire(80000))
    assert gate.delay(1061, 1) == pytest.approx(59)
    assert gate.tpm == 40000


def test_cached_response_replay_skips_api_and_budget_even_after_restart(tmp_path):
    calls = []

    async def api(config, wire, budget, key, stage, folder, **kwargs):
        calls.append(copy.deepcopy(wire))
        rid = budget.reserve(key, stage, wire['messages'], wire['max_completion_tokens'])
        result = response()
        budget.finish(rid, result['usage'])
        write_json(Path(folder) / 'response.json', result)
        return result

    budget = Budget(tmp_path / 'budget.sqlite', 32, None)
    config, wire, gate = RunConfig().judge_api, request(), Gate()
    original = recheck.Transport(api, gate, tmp_path)
    first = asyncio.run(original.request(config, wire, budget, 'attempt1', 'assessment', tmp_path / 'first'))
    restarted = recheck.Transport(api, gate, tmp_path)
    second = asyncio.run(restarted.request(config, wire, budget, 'attempt2', 'assessment', tmp_path / 'second'))
    assert first == second
    assert len(calls) == len(gate.acquired) == budget.summary()['physical_requests'] == 1
    assert (tmp_path / 'second/4096/cache_replay.json').is_file()
    cached = read_json(next((tmp_path / 'response_cache').glob('*.json')))
    write_json(cached['response_path'], response(text='modified after caching'))
    with pytest.raises(ValueError, match='cached_response_hash_mismatch'):
        asyncio.run(restarted.request(config, wire, budget, 'attempt3', 'assessment', tmp_path / 'third'))
    assert len(calls) == 1


def test_crash_before_cache_index_commit_recovers_paid_response_from_intent(tmp_path):
    calls = []

    async def api(config, wire, budget, key, stage, folder, **kwargs):
        calls.append(key)
        rid = budget.reserve(key, stage, wire['messages'], wire['max_completion_tokens'])
        result = response()
        budget.finish(rid, result['usage'])
        write_json(Path(folder) / 'response.json', result)
        return result

    budget_path = tmp_path / 'budget.sqlite'
    budget, gate = Budget(budget_path, 32, None), Gate()
    config, wire = RunConfig().judge_api, request()
    original = recheck.Transport(api, gate, tmp_path)
    first = asyncio.run(original.request(config, wire, budget, 'original',
                                          'assessment', tmp_path / 'first'))
    cache_entry = next((tmp_path / 'response_cache').glob('*.json'))
    saved = read_json(cache_entry)
    pending = list((tmp_path / 'response_cache/pending').glob('*/*.json'))
    assert len(pending) == 1
    assert read_json(pending[0])['physical_key'] == calls[0]
    assert budget.request_record(calls[0]) is not None
    cache_entry.unlink()  # Response persisted; index write was lost at shutdown.
    restarted = recheck.Transport(api, gate, tmp_path)
    second = asyncio.run(restarted.request(config, wire, Budget(budget_path, 32, None),
                                            'resume', 'assessment', tmp_path / 'second'))
    assert second == first
    assert read_json(cache_entry) == saved
    assert len(calls) == len(gate.acquired) == budget.summary()['physical_requests'] == 1
    assert (tmp_path / 'second/4096/cache_replay.json').exists()


def test_orphan_intent_without_budget_record_cannot_supply_a_cached_answer(tmp_path):
    budget, gate = Budget(tmp_path / 'budget.sqlite', 32, None), Gate()
    config, wire = RunConfig().judge_api, request()
    effective_wire = dict(wire, max_completion_tokens=4096)
    fingerprint = recheck.r.digest({'api': config.model_dump(mode='json'),
                                    'wire': effective_wire, 'stage': 'assessment'})
    orphan = tmp_path / 'orphan/response.json'
    write_json(orphan, response(text='unaccounted answer'))
    write_json(tmp_path / 'response_cache/pending' / fingerprint / 'orphan.json',
               {'request_sha256': fingerprint, 'response_path': str(orphan),
                'physical_key': 'missing-from-budget'})
    calls = []

    async def api(config, wire, budget, key, stage, folder, **kwargs):
        calls.append(key)
        rid = budget.reserve(key, stage, wire['messages'], wire['max_completion_tokens'])
        result = response(text='accounted answer')
        budget.finish(rid, result['usage'])
        write_json(Path(folder) / 'response.json', result)
        return result

    transport = recheck.Transport(api, gate, tmp_path)
    actual = asyncio.run(transport.request(config, wire, budget, 'logical',
                                           'assessment', tmp_path / 'request'))
    assert actual['choices'][0]['message']['content'] == 'accounted answer'
    assert len(calls) == budget.summary()['physical_requests'] == 1


def test_4096_truncation_fallback_uses_exact_original_input_and_replays(tmp_path):
    wires = []

    async def api(config, wire, budget, key, stage, folder, **kwargs):
        wires.append(copy.deepcopy(wire))
        rid = budget.reserve(key, stage, wire['messages'], wire['max_completion_tokens'])
        result = response('length', '{partial invalid output') if len(wires) == 1 else response()
        budget.finish(rid, result['usage'])
        write_json(Path(folder) / 'response.json', result)
        return result

    wire, gate = request(), Gate()
    before = copy.deepcopy(wire)
    budget = Budget(tmp_path / 'budget.sqlite', 32, None)
    transport = recheck.Transport(api, gate, tmp_path)
    result = asyncio.run(transport.request(RunConfig().judge_api, wire, budget,
                                           'logical', 'assessment', tmp_path / 'first'))
    assert result['choices'][0]['finish_reason'] == 'stop'
    assert [w['max_completion_tokens'] for w in wires] == [4096, 8192]
    assert wires[0]['messages'] == wires[1]['messages'] == before['messages']
    assert wire == before
    assert read_json(tmp_path / 'first/output_fallback.json')['partial_output_discarded'] is True
    assert budget.summary()['physical_requests'] == 2
    asyncio.run(transport.request(RunConfig().judge_api, wire, budget,
                                  'resume', 'assessment', tmp_path / 'resume'))
    assert len(wires) == budget.summary()['physical_requests'] == 2


def test_cache_cannot_replace_mandatory_review_with_assessment_response(tmp_path):
    stages = []

    async def api(config, wire, budget, key, stage, folder, **kwargs):
        stages.append(stage)
        result = response(text=stage)
        write_json(Path(folder) / 'response.json', result)
        return result

    budget = Budget(tmp_path / 'budget.sqlite', 32, None)
    transport = recheck.Transport(api, Gate(), tmp_path)
    for stage in ('assessment', 'review'):
        result = asyncio.run(transport.request(RunConfig().judge_api, request(), budget,
                                               stage, stage, tmp_path / stage))
        assert result['choices'][0]['message']['content'] == stage
    assert stages == ['assessment', 'review']


def test_hardlinks_only_sealed_inputs_and_keeps_mutable_result_separate(tmp_path):
    source, target = tmp_path / 'original', tmp_path / 'copy'
    source.mkdir()
    write_json(source / 'task.json', {'task_instance_id': 'task'})
    write_json(source / 'trajectory.json', {'events': []})
    write_json(source / 'blobs/answer.json', {'answer': 'evidence'})
    write_json(source / 'episode_result.json', {'evaluation_status': 'evaluation_error'})
    write_json(source / 'private_unsealed.json', {'must_not_copy': True})
    manifest = seal(source)
    original_result = file_hash(source / 'episode_result.json')
    provenance = recheck.sealed_link_copy(source, target)
    assert set(provenance['sealed_hardlinks']) == set(manifest)
    for name in manifest:
        assert (source / name).stat().st_ino == (target / name).stat().st_ino
    for name in ('episode_result.json', 'evidence_seal.json'):
        assert (source / name).stat().st_ino != (target / name).stat().st_ino
    assert not (target / 'private_unsealed.json').exists()
    write_json(target / 'episode_result.json', {'evaluation_status': 'valid', 'score': 1})
    assert file_hash(source / 'episode_result.json') == original_result
    assert recheck.r.verify_seal(source) == recheck.r.verify_seal(target) == manifest


@pytest.mark.parametrize('pilot_valid,expected_calls,passed', [(2, 4, False), (3, 7, True)])
def test_pilot_threshold_and_fixed_scope(tmp_path, monkeypatch, pilot_valid, expected_calls, passed):
    run, data = tmp_path / 'run', tmp_path / 'data'
    (run / 'judge_jobs').mkdir(parents=True)
    data.mkdir()
    monkeypatch.setattr(recheck, 'RUN', run)
    monkeypatch.setattr(recheck, 'DATA', data)
    monkeypatch.setattr(recheck, 'DRAIN', False)
    monkeypatch.setattr(recheck, 'SmoothGate', Gate)
    monkeypatch.setattr(recheck.shutil, 'disk_usage', lambda *_: SimpleNamespace(free=10 * 1024 ** 3))
    config = RunConfig(judge='api', judge_protocol='rubric_v2')
    monkeypatch.setenv(config.judge_api.api_key_env, 'offline-placeholder')

    async def forbidden_api(*args, **kwargs):
        raise AssertionError('Offline scheduler test cannot call a provider')

    runtime = SimpleNamespace(api_request=forbidden_api, config_hash=lambda _: 'config')
    monkeypatch.setattr(recheck.r, 'recovery_judge', lambda: runtime)
    calls = []

    async def assess(row, config, runtime):
        calls.append(row['key'])
        index = int(row['key'])
        return {'score': 1 if index < pilot_valid or index >= 4 else None}

    monkeypatch.setattr(recheck, 'assess', assess)
    cohort = [{'key': str(i), 'candidate': 'candidate', 'task_id': str(i)} for i in range(7)]
    plan = {'config': config.model_dump(mode='json'), 'cohort': cohort}
    write_json(run / 'plan.json', plan)
    # Unrelated records cannot expand the explicitly frozen cohort.
    write_json(run / 'judge_jobs/unrelated/recovery.json', {'score': 1})
    asyncio.run(recheck.drive(plan))
    assert calls == [str(i) for i in range(expected_calls)]
    assert read_json(run / 'pilot.json')['passed'] is passed
    status = read_json(run / 'status.json')
    assert status['total'] == 7
    assert status['completed'] == expected_calls
    assert status['stage'] == ('complete' if passed else 'needs_attention')
    if not passed:
        assert status['fatal'] == ['pilot_validity_below_3_of_4']


def test_completed_jobs_are_reused_and_do_not_expand_the_pilot(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    (run / 'judge_jobs').mkdir(parents=True)
    monkeypatch.setattr(recheck, 'RUN', run)
    monkeypatch.setattr(recheck, 'DATA', run)
    monkeypatch.setattr(recheck, 'DRAIN', False)
    monkeypatch.setattr(recheck, 'SmoothGate', Gate)
    monkeypatch.setattr(recheck.shutil, 'disk_usage', lambda *_: SimpleNamespace(free=10 * 1024 ** 3))
    config = RunConfig(judge='api', judge_protocol='rubric_v2')
    monkeypatch.setenv(config.judge_api.api_key_env, 'offline-placeholder')
    monkeypatch.setattr(recheck.r, 'recovery_judge', lambda: SimpleNamespace(api_request=None, config_hash=lambda _: 'config'))
    calls = []

    async def assess(row, config, runtime):
        calls.append(row['key'])
        return {'score': 0.3}

    monkeypatch.setattr(recheck, 'assess', assess)
    cohort = [{'key': str(i), 'candidate': 'candidate', 'task_id': str(i)} for i in range(7)]
    for i in range(5):
        write_json(run / f'judge_jobs/{i}/recovery.json', {'score': 1})
    plan = {'config': config.model_dump(mode='json'), 'cohort': cohort}
    write_json(run / 'plan.json', plan)
    asyncio.run(recheck.drive(plan))
    assert calls == ['5', '6']
    assert read_json(run / 'status.json')['completed'] == 7
