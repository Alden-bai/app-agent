import sqlite3

import pytest

from harness_awm.io import write_json
from harness_awm.training.selection import init_queue, claim, finish
from scripts.sft_eval_memory_recovery import cuda_oom, recovered_candidates


def test_retries_only_proven_cuda_oom_and_preserves_other_outcomes(tmp_path):
    attempts = [
        {'evaluation_valid': True, 'execution_integrity': 'valid', 'termination_reason': 'finish', 'error': None},
        {'evaluation_valid': True, 'execution_integrity': 'valid', 'termination_reason': 'protocol_error_limit', 'error': None},
        {'evaluation_valid': False, 'execution_integrity': 'valid', 'termination_reason': 'token_limit', 'error': None},
        {'evaluation_valid': False, 'execution_integrity': 'runtime_error', 'termination_reason': 'runtime_error',
         'error': 'OutOfMemoryError: CUDA out of memory. Tried to allocate 20 MiB'},
    ]
    candidates = []
    for index, result in enumerate(attempts):
        folder = tmp_path / 'original' / str(index)
        result['task_instance_id'] = str(index)
        write_json(folder / 'episode_result.json', result)
        write_json(folder / 'evidence_seal.json', {'synthetic': True})
        candidates.append({'task_instance_id': str(index), 'path': str(folder)})
    destination = tmp_path / 'recovery' / 'validation' / 'model'
    records, jobs, audit = recovered_candidates({'candidates': candidates}, destination)
    assert len(records) == 4 and len(jobs) == 1
    assert jobs[0][1] == '3'
    assert [r['path'] for r in records[:3]] == [r['path'] for r in candidates[:3]]
    assert records[3]['original_attempt'] == candidates[3]['path']
    assert [r['retry_cuda_oom'] for r in audit] == [False, False, False, True]
    assert not cuda_oom({**attempts[3], 'evaluation_valid': True})
    assert not cuda_oom({**attempts[3], 'error': 'environment disconnected'})


def test_retirement_trigger_preserves_owned_task_and_unclaimed_job(tmp_path):
    queue = tmp_path / 'queue.sqlite'
    init_queue(queue, [('m', 'first', 'out1'), ('m', 'second', 'out2')])
    owned = claim(queue, 18)
    with sqlite3.connect(queue) as db:
        db.execute("""CREATE TRIGGER retire BEFORE UPDATE OF state ON jobs
            WHEN OLD.state='pending' AND NEW.state='claimed' AND NEW.owner=18
            BEGIN SELECT RAISE(ABORT,'retired'); END""")
    finish(queue, owned['id'], 18, 'done')
    with pytest.raises(sqlite3.IntegrityError, match='retired'):
        claim(queue, 18)
    other = claim(queue, 0)
    assert other['task'] == 'second'
    finish(queue, other['id'], 0, 'done')
    with sqlite3.connect(queue) as db:
        assert db.execute("SELECT count(*) FROM jobs WHERE state='done'").fetchone()[0] == 2
