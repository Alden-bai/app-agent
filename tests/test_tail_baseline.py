from concurrent.futures import ThreadPoolExecutor

import pytest

from scripts.tail_baseline import claim, finish, init_queue


def test_concurrent_workers_claim_each_task_once_and_fast_workers_take_more(tmp_path):
    path = tmp_path / 'queue.sqlite'
    init_queue(path, [{'task_id': str(i), 'split': 'test', 'output': str(tmp_path / str(i))} for i in range(39)])
    slow = claim(path, 0)
    def drain(owner):
        taken = []
        while job := claim(path, owner):
            taken.append(job['task_id'])
            finish(path, job['task_id'], owner, 'done')
        return taken
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(drain, range(1, 9)))
    ids = [ident for group in results for ident in group]
    assert len(ids) == len(set(ids)) == 38
    assert slow['task_id'] not in ids
    assert claim(path, 10) is None  # In-flight work is never stolen or retried.
    finish(path, slow['task_id'], 0, 'done')
    assert claim(path, 11) is None


def test_wrong_owner_and_failed_attempt_cannot_be_reclaimed(tmp_path):
    path = tmp_path / 'queue.sqlite'
    init_queue(path, [{'task_id': 'one', 'split': 'test', 'output': str(tmp_path / 'one')}])
    job = claim(path, 3)
    with pytest.raises(ValueError, match='not owned'):
        finish(path, job['task_id'], 4, 'done')
    finish(path, job['task_id'], 3, 'error')
    assert claim(path, 4) is None
