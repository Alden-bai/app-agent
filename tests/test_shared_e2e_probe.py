import asyncio
from types import SimpleNamespace
import pytest
from scripts.benchmark_shared_e2e import pick_tasks, take_batch, SharedActor, completed_queue_counts
from scripts.benchmark_shared_batch import batch_decode
from test_shared_batch_probe import policy


def test_stratified_sample_is_unique_and_deterministic():
    tasks = [SimpleNamespace(instruction='x' * i, task_instance_id=str(i)) for i in range(100)]
    picked = pick_tasks(tasks, 10, 123)
    assert [x.task_instance_id for x in picked] == [x.task_instance_id for x in pick_tasks(tasks, 10, 123)]
    assert sorted(int(x.task_instance_id) // 10 for x in picked) == list(range(10))


def test_completion_uses_final_commit_not_stale_monitor_snapshot(tmp_path):
    import sqlite3
    queue = tmp_path / 'queue.sqlite'
    with sqlite3.connect(queue) as db:
        db.execute('create table jobs(state text)')
        db.execute("insert into jobs values('claimed')")
    assert completed_queue_counts(queue, [None]) is None
    with sqlite3.connect(queue) as db: db.execute("update jobs set state='done'")
    assert completed_queue_counts(queue, [0]) == {'done': 1}
    with pytest.raises(RuntimeError): completed_queue_counts(queue, [1])


def test_batch_respects_padding_limits_budget_and_cancellation():
    async def check():
        loop = asyncio.get_running_loop()
        rows = [{'prompt': list(range(n)), 'max_tokens': limit, 'future': loop.create_future()}
                for n, limit in ((10, 4), (100, 4), (12, 4), (10, 8), (10, 4))]
        rows[-1]['future'].cancel()
        group, left = take_batch(rows, 8, 40)
        assert group == [rows[0], rows[2]]
        assert left == [rows[1], rows[3]]
    asyncio.run(check())


def test_finished_rows_are_delivered_before_long_rows_and_once():
    model, tokenizer = policy(); callbacks = []
    result = batch_decode(model, tokenizer, [[3], [4, 5]], max_tokens=5,
        on_complete=lambda i, ids: callbacks.append((i, ids, model.calls)))
    assert callbacks == [(0, [2], 1), (1, [7, 1], 2)]
    assert result['completion_ids'] == [[2], [7, 1]]


def test_cancelled_request_does_not_continue_generation():
    model, tokenizer = policy(); callbacks = []
    result = batch_decode(model, tokenizer, [[3], [4, 5]], max_tokens=5,
        on_complete=lambda i, ids: callbacks.append((i, ids)),
        cancelled=lambda i: i == 1)
    assert result['completion_ids'] == [[2], []]
    assert callbacks == [(0, [2])]
    assert model.calls == 1


def test_proxy_preserves_eos_usage_and_checkpoint_identity():
    from aiohttp import web
    async def check():
        async def handler(request):
            data = await request.json()
            assert data == {'prompt': [3, 4], 'max_tokens': 7, 'policy_version': 'frozen'}
            return web.json_response({'completion_ids': [6, 2], 'eos_token_ids': [2], 'policy_version': 'frozen'})
        app = web.Application(); app.router.add_post('/generate', handler)
        runner = web.AppRunner(app); await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0); await site.start()
        port = site._server.sockets[0].getsockname()[1]
        tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: [3, 4], decode=lambda *a, **kw: 'answer')
        actor = SharedActor(SimpleNamespace(port=port), tokenizer, policy_version='frozen',
            max_output_tokens=20, context_limit=100, greedy=True, record_policy=False)
        try:
            result = await actor.generate([], None, 1, output_tokens=7)
            assert result['finish_reason'] == 'stop'
            assert result['usage'] == {'prompt_tokens': 2, 'completion_tokens': 2, 'total_tokens': 4}
            assert result['model_version'] == 'frozen'
        finally: await runner.cleanup()
    asyncio.run(check())
