"""Offline safety checks for the independent final batch; no paid model calls."""
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import rubric_luna_batch04 as batch
import rubric_luna_batch04_production_v1 as production
from rubric_luna_dispatch import DispatchPaused


class Batch04Tests(TestCase):
    def test_exact_complement_and_conflict(self):
        population = [f's:{i}' for i in range(10000)]
        frozen = set(population[:7000])
        selected = batch.select_remaining(population, frozen, frozen)
        self.assertEqual(selected, sorted(population[7000:]))
        self.assertEqual(len(selected), 3000)
        with self.assertRaisesRegex(RuntimeError, 'occupancy_conflict'):
            batch.select_remaining(population, frozen, frozen - {population[0]})
        with self.assertRaisesRegex(RuntimeError, 'occupancy_conflict'):
            batch.select_remaining(population, frozen, frozen | {population[7000]})

    def test_authoritative_event_replay_and_idempotent_commit(self):
        with TemporaryDirectory() as tmp:
            registry = Path(tmp)
            with patch.object(batch.legacy, 'REGISTRY', registry):
                first = dict(run_id='batch03', dataset_id='awm-dde80a0', task_id='s:1')
                second = dict(run_id=batch.RUN_ID, dataset_id='awm-dde80a0', task_id='s:2')
                with batch.legacy.locked(registry / '.lock'):
                    batch.legacy.commit('reserve03', tasks=[first])
                    batch.legacy.commit('reserve04', tasks=[second])
                    before = (registry / 'events.jsonl').read_bytes()
                    batch.legacy.commit('repeat04', tasks=[second])
                    self.assertEqual(before, (registry / 'events.jsonl').read_bytes())
                    (registry / 'tasks.jsonl').unlink()
                    batch.legacy.REGISTRY_CACHE.clear()
                    _, rows = batch.legacy.replay()
                    self.assertEqual({row['task_id'] for row in rows.values()}, {'s:1', 's:2'})

    def test_second_runner_lock_rejected(self):
        with TemporaryDirectory() as tmp:
            lock = Path(tmp) / '.runner.lock'
            with batch.legacy.locked(lock, blocking=False):
                child = subprocess.run([sys.executable, '-c',
                    'import fcntl,sys; f=open(sys.argv[1],"a"); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)',
                    str(lock)], capture_output=True, text=True)
                self.assertNotEqual(child.returncode, 0)
            self.assertTrue(lock.exists())

    def test_parallel_cap_and_stop_prevent_fourth_launch(self):
        async def case():
            async def persist(*_):
                pass
            with patch.object(production, 'batch03_clear', return_value=False):
                gate = production.Batch04Gate(persist)
                self.assertEqual(gate.limit, 3)
                async def launch(key):
                    return await gate.launch(key, lambda: asyncio.sleep(0),
                                             lambda: asyncio.sleep(0, result=type('P', (), {'pid': 10})()))
                first = await asyncio.gather(*(launch(str(i)) for i in range(3)))
                self.assertEqual(len(first), 3)
                fourth = asyncio.create_task(launch('fourth'))
                await asyncio.sleep(.01)
                self.assertFalse(fourth.done())
                self.assertEqual(gate.peak, 3)
                await gate.observe()
                self.assertEqual(gate.limit, 3)
                await gate.stop('user_stop')
                with self.assertRaises(DispatchPaused):
                    await fourth
                self.assertNotIn('fourth', gate.active)
        asyncio.run(case())

    def test_verified_raw_configuration_and_unknown_recovery(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout = root / 'raw.jsonl'
            rollout.write_text('\n'.join(json.dumps(row) for row in [
                dict(type='session_meta', payload=dict(id='thread-a')),
                dict(type='turn_context', payload=dict(model=batch.MODEL, effort=batch.EFFORT))]) + '\n')
            usage = dict(thread_id='thread-a', rollout_ref=str(rollout), model=None,
                         reasoning_effort=None, actual_usage=None, usage_unknown=True)
            self.assertTrue(production.verify_usage(usage, 'thread-a')['actual_config_verified'])
            self.assertFalse(production.verify_usage({**usage, 'model': 'other'}, 'thread-a')['actual_config_verified'])
            identity = dict(stage='initial_generation')
            folder = root / 'stage'
            folder.mkdir()
            batch.freeze(folder / 'identity.json', identity)
            runner = production.Production(dict(tasks=[]))
            async def case():
                with self.assertRaises(DispatchPaused):
                    await runner.recover(folder, identity)
                batch.freeze(folder / 'response.txt', '{"ok":true}', raw=True)
                batch.freeze(folder / 'usage.json', dict(actual_config_verified=True))
                batch.freeze(folder / 'completed.json', dict(identity=identity,status='completed',
                    response_sha256=batch.digest(folder / 'response.txt')))
                self.assertEqual(await runner.recover(folder, identity), '{"ok":true}')
            asyncio.run(case())

    def test_all_four_stage_outputs_are_reused_without_launch(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = production.Production(dict(tasks=[]),launcher=lambda *args: self.fail('unexpected launch'))
            async def case():
                for stage in ('generate_0','judge_0','revise_1','judge_1'):
                    folder = root / stage
                    folder.mkdir()
                    identity = dict(run_id=batch.RUN_ID,stage=stage)
                    response = '{"stage":"'+stage+'"}'
                    batch.freeze(folder/'identity.json',identity)
                    batch.freeze(folder/'response.txt',response,raw=True)
                    batch.freeze(folder/'usage.json',dict(actual_config_verified=True))
                    batch.freeze(folder/'completed.json',dict(identity=identity,status='completed',
                        response_sha256=batch.digest(folder/'response.txt')))
                    self.assertEqual(await runner.recover(folder,identity),response)
            asyncio.run(case())

    def test_usage_deduplication(self):
        values = dict(input_tokens=10,cached_input_tokens=2,output_tokens=5,
                      reasoning_output_tokens=1,total_tokens=15)
        row = dict(thread_id='a',stage='initial_generation',actual_usage=values,
                   response_usage={'r1':values},usage_unknown=False)
        result = batch.legacy.usage_summary([row])
        self.assertEqual(result['totals']['total_tokens'], 15)
        self.assertEqual(result['totals']['noncached_input_tokens'], 8)
        with self.assertRaisesRegex(ValueError, 'duplicate_thread_usage'):
            batch.legacy.usage_summary([row,row])
        with self.assertRaises(AssertionError):
            batch.legacy.usage_summary([row,{**row,'thread_id':'b'}])
