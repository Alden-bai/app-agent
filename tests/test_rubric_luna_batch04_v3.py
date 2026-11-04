"""Offline tests of the user-requested 12 -> 6 concurrency policy."""
import asyncio
import json
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import rubric_luna_batch04_production_v3 as v3
from rubric_luna_dispatch import DispatchPaused


class ConcurrencyTests(TestCase):
    def run_case(self, case):
        with patch.object(v3.previous, 'batch03_clear', return_value=True):
            asyncio.run(case())

    def test_twelve_permits_and_fallback_drains_without_cancelling(self):
        async def case():
            saved = []
            async def persist(event, state):
                saved.append((event, state))
            gate = v3.Batch04Gate(persist)
            async def launch(key):
                return await gate.launch(key, lambda: asyncio.sleep(0),
                    lambda: asyncio.sleep(0, result=type('P', (), {'pid': 1})()))
            for i in range(12):
                await launch(str(i))
            waiter = asyncio.create_task(launch('next'))
            await asyncio.sleep(.01)
            self.assertFalse(waiter.done())
            await gate.complete('0', infrastructure_failure=True)
            self.assertEqual(gate.limit, 6)
            self.assertEqual(len(gate.active), 11)
            self.assertEqual(saved[-1][1]['limit'], 6)
            for i in range(1, 6):
                await gate.complete(str(i))
            await asyncio.sleep(.01)
            self.assertFalse(waiter.done())
            await gate.complete('6')
            await asyncio.wait_for(waiter, 1)
            self.assertEqual(len(gate.active), 6)
            restarted = v3.Batch04Gate(persist, state={**gate.state(), 'active': {}})
            self.assertEqual(restarted.limit, 6)
            await restarted.observe()
            self.assertEqual(restarted.limit, 6)
        self.run_case(case)

    def test_rate_limit_keeps_backoff_and_exact_six_fallback(self):
        async def case():
            async def persist(*_):
                pass
            gate = v3.Batch04Gate(persist, clock=lambda: 1000)
            gate.active['a'] = {'launch_limit': 12}
            await gate.complete('a', rate_limited=True, retry_after=90)
            self.assertEqual(gate.limit, 6)
            self.assertGreaterEqual(gate.cooldown_until, 1090)
            gate.active['b'] = {'launch_limit': 6}
            await gate.complete('b', rate_limited=True)
            self.assertEqual(gate.limit, 6)
            self.assertEqual(gate.backoff_attempt, 2)
        self.run_case(case)

    def test_unknown_and_model_configuration_still_stop(self):
        async def case():
            async def persist(*_):
                pass
            for error in ('unresolved', 'model_bad'):
                gate = v3.Batch04Gate(persist)
                gate.active['a'] = {'launch_limit': 12}
                await gate.complete('a', **{error: True})
                self.assertEqual(gate.limit, 6)
                self.assertTrue(gate.stop_reason)
                if error == 'unresolved':
                    self.assertIn('a', gate.remote_holds)
                with self.assertRaises(DispatchPaused):
                    await gate.launch('b', lambda: asyncio.sleep(0), lambda: asyncio.sleep(0))
        self.run_case(case)

    def test_two_slow_windows_fall_directly_to_six(self):
        async def case():
            async def persist(*_):
                pass
            clock = [1000]
            gate = v3.Batch04Gate(persist, clock=lambda: clock[0])
            gate.baseline = {'groups': {}}
            for _ in range(2):
                clock[0] += 301
                gate.window_rows = [dict(completed_at=clock[0], infrastructure_failure=False,
                    rate_limited=False, success=True, model_seconds=30,
                    input_bytes=10000, stage='initial_generation') for _ in range(30)]
                with patch.object(v3, 'compare', return_value=dict(qualified=True,
                    p95_ratio=3, throughput_ratio=.8)):
                    await gate.observe()
            self.assertEqual(gate.limit, 6)
            self.assertEqual(gate.fallback_reason, 'two_qualified_slow_windows')
        self.run_case(case)

    def test_batch03_occupancy_still_caps_at_three(self):
        async def case():
            async def persist(*_):
                pass
            with patch.object(v3.previous, 'batch03_clear', return_value=False):
                gate = v3.Batch04Gate(persist)
                self.assertEqual(gate.limit, 3)
                await gate.observe()
                self.assertEqual(gate.limit, 3)
        self.run_case(case)

    def test_archived_unknown_usage_preserved_and_duplicate_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage = root/'tasks/s__1/round_0/generate'
            stage.mkdir(parents=True)
            live = dict(task_id='s:1', thread_id='live', actual_usage={'total_tokens': 10})
            (stage/'usage.json').write_text(json.dumps(live))
            archive = root/'interrupted_attempts/request1'
            archive.mkdir(parents=True)
            (archive/'identity.json').write_text(json.dumps(dict(task_id='s:1')))
            with patch.object(v3.previous.legacy, 'parse_events', return_value=('old', False, False)):
                rows = v3.actual_usage(root)
                self.assertEqual(set(rows), {'live', 'old'})
                self.assertTrue(rows['old']['usage_unknown'])
                self.assertIsNone(rows['old']['actual_usage'])
                (archive/'usage.json').write_text(json.dumps(live))
                with self.assertRaisesRegex(AssertionError, 'duplicate_usage_thread'):
                    v3.actual_usage(root)
