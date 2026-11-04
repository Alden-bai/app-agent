"""Offline regression tests for raw configuration evidence and saved final review."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import rubric_luna_batch03_resume_v3 as r
import test_rubric_luna_production_v1 as fixtures


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'rollout.jsonl'
        self.rows = [dict(type='session_meta', payload=dict(id='thread')),
                     dict(type='turn_context', payload=dict(model=r.p.MODEL, effort=r.p.EFFORT))]
        self.usage = dict(thread_id='thread', model=None, reasoning_effort=None, rollout_ref=str(self.path))

    def tearDown(self):
        self.temp.cleanup()

    def verify(self):
        self.path.write_text(''.join(json.dumps(row)+'\n' for row in self.rows))
        return r.verify_usage(self.usage, 'thread')

    def test_null_index_with_matching_raw_session(self):
        value = self.verify()
        self.assertTrue(value['actual_config_verified'])
        self.assertEqual(value['model'], r.p.MODEL)
        self.assertTrue(value['configuration_evidence']['used_missing_index_fallback'])
        self.assertIsNone(self.usage['model'])

    def test_matching_index_and_raw_context(self):
        self.usage.update(model=r.p.MODEL, reasoning_effort=r.p.EFFORT)
        self.assertTrue(self.verify()['actual_config_verified'])

    def test_explicit_index_conflicts_never_overridden(self):
        for key, value in [('model','another-model'),('reasoning_effort','low'),('model','')]:
            with self.subTest(key=key, value=value):
                self.usage.update(model=None, reasoning_effort=None)
                self.usage[key] = value
                self.assertFalse(self.verify()['actual_config_verified'])

    def test_missing_wrong_or_multiple_contexts_rejected(self):
        original = copy.deepcopy(self.rows)
        variants = [original[:1], original+[original[1]],
                    [original[0], dict(type='turn_context', payload=dict(model=r.p.MODEL, effort='low'))],
                    [dict(type='session_meta', payload=dict(id='another-thread')), original[1]],
                    [original[1]], [original[0], dict(type='turn_context', payload={})]]
        for rows in variants:
            self.rows = rows
            self.assertFalse(self.verify()['actual_config_verified'])

    def test_malformed_log_and_missing_log_do_not_pass(self):
        self.path.write_text('{bad json')
        with self.assertRaises(ValueError):
            r.verify_usage(self.usage, 'thread')
        self.usage.pop('rollout_ref')
        self.assertFalse(r.verify_usage(self.usage, 'thread')['actual_config_verified'])


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = fixtures.ProductionTests('test_full_four_stage_pipeline_and_terminal_resume_without_calls')
        await self.fixture.asyncSetUp()
        self.helper = self.fixture.helper
        self.manifest = self.helper.fixture()
        r.p.freeze(r.p.RUN/'manifest.json', self.manifest)
        await r.v2.Production(self.manifest, launcher=self.helper.launcher).run()
        self.folder = r.p.RUN/'tasks/test__0/round_1/judge'
        self.task_folder = self.folder.parent.parent
        self.good_usage = r.p.read(self.folder/'usage.json')
        self.good_state = r.p.read(self.task_folder/'status.json')
        usage = {**self.good_usage, 'actual_config_verified':False, 'model':None, 'reasoning_effort':None}
        r.p.snapshot(self.folder/'usage.json', usage)
        end = r.p.read(self.folder/'completed.json')
        end['status'] = 'call_failed'
        r.p.snapshot(self.folder/'completed.json', end)
        state = copy.deepcopy(self.good_state)
        state.pop('accepted_revision')
        state['status'] = 'call_failed'
        state['rounds'][-1] = dict(revision=1, status='call_failed', error='call_failed', failed_stage='judge')
        r.p.snapshot(self.task_folder/'status.json', state)
        for path in [self.folder/'output.json', self.folder.parent/'evaluation.json', r.p.RUN/'accepted/test__0.json']:
            path.unlink()
        self.good_usage.update(configuration_evidence=dict(verified=True), per_response_sum_matches_cumulative=True)
        usage['per_response_sum_matches_cumulative'] = True
        r.p.snapshot(self.folder/'usage.json', usage)
        self.usage_patch = patch.object(r, 'thread_usage', return_value=self.good_usage)
        self.usage_patch.start()

    async def asyncTearDown(self):
        self.usage_patch.stop()
        await self.fixture.asyncTearDown()

    async def test_repair_uses_cached_judgment_no_new_model_calls(self):
        raw = {f:r.p.digest(self.folder/f) for f in ('response.txt','events.jsonl','input.json')}
        plan = r.repair_plan(self.folder)
        self.assertIsNotNone(plan)
        self.assertEqual(r.p.read(self.task_folder/'status.json')['status'], 'call_failed')
        r.apply_metadata_repairs([plan])
        self.assertEqual(r.p.read(self.task_folder/'status.json'), self.good_state)
        self.assertEqual(len(self.helper.messages), 4)
        self.assertEqual({f:r.p.digest(self.folder/f) for f in raw}, raw)
        before = {str(f):r.p.digest(f) for f in r.p.RUN.rglob('*') if f.is_file()}
        r.apply_metadata_repairs([plan])
        self.assertEqual({str(f):r.p.digest(f) for f in r.p.RUN.rglob('*') if f.is_file()}, before)
        self.assertEqual(r.pending_repairs(), [])
        await r.Production(self.manifest, launcher=self.helper.launcher).run()
        self.assertEqual(len(self.helper.messages), 4)

    async def test_repair_preserves_rejection_instead_of_assuming_pass(self):
        judgment = r.p.read(self.folder/'response.txt')
        judgment['dimensions'][0]['score'] = 0
        r.p.snapshot(self.folder/'response.txt', judgment)
        end = r.p.read(self.folder/'completed.json')
        end['response_sha256'] = r.p.digest(self.folder/'response.txt')
        r.p.snapshot(self.folder/'completed.json', end)
        r.apply_metadata_repairs([r.repair_plan(self.folder)])
        self.assertEqual(r.p.read(self.task_folder/'status.json')['status'], 'rejected')
        self.assertFalse((r.p.RUN/'accepted/test__0.json').exists())

    async def test_response_mismatch_and_usage_change_block(self):
        changed = {**self.good_usage, 'actual_usage':{'total_tokens':999}}
        with patch.object(r, 'thread_usage', return_value=changed):
            with self.assertRaisesRegex(RuntimeError, 'usage_changed'):
                r.repair_plan(self.folder)
        (self.folder/'response.txt').write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'response_hash_mismatch'):
            r.repair_plan(self.folder)

    async def test_unverified_and_unknown_usage_not_recovered(self):
        for values in [dict(actual_config_verified=False), dict(usage_unknown=True), dict(tool_call_count=1)]:
            with patch.object(r, 'thread_usage', return_value={**self.good_usage, **values}):
                self.assertIsNone(r.repair_plan(self.folder))

    async def test_crash_replays_derived_records_without_model_calls(self):
        plan = r.repair_plan(self.folder)
        original = r.p.batch.atomic
        writes = 0
        def fail(path, content):
            nonlocal writes
            if Path(path).name == 'completed.json':
                writes += 1
                if writes == 1:
                    raise OSError('injected crash')
            return original(path, content)
        with patch.object(r.p.batch, 'atomic', side_effect=fail):
            with self.assertRaises(OSError):
                r.apply_metadata_repairs([plan])
        plans = r.pending_repairs()
        self.assertEqual(len(plans), 1)
        r.apply_metadata_repairs(plans)
        self.assertEqual(r.p.read(self.task_folder/'status.json')['status'], 'passed')
        self.assertEqual(len(self.helper.messages), 4)

    async def test_missing_configuration_stops_instead_of_business_failure(self):
        trial = r.Production(self.manifest)
        trial.gate = type('Gate', (), {'stop':AsyncMock()})()
        with patch.object(r.v2.Production, 'call', side_effect=r.p.legacy.base.CallFailure('call_failed')):
            with self.assertRaises(r.p.legacy.base.ModelUnavailable):
                await trial.call('judge', 1, 'test:0', {}, None)
        trial.gate.stop.assert_awaited_once_with('model_configuration')


if __name__ == '__main__':
    unittest.main()
