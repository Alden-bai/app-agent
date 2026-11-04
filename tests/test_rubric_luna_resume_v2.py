"""Offline entry recovery tests. Fake processes only; no model requests."""
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import rubric_luna_batch03_resume_v2 as r
import test_rubric_luna_production_v1 as fixtures


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run_patch = patch.object(r.p, 'RUN', self.root)
        self.run_patch.start()
        self.check_patch = patch.object(r, 'check_processes')
        self.check_patch.start()
        r.p.freeze(self.root/'manifest.json', dict(sample_size=3000))
        self.folder = self.root/'tasks/task__0/round_0/generate'
        self.identity = dict(run_id=r.p.RUN_ID, task_id='task:0', stage='initial_generation', revision=0)
        self.key = r.request_key(self.identity)
        r.p.freeze(self.folder/'identity.json', self.identity)
        r.p.freeze(self.folder/'events.jsonl', '{"type":"thread.started","thread_id":"old-thread"}\n', raw=True)
        r.p.freeze(self.folder/'started.json', dict(process=dict(pid=999, pgid=999)))
        r.p.snapshot(self.root/'scheduler_state.json', dict(stop_reason='unresolved_remote_request',
            active={self.key:dict(pid=999, pgid=999)}, remote_holds={}, limit=6))
        self.row = dict(folder=str(self.folder.relative_to(self.root)), identity=self.identity, request_key=self.key,
                       files={f.name:r.p.digest(f) for f in self.folder.iterdir()})
        self.approval = dict(run_id=r.p.RUN_ID, manifest_sha256=r.p.digest(self.root/'manifest.json'), requests=[self.row])
        self.approval_path = self.root/'approval.json'
        r.p.freeze(self.approval_path, self.approval)

    def tearDown(self):
        self.check_patch.stop()
        self.run_patch.stop()
        self.temp.cleanup()

    def hashes(self):
        return {str(f):r.p.digest(f) for f in self.root.rglob('*') if f.is_file()}

    def test_preflight_is_read_only_and_counts_remaining(self):
        for i in range(112):
            r.p.freeze(self.root/f'tasks/done__{i}/status.json', dict(status='passed'))
        before = self.hashes()
        plan = r.inspect(self.approval)
        self.assertTrue(plan['ready'])
        self.assertEqual(plan['remaining'], 2888)
        self.assertEqual(len(plan['retry_requests']), 1)
        self.assertEqual(self.hashes(), before)

    def test_archive_preserves_every_byte_and_unknown_usage(self):
        r.apply_recovery(r.inspect(self.approval), self.approval_path)
        self.assertFalse(self.folder.exists())
        self.assertTrue(r.matches(r.archive_path(self.row), self.row))
        self.assertTrue(r.archived_usage()[0]['usage_unknown'])
        self.assertEqual(r.archived_usage()[0]['thread_id'], 'old-thread')
        state = r.p.read(self.root/'scheduler_state.json')
        self.assertEqual(state['limit'], 3)
        self.assertFalse(state['active'])
        # A crash after archive/clear can safely re-enter without losing evidence.
        r.apply_recovery(r.inspect(self.approval), self.approval_path)
        self.assertEqual(len(r.archived_usage()), 1)

    def test_changed_request_and_unapproved_hold_block(self):
        r.p.freeze(self.folder/'stderr.txt', 'new evidence', raw=True)
        self.assertFalse(r.inspect(self.approval)['ready'])

    def test_approval_never_applies_to_retry(self):
        r.apply_recovery(r.inspect(self.approval), self.approval_path)
        r.p.freeze(self.folder/'identity.json', self.identity)
        r.p.freeze(self.folder/'events.jsonl', '{"type":"thread.started","thread_id":"new-thread"}\n', raw=True)
        self.assertFalse(r.inspect(self.approval)['ready'])
        self.assertEqual(len(r.archived_usage()), 1)

    def test_archive_crash_before_state_commit_is_recoverable(self):
        archive = r.archive_path(self.row)
        archive.parent.mkdir()
        self.folder.rename(archive)
        plan = r.inspect(self.approval)
        self.assertTrue(plan['ready'])
        r.apply_recovery(plan, self.approval_path)
        self.assertFalse(r.p.read(self.root/'scheduler_state.json')['active'])

    def test_terminal_task_cannot_be_retried(self):
        r.p.freeze(self.folder.parent.parent/'status.json', dict(status='passed'))
        with self.assertRaisesRegex(RuntimeError, 'cannot_retry_terminal'):
            r.inspect(self.approval)

    def test_live_group_blocks_even_with_approval(self):
        self.check_patch.stop()
        with patch.object(r.p, 'group_alive', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'still_alive'):
                r.inspect(self.approval)

    def test_report_includes_archive_once_and_restores_summary(self):
        r.apply_recovery(r.inspect(self.approval), self.approval_path)
        original = r.p.legacy.usage_summary
        def report(_):
            summary = r.p.legacy.usage_summary([])
            self.assertIs(r.p.legacy.usage_summary, original)
            return summary
        with patch.object(r.p, 'report', side_effect=report):
            result = r.report({})
        self.assertEqual(len(result['unknown_usage']), 1)
        self.assertIs(r.p.legacy.usage_summary, original)


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = fixtures.ProductionTests('test_full_four_stage_pipeline_and_terminal_resume_without_calls')
        await self.fixture.asyncSetUp()
        self.helper = self.fixture.helper

    async def asyncTearDown(self):
        await self.fixture.asyncTearDown()

    async def test_retry_then_terminal_resume_and_original_messages(self):
        manifest = self.helper.fixture()
        r.p.freeze(r.p.RUN/'manifest.json', manifest)
        await r.Production(manifest, launcher=self.helper.launcher).run()
        self.assertEqual(len(self.helper.messages), 4)
        await r.Production(manifest, launcher=self.helper.launcher).run()
        self.assertEqual(len(self.helper.messages), 4)
        self.assertEqual(r.p.read(r.p.RUN/'tasks/test__0/status.json')['status'], 'passed')

    async def test_timeout_waits_for_cleanup_and_stops(self):
        manifest = self.helper.fixture()
        trial = r.Production(manifest)
        trial.gate = type('Gate', (), {'stop':AsyncMock()})()
        cleaned = []
        async def stalled(*args, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(.01)
                cleaned.append(True)
        with patch.object(r.p.Production, 'call', stalled), patch.object(r, 'STAGE_TIMEOUT', .01):
            with self.assertRaises(r.p.DispatchPaused):
                await trial.call('generate', 0, 'test:0', {}, None)
        self.assertEqual(cleaned, [True])
        trial.gate.stop.assert_awaited_once_with('stage_timeout_requires_reconciliation')

    async def test_partial_task_reuses_generation_and_keeps_archived_usage(self):
        manifest = self.helper.fixture()
        r.p.freeze(r.p.RUN/'manifest.json', manifest)
        async def interrupted_launcher(*cmd, **kwargs):
            proc = await self.helper.launcher(*cmd, **kwargs)
            if Path(cmd[cmd.index('-o')+1]).parent.name=='judge':
                async def communicate(message):
                    kwargs['stdout'].write(b'{"type":"thread.started","thread_id":"lost-thread"}\n')
                    kwargs['stdout'].flush()
                    proc.returncode = 1
                    return None, None
                proc.communicate = communicate
            return proc
        def usage(tid, wall):
            return dict(thread_id=tid, usage_unknown=True, actual_usage=None) if tid=='lost-thread' else self.helper.usage(tid)
        with patch.object(r.p, 'measured_usage', side_effect=usage):
            await r.Production(manifest, launcher=interrupted_launcher).run()
        folder = r.p.RUN/'tasks/test__0/round_0/judge'
        identity = r.p.read(folder/'identity.json')
        row = dict(folder=str(folder.relative_to(r.p.RUN)), identity=identity, request_key=r.request_key(identity),
                   files={f.name:r.p.digest(f) for f in folder.iterdir() if f.is_file()})
        approval = dict(run_id=r.p.RUN_ID, manifest_sha256=r.p.digest(r.p.RUN/'manifest.json'), requests=[row])
        approval_path = r.p.RUN/'approval.json'
        r.p.freeze(approval_path, approval)
        original_generation = r.p.digest(folder.parent/'generate/response.txt')
        with patch.object(r, 'check_processes'):
            r.apply_recovery(r.inspect(approval), approval_path)
        trial = r.Production(manifest, launcher=self.helper.launcher)
        await trial.run()
        self.assertEqual([m['role'] for m in self.helper.messages], ['generate','judge','revise','judge'])
        self.assertEqual(r.p.digest(folder.parent/'generate/response.txt'), original_generation)
        self.assertEqual(r.p.read(folder.parent.parent/'status.json')['status'], 'passed')
        self.assertEqual(r.p.read(r.p.RUN/'progress.json')['unknown_usage_calls'], 1)
        self.assertTrue(r.matches(r.archive_path(row), row))


if __name__=='__main__':
    unittest.main()
