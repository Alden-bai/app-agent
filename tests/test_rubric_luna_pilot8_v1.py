"""Bounded pilot offline checks: no ninth task, cached stages, restart scope."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import rubric_luna_batch03_pilot8_v1 as r
import test_rubric_luna_production_v1 as fixtures


class PilotTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = fixtures.ProductionTests('test_full_four_stage_pipeline_and_terminal_resume_without_calls')
        await self.fixture.asyncSetUp()
        self.helper = self.fixture.helper

    async def asyncTearDown(self):
        await self.fixture.asyncTearDown()

    async def test_scope_excludes_terminal_and_remains_fixed_on_restart(self):
        manifest = self.helper.fixture()
        task = manifest['tasks'][0]
        manifest['tasks'] = [{**task, 'task_id':f'test:{i}'} for i in range(11)]
        manifest['sample_size'] = 11
        r.p.freeze(r.p.RUN/'manifest.json', manifest)
        for i in [0,1]:
            r.p.freeze(r.p.RUN/f'tasks/test__{i}/status.json', dict(task_id=f'test:{i}', status='passed'))
        scope = r.select_scope(manifest)
        self.assertEqual(scope['task_ids'], [f'test:{i}' for i in range(2,10)])
        self.assertFalse((r.pilot_dir()/'scope.json').exists())
        r.p.freeze(r.pilot_dir()/'scope.json', scope)
        r.p.freeze(r.p.RUN/'tasks/test__2/status.json', dict(task_id='test:2', status='passed'))
        self.assertEqual(r.select_scope(manifest), scope)

    async def test_ninth_task_never_processed_and_resume_does_not_expand(self):
        manifest = self.helper.fixture()
        task = manifest['tasks'][0]
        manifest['tasks'] = [{**task, 'task_id':f'test:{i}'} for i in range(9)]
        manifest['sample_size'] = 9
        r.p.freeze(r.p.RUN/'manifest.json', manifest)
        selected = [f'test:{i}' for i in range(8)]
        calls = []
        async def process(task, sem):
            calls.append(task['task_id'])
            state = dict(task_id=task['task_id'], status='passed')
            r.p.freeze(r.p.RUN/'tasks'/task['task_id'].replace(':','__')/'status.json', state)
            return state
        with patch.object(r.p.legacy, 'RAW_PROCESS', side_effect=process), patch.object(r.Pilot, 'record', return_value={}):
            result = await r.Pilot(manifest, selected).run()
            self.assertEqual(set(calls), set(selected))
            self.assertEqual(result['new_terminal'], 8)
            self.assertEqual(result['stop_reason'], 'pilot_scope_complete')
            self.assertFalse((r.p.RUN/'tasks/test__8').exists())
            await r.Pilot(manifest, selected).run()
            self.assertEqual(len(calls), 8)
        progress = r.p.read(r.p.RUN/'progress.json')
        self.assertEqual(progress['frozen'], 9)
        self.assertEqual(progress['terminal'], 8)
        self.assertEqual(progress['statuses']['pending'], 1)

    async def test_actual_pipeline_cached_stages_and_terminal_not_resent(self):
        manifest = self.helper.fixture()
        r.p.freeze(r.p.RUN/'manifest.json', manifest)
        first = r.Pilot(manifest, ['test:0'], launcher=self.helper.launcher)
        result = await first.run()
        self.assertEqual(len(self.helper.messages), 4)
        self.assertEqual(result['new_terminal'], 1)
        await r.Pilot(manifest, ['test:0'], launcher=self.helper.launcher).run()
        self.assertEqual(len(self.helper.messages), 4)
        with self.assertRaisesRegex(RuntimeError, 'outside_pilot_scope'):
            await first.call('generate', 0, 'test:99', {}, None)
        with self.assertRaisesRegex(ValueError, 'scope_exceeded'):
            r.Pilot(manifest, [f'test:{i}' for i in range(9)])


if __name__=='__main__':
    unittest.main()
