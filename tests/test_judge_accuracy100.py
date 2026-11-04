import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from harness_awm.backend import Budget, ScriptedBackend
from harness_awm.contracts import RunConfig
from harness_awm.io import file_hash, read_json
from harness_awm.runner import EpisodeRunner
from harness_support import LIST, call, finish, fixture

spec = importlib.util.spec_from_file_location('accuracy100', Path(__file__).resolve().parents[1] / 'scripts/judge_accuracy100.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class EvidenceTests(unittest.TestCase):
    def test_queries_cannot_mutate_or_attach_and_mark_incomplete_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.db'
            with sqlite3.connect(path) as db:
                db.execute('create table items(value integer)')
                db.executemany('insert into items values(?)', [(i,) for i in range(110)])
            original = file_hash(path)
            def query(sql): return study.sql_evidence(path, path, {'id': 'q1', 'sql': sql})
            for sql in ['DELETE FROM final.items', 'ATTACH DATABASE ":memory:" AS evil',
                        'PRAGMA final.user_version=2', 'SELECT load_extension("oops")']:
                self.assertFalse(query(sql)['valid'], sql)
            self.assertEqual(file_hash(path), original)
            self.assertTrue(query('select * from final.items')['truncated'])
            self.assertEqual(query('select count(*) from final.items')['rows'], [[110]])
            self.assertEqual(query('select count(*) from initial.items')['rows'], [[110]])

    def test_required_unknown_cannot_become_pass_or_failure(self):
        rubric = {'rubric': [{'id': 'R1', 'requirement_level': 'required'}]}
        value = {'action': 'finish', 'task_label': 'unknown', 'items': [
            {'id': 'R1', 'status': 'unknown', 'reason': 'missing evidence', 'evidence_ids': ['trajectory']}]}
        evidence = {'trajectory': {'valid': True}}
        study.validate_reference(value, rubric, evidence)
        for label in ['pass', 'fail']:
            with self.assertRaises(ValueError):
                study.validate_reference({**value, 'task_label': label}, rubric, evidence)
        bad = {**value, 'items': [{**value['items'][0], 'evidence_ids': ['invented']}]}
        with self.assertRaises(ValueError): study.validate_reference(bad, rubric, evidence)


class DeferredJudgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_execution_can_finish_without_dispatching_judge(self):
        import asyncio
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); task = fixture(root / 'input')
            config = RunConfig(judge='api', evaluator_timeout=20, startup_timeout=15, context_limit=100000)
            budget = Budget(root / 'budget.sqlite', 0, 0)
            evaluator = AsyncMock(side_effect=AssertionError('premature judge'))
            with patch('harness_awm.runner.evaluate_episode', evaluator):
                result = await EpisodeRunner(config, budget, asyncio.Semaphore(1)).run(
                    task, root / 'episode', 't0', ScriptedBackend([LIST, call('write_value', value=80), finish('80')]),
                    evaluate_after=False)
            evaluator.assert_not_called()
            self.assertEqual(result['termination_reason'], 'finish', result)
            self.assertEqual(result['terminal_state_status'], 'trusted')
            self.assertEqual(result['evaluation_status'], 'not_run')
            self.assertEqual(budget.summary()['physical_requests'], 0)
            self.assertTrue(read_json(root / 'episode/trajectory.json')['finished'])
