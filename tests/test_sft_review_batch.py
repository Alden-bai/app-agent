import importlib.util
import asyncio
import contextlib
import json
import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('batch',Path(__file__).resolve().parents[1]/'scripts/sft_review_batch.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class BatchTests(unittest.TestCase):
    def test_fixed_model_effort_fresh_context_no_nested_agents(self):
        command=m.command()
        self.assertIn('gpt-6.1-terra',command);self.assertIn('model_reasoning_effort="high"',command)
        self.assertIn('--ephemeral',command);self.assertNotIn('resume',command);self.assertNotIn('fork',command)
        self.assertIn('multi_agent',command);self.assertEqual(m.LIMIT,12)
        self.assertNotIn('danger-full-access',command)

    def test_no_api_key_fallback_and_no_home_reassignment(self):
        with patch.dict(os.environ,{'OPENAI_API_KEY':'sensitive','CODEX_API_KEY':'sensitive','CODEX_HOME':'original'}):
            env=m.environment()
        self.assertNotIn('OPENAI_API_KEY',env);self.assertNotIn('CODEX_API_KEY',env)
        self.assertEqual(env['CODEX_HOME'],'original')

    def test_does_not_log_trajectory_readbacks(self):
        self.assertIsNone(m.compact_event({'type':'item.completed','item':{'type':'command_execution','output':'full trajectory'}}))
        self.assertIsNone(m.compact_event({'type':'item.completed','item':{'type':'reasoning','text':'reasoning'}}))
        self.assertEqual(m.compact_event({'type':'turn.completed','usage':{'input_tokens':123}})['usage']['input_tokens'],123)

    def test_blind_prompt_contains_only_assigned_item(self):
        text=m.prompt({'id':'t1','lane':'secondary','prompt':'prompt','material':'material','episode':'episode','output':'output'})
        self.assertNotIn('old_decision',text);self.assertIn('完整',text);self.assertIn('禁止',text)
        self.assertIn('sft_review_full.py',text);self.assertIn('output',text)

    def test_twelve_workers_claim_once_and_drain_without_cancelling(self):
        async def scenario():
            pending = [f't{i:04d}' for i in range(60)]
            claimed = []; active = set(); peak = 0
            gate = asyncio.Event(); filled = asyncio.Event()
            runner = m.Runner()
            def status():
                return {'lanes': {lane: {'pending': list(pending) if lane == 'primary' else []}
                                  for lane in ('primary','targeted','secondary')}}
            def claim(ident, lane, agent):
                pending.remove(ident); claimed.append(ident)
                return {'id': ident, 'lane': lane}
            async def execute(job):
                nonlocal peak
                self.assertNotIn(job['id'], active)
                active.add(job['id']); peak = max(peak, len(active))
                if len(active) == 12: filled.set()
                await gate.wait()
                active.remove(job['id'])
                return {'id': job['id'], 'decision': 'keep'}
            with patch.object(m, 'scheduler_lock', contextlib.nullcontext), \
                 patch.object(m.f, 'status', status), patch.object(m.f, 'claim', claim), \
                 patch.object(m.f, 'finalize'), patch.object(m, 'execute', execute), \
                 patch.object(runner, 'save'):
                workers = [asyncio.create_task(runner.worker()) for _ in range(12)]
                await asyncio.wait_for(filled.wait(), 2)
                runner.drain(); gate.set()
                await asyncio.gather(*workers)
            self.assertEqual(peak,12)
            self.assertEqual(len(claimed),len(set(claimed)))
            self.assertEqual(runner.completed,12)
            self.assertEqual(len(pending),48)
            self.assertFalse(active)
        asyncio.run(scenario())

    def test_concurrency_change_preserves_previous_config_and_rejects_model_change(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(m.f,'OUT',Path(tmp)):
            old=dict(m.CONFIG,max_concurrency=3)
            path=Path(tmp)/'batch_executor_config.json'
            m.f.write(path,old)
            m.configure_executor()
            self.assertEqual(json.loads(path.read_text())['max_concurrency'],12)
            saved=list((Path(tmp)/'executor_history').glob('*.json'))
            self.assertEqual(len(saved),1)
            self.assertEqual(json.loads(saved[0].read_text()),old)
            m.f.write(path,dict(old,model='different'))
            with self.assertRaises(ValueError):m.configure_executor()

    def test_scheduler_lock_excludes_another_process(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp, patch.object(m.f,'OUT',Path(tmp)):
            with m.scheduler_lock():
                result=subprocess.run(['python','-c',
                    'import fcntl,sys; f=open(sys.argv[1],"a"); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)',
                    str(Path(tmp)/'.scheduler.lock')],capture_output=True)
                self.assertNotEqual(result.returncode,0)
                self.assertIn(b'BlockingIOError',result.stderr)


if __name__=='__main__': unittest.main()
