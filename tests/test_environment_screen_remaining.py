"""Queue ownership, review gates, deterministic sampling and evidence integrity."""
import importlib.util
import asyncio
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('remaining', ROOT / 'scripts/environment_screen_remaining.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class RemainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.db = mod.db_open(self.run)
        self.addCleanup(self.db.close)
        self.db.executescript('''CREATE TABLE jobs(id TEXT PRIMARY KEY,case_id TEXT,kind TEXT,status TEXT,
          attempts INTEGER,result_path TEXT,error TEXT,started_at TEXT,finished_at TEXT);
          CREATE TABLE tasks(case_id TEXT PRIMARY KEY,task_id TEXT,scenario TEXT,status TEXT,
          first_ref TEXT,final_ref TEXT,reason TEXT);
          CREATE TABLE events(seq INTEGER PRIMARY KEY,at TEXT,job_id TEXT,event TEXT,detail TEXT);''')

    def add_task(self, cid, label='pass', scenario='a'):
        folder = self.run / cid
        folder.mkdir()
        path = folder / 'first.json'
        path.write_text(json.dumps(dict(case_id=cid, decision=label)))
        self.db.execute('INSERT INTO tasks VALUES(?,?,?,?,?,?,?)', (cid, scenario + ':' + cid, scenario, 'pending', str(path), None, None))
        mod.add_job(self.db, cid, 'initial')
        self.db.execute("UPDATE jobs SET status='completed' WHERE case_id=?", (cid,))
        self.db.commit()
        return folder

    def test_atomic_claim_not_duplicated_across_connections(self):
        mod.add_job(self.db, 'T1', 'initial')
        self.db.commit()
        a = mod.claim(self.db, ['initial'])
        with mod.db_open(self.run) as other:
            self.assertIsNone(mod.claim(other, ['initial']))
        self.assertEqual(a['id'], 'T1-initial')

    def test_exclude_requires_review_and_first_is_immutable(self):
        case = self.add_task('T1', 'exclude')
        job = dict(case_id='T1', kind='initial')
        mod.ingest(self.run, self.db, job, case, dict(case_id='T1', decision='exclude'))
        self.assertEqual(self.db.execute("SELECT status FROM tasks").fetchone()[0], 'pending')
        self.assertEqual(self.db.execute("SELECT kind FROM jobs WHERE status='pending'").fetchone()[0], 'exclude_review')
        self.assertTrue(self.db.execute('SELECT first_ref FROM tasks').fetchone()[0].endswith('decision.first_pass.json'))
        with self.assertRaises(FileExistsError):
            mod.ingest(self.run, self.db, job, case, dict(case_id='T1', decision='pass'))

    def test_first_artifact_survives_queue_publication_rollback(self):
        case=self.add_task('T1','exclude')
        job=dict(case_id='T1',kind='initial')
        decision=dict(case_id='T1',decision='exclude')
        mod.ingest(self.run,self.db,job,case,decision)
        self.db.rollback()
        sha=mod.file_hash(case/'decision.first_pass.json')
        mod.ingest(self.run,self.db,job,case,decision)
        self.db.commit()
        self.assertEqual(mod.file_hash(case/'decision.first_pass.json'),sha)
        self.assertEqual(self.db.execute("SELECT count(*) FROM jobs WHERE kind='exclude_review'").fetchone()[0],1)

    def test_disagreement_stays_blocked_until_master_adjudication(self):
        case = self.add_task('T1', 'exclude')
        mod.ingest(self.run, self.db, dict(case_id='T1', kind='exclude_review'), case,
                   dict(case_id='T1', decision='pass'))
        task = self.db.execute('SELECT * FROM tasks').fetchone()
        self.assertEqual(task['status'], 'blocked')
        self.assertEqual(task['reason'], 'requires_master_adjudication')
        self.assertIsNone(task['final_ref'])

    def test_stratified_sample_and_review_population_are_not_mixed(self):
        for i in range(21):
            self.add_task(f'A{i:02d}', 'pass', 'a')
        self.add_task('B00', 'pass', 'b')
        self.add_task('X00', 'exclude', 'a')
        self.assertTrue(mod.freeze_sample(self.run, self.db))
        data = mod.read_json(self.run / 'pass_sample.json')
        self.assertEqual(len(data['selected']), 3)
        self.assertEqual({r['scenario'] for r in data['selected']}, {'a', 'b'})
        self.assertFalse(any(r['case_id'] == 'X00' for r in data['selected']))
        self.assertEqual(self.db.execute("SELECT count(*) FROM tasks WHERE status='completed'").fetchone()[0], 19)
        self.assertFalse(mod.freeze_sample(self.run, self.db))

    def test_sampling_waits_for_all_initial_sessions(self):
        self.add_task('T1')
        mod.add_job(self.db, 'T2', 'initial')
        self.db.commit()
        self.assertFalse(mod.freeze_sample(self.run, self.db))

    def test_sampling_recovers_if_manifest_publication_is_interrupted(self):
        for i in range(3):
            self.add_task('T' + str(i))
        with patch.object(mod, 'write_json', side_effect=OSError('publication interrupted')):
            with self.assertRaises(OSError):
                mod.freeze_sample(self.run, self.db)
        with mod.db_open(self.run) as other:
            chosen = {r[0] for r in other.execute("SELECT case_id FROM jobs WHERE kind='pass_audit'")}
            self.assertEqual(len(chosen), 1)
            self.assertEqual(other.execute("SELECT count(*) FROM tasks WHERE status='completed'").fetchone()[0], 2)
        self.assertTrue(mod.freeze_sample(self.run, self.db))
        self.assertEqual({r['case_id'] for r in mod.read_json(self.run / 'pass_sample.json')['selected']}, chosen)

    def test_interrupted_claims_preserve_attempt_limit(self):
        self.add_task('T1')
        self.add_task('T2')
        self.db.execute("UPDATE jobs SET status='running',attempts=1 WHERE case_id='T1'")
        self.db.execute("UPDATE jobs SET status='running',attempts=3 WHERE case_id='T2'")
        self.db.commit()
        asyncio.run(mod.recover_claims(self.run, self.db))
        states = {r['case_id']:r['status'] for r in self.db.execute('SELECT * FROM jobs')}
        self.assertEqual(states, {'T1':'pending','T2':'blocked'})
        self.assertEqual(self.db.execute("SELECT status FROM tasks WHERE case_id='T2'").fetchone()[0], 'blocked')

    def test_completed_orphan_is_published_without_another_model_attempt(self):
        self.add_task('T1')
        self.db.execute("UPDATE jobs SET status='running',attempts=1 WHERE case_id='T1'")
        self.db.commit()
        case=self.run/'work/T1-initial/attempt-00';case.mkdir(parents=True)
        mod.write_json(case/'decision.json',dict(case_id='T1',decision='pass'))
        folder=self.run/'sessions/T1-initial/attempt-00';folder.mkdir(parents=True)
        mod.jsonl(folder/'events.jsonl',[dict(type='thread.started',thread_id='t'),dict(type='turn.completed')])
        with patch.object(mod,'usage',return_value=dict(actual_config_verified=True)), \
                patch.object(mod,'validate_decision',return_value=(dict(case_id='T1',decision='pass'),[])):
            asyncio.run(mod.recover_claims(self.run,self.db))
        row=self.db.execute("SELECT status,attempts FROM jobs WHERE case_id='T1'").fetchone()
        self.assertEqual(tuple(row),('completed',1))
        self.assertTrue((case/'decision.first_pass.json').exists())

    def test_probe_reexecutes_with_frozen_virtualenv(self):
        with patch.object(mod.sys, 'prefix', '/wrong-python'), patch.object(mod.sys, 'argv',
                ['runner', 'probe', '--case', str(self.run), '--plan', str(self.run/'p.json')]), \
                patch.object(mod.os, 'execv', side_effect=RuntimeError('reexec')) as execute:
            with self.assertRaisesRegex(RuntimeError, 'reexec'):
                mod.main()
        self.assertEqual(execute.call_args.args[0], str(mod.PYTHON))

    def test_malformed_audit_query_is_rejected_before_writes(self):
        with self.assertRaisesRegex(ValueError, 'query must be an object'):
            mod.validate_plan(dict(calls=[dict(tool='create', arguments={})], queries=['SELECT id FROM x']))
        mod.validate_plan(dict(calls=[dict(tool='create', arguments={})], queries=[dict(sql='SELECT id FROM x', params=[])]))

    def test_reveal_preserves_original_bytes_even_if_draft_is_amended(self):
        case = self.run / 'review'
        case.mkdir()
        original = b'{"case_id":"R1","decision":"exclude"}\n'
        (case/'independent.json').write_bytes(original)
        mod.write_json(case/'task.json',dict(case_id='R1'))
        mod.write_json(case/'review_target.json',dict(job_id='R1-regression'))
        source = self.run/'old.json';mod.write_json(source,dict(case_id='R1',decision='pass'))
        self.db.execute('INSERT INTO jobs(id,case_id,kind,status,attempts,result_path) VALUES(?,?,?,?,?,?)',
                        ('R1-regression','R1','regression','completed',1,str(source)))
        self.db.commit()
        with patch.object(mod,'validate_decision',return_value=({},[])):
            mod.reveal(self.run,case)
        (case/'independent.json').write_text('{"decision":"pass"}')
        self.assertEqual((case/'independent.frozen.json').read_bytes(),original)
        self.assertEqual(mod.file_hash(case/'independent.frozen.json'),mod.read_json(case/'independent.seal.json')['sha256'])
        with patch.object(mod,'validate_decision',return_value=({},[])):
            with self.assertRaises(FileExistsError):
                mod.reveal(self.run,case)

    def test_raw_review_bundle_includes_queries_but_excludes_judgments(self):
        source = self.run/'raw_source'
        mod.write_json(source/'evidence/query.json',dict(sql='SELECT id FROM x',rows=[[1]]))
        mod.write_json(source/'evidence/decision.json',dict(decision='exclude'))
        mod.write_json(source/'diagnostics/state.json',dict(decision='pass'))
        mod.write_json(source/'probes/p0/result.json',dict(setup_ok=True,calls=[]))
        mod.write_json(source/'decision.first_pass.json',dict(decision='exclude'))
        target=self.run/'blind_copy'
        mod.copy_raw_evidence(source,target)
        self.assertTrue((target/'evidence/query.json').exists())
        self.assertTrue((target/'probes/p0/result.json').exists())
        self.assertFalse((target/'evidence/decision.json').exists())
        self.assertFalse((target/'diagnostics/state.json').exists())
        self.assertFalse((target/'decision.first_pass.json').exists())

    def test_sql_limit_marks_unread_rows_and_is_readonly(self):
        with sqlite3.connect(':memory:') as db:
            db.execute('CREATE TABLE x(id INTEGER)')
            db.executemany('INSERT INTO x VALUES(?)', [(1,), (2,)])
            result = mod.sql_evidence(db, dict(sql='SELECT id FROM x', limit=1), [])
            self.assertTrue(result['truncated'])
            with self.assertRaises(sqlite3.OperationalError):
                mod.sql_evidence(db, dict(sql='DELETE FROM x'), [])

    def test_reference_escape_and_unsupported_pass_rejected(self):
        case = self.run / 'case'
        case.mkdir()
        mod.write_json(case / 'task.json', dict(case_id='T1'))
        (self.run / 'other.json').write_text('{}')
        data = dict(case_id='T1', decision='pass', summary='ok', verification_level='runtime',
                    requirements=[dict(requirement='must work', status='supported', evidence_refs=['../other.json'])])
        mod.write_json(case / 'decision.json', data)
        with self.assertRaisesRegex(ValueError, 'out-of-case'):
            mod.validate_decision(case)
        data['requirements'][0]['status'] = 'unresolved'
        mod.write_json(case / 'decision.json', data)
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            mod.validate_decision(case)

    def test_usage_counts_compaction_and_deduplicates_response_ids(self):
        rollout = self.run / 'rollout.jsonl'
        records = [dict(type='turn_context', payload=dict(model='gpt-6-luna', effort='high'))]
        cumulative = dict(input_tokens=30, cached_input_tokens=10, output_tokens=5, reasoning_output_tokens=2)
        for rid, inp, out in [('r1', 10, 2), ('r1', 10, 2), ('compaction-request', 20, 3)]:
            records.append(dict(type='token_usage_record', payload=dict(response_id=rid,
                usage=dict(input_tokens=inp, cached_input_tokens=inp//3, output_tokens=out,
                           reasoning_output_tokens=1), thread_token_usage=cumulative)))
        records.append(dict(type='compacted', payload={}))
        mod.jsonl(rollout, records)
        metadata = self.run / 'metadata.sqlite'
        real_connect = sqlite3.connect
        with real_connect(metadata) as db:
            db.execute('CREATE TABLE threads(id TEXT,model TEXT,reasoning_effort TEXT,rollout_path TEXT)')
            db.execute('INSERT INTO threads VALUES(?,?,?,?)', ('thread', 'gpt-6-luna', 'high', str(rollout)))
        with patch.object(mod.sqlite3, 'connect', side_effect=lambda *a, **k: real_connect(metadata)):
            result = mod.usage('thread')
        self.assertTrue(result['actual_config_verified'])
        self.assertEqual(result['request_count'], 2)
        self.assertEqual(result['usage']['total_tokens'], 35)
        self.assertEqual(result['usage']['reasoning_output_tokens'], 2)
        self.assertEqual(result['compaction_count'], 1)
        self.assertFalse(result['sum_matches_cumulative'])


if __name__ == '__main__':
    unittest.main()
