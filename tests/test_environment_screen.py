"""Admission must not confuse actor failure with a broken environment."""
import ast
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import unittest

from harness_awm.environment_screen import admission, sample_disposition
from harness_awm.io import digest, file_hash, write_json

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('screen',ROOT/'scripts/screen_bank_environments.py')
screen=importlib.util.module_from_spec(spec);spec.loader.exec_module(screen)
worker_spec=importlib.util.spec_from_file_location('screen_worker',ROOT/'scripts/screen_environment_worker.py')
worker=importlib.util.module_from_spec(worker_spec);worker_spec.loader.exec_module(worker)


class SamplingTests(unittest.TestCase):
    def test_documentation_id_is_bound_to_existing_seed(self):
        with sqlite3.connect(':memory:') as db:
            db.execute('CREATE TABLE addresses(id INTEGER PRIMARY KEY)')
            db.execute('INSERT INTO addresses VALUES(1)')
            db.execute('CREATE TABLE customers(address_id INTEGER)')
            provenance=[]
            self.assertEqual(worker.bind_references(42,{'type':'integer'},{},db,'address_id',provenance),1)
            self.assertEqual(provenance[-1]['example_replaced'],42)
            self.assertEqual(worker.bind_references(1,{'type':'integer'},{},db,'address_id',[]),1)

    def test_model_failure_remains_valid(self):
        self.assertEqual(sample_disposition(evaluation_valid=True,business_result='fail')['disposition'],'keep')

    def test_business_rejection_does_not_drop_sample(self):
        result=sample_disposition(faults=[{'kind':'business_rejection'}],evaluation_valid=True,business_result='fail')
        self.assertTrue(result['assign_reward'])

    def test_server_error_requires_attribution(self):
        result=sample_disposition(faults=[{'kind':'tool','confirmed':False}],evaluation_valid=True,business_result='fail')
        self.assertEqual(result['disposition'],'quarantine')

    def test_confirmed_impacting_fault_discards_sample(self):
        result=sample_disposition(faults=[dict(kind='database',confirmed=True,affects_outcome=True,evidence_refs=['traceback.json'])],evaluation_valid=True,business_result='pass')
        self.assertEqual(result['disposition'],'discard_sample');self.assertFalse(result['assign_reward'])

    def test_no_evidence_cannot_confirm_fault(self):
        result=sample_disposition(faults=[dict(kind='tool',confirmed=True,affects_outcome=True)],evaluation_valid=True,business_result='fail')
        self.assertEqual(result['disposition'],'quarantine')

    def test_unknown_write_outcome_blocks_reward_and_retry(self):
        result=sample_disposition(outcome_unknown=True,evaluation_valid=True,business_result='pass')
        self.assertFalse(result['assign_reward']);self.assertFalse(result['retry_write'])

    def test_judge_error_is_not_environment_failure(self):
        result=sample_disposition(faults=[dict(kind='judge',confirmed=True,affects_outcome=True,evidence_refs=['judge.json'])])
        self.assertEqual(result['disposition'],'quarantine')

    def test_routing_conflict_includes_successful_wrong_route(self):
        tree=ast.parse('@app.get("/items/{id}", operation_id="get_item")\ndef first(): pass\n@app.get("/items/search", operation_id="search")\ndef second(): pass')
        routes,errors=screen.static_routes(tree)
        self.assertEqual(errors[0]['kind'],'route_shadow')
        self.assertEqual(errors[0]['actual']['operation'],'get_item')

    def test_missing_runtime_not_passed(self):
        self.assertEqual(screen.summarize_scenario({'defects':[],'warnings':[]},None)['status'],'review')

    def test_4xx_not_confirmed_environment_defect(self):
        r=dict(mcp_ready=True,mcp_stopped=True,initial_unchanged=True,http={'probes':[dict(tool='x',status=404,result_ref='x.json')],'counts':{},'skipped':[]})
        self.assertEqual(screen.summarize_scenario({'defects':[],'warnings':[]},r)['status'],'screen_pass')

    def test_admission_rejects_changed_db_and_incomplete_screen(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);db=p/'db';db.write_bytes(b'original')
            env=p/'env.json';write_json(env,{'full_code':'test'})
            tasks=p/'tasks.jsonl';tasks.write_text(json.dumps(dict(task_id='s:0',sampling_eligible=True,environment_status='screen_pass',rubric_accepted=True,initial_sha256=file_hash(db),environment_sha256=digest({'full_code':'test'})))+'\n')
            manifest=p/'manifest.json';write_json(manifest,{})
            record=dict(complete=True,screen_manifest=str(manifest),screen_manifest_sha256=file_hash(manifest),tasks_ref=str(tasks),tasks_sha256=file_hash(tasks))
            ap=p/'admission.json';write_json(ap,record)
            self.assertTrue(admission(ap,'s:0',db,env)['allowed'])
            db.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'version_mismatch'):admission(ap,'s:0',db,env)
            record['complete']=False;write_json(ap,record)
            with self.assertRaisesRegex(ValueError,'incomplete'):admission(ap,'s:0',db,env)

    def test_http_write_probes_use_fresh_database_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);initial=p/'initial.db'
            with sqlite3.connect(initial) as db:
                db.execute('CREATE TABLE items(id INTEGER PRIMARY KEY, value INTEGER)')
                db.execute('INSERT INTO items VALUES (1,0)')
            original=str(p/'not_used.db')
            code='''from fastapi import FastAPI
import sqlite3
app=FastAPI()
DB=ORIGINAL
@app.post('/change',operation_id='change')
def change():
    with sqlite3.connect(DB) as db:
        db.execute('UPDATE items SET value=9')
    return {'ok':True}
@app.get('/state',operation_id='state')
def state():
    with sqlite3.connect(DB) as db:
        return {'value':db.execute('SELECT value FROM items').fetchone()[0]}
'''.replace('ORIGINAL',repr(original))
            (p/'server.py').write_text(code);write_json(p/'contract.json',{'api_spec':{'api_groups':[]}})
            write_json(p/'job.json',dict(work=str(p),original_db=original,timeout=10,scenario='fixture'))
            python=ROOT/'agent-world-model/.venv/bin/python'
            subprocess.run([str(python),str(ROOT/'scripts/screen_environment_worker.py'),str(p/'job.json')],check=True,capture_output=True,timeout=30)
            result=json.loads((p/'result.json').read_text())
            self.assertNotIn('setup_error',result)
            self.assertEqual(result['counts']['http_2xx'],2)
            read=json.loads(Path(result['probes'][1]['result_ref']).read_text())
            self.assertEqual(json.loads(read['response_body'])['value'],0)
            with sqlite3.connect(initial) as db:self.assertEqual(db.execute('SELECT value FROM items').fetchone()[0],0)


if __name__=='__main__':unittest.main()
