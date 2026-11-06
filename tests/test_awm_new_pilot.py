"""Offline contract tests for paid dispatch, source binding and staged admission."""
import asyncio
import ast
import copy
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import awm_new_pilot as pilot
from harness_awm.backend import Budget, BudgetExceeded
from harness_awm.io import file_hash, write_json


def test_freeze_rejects_changed_input(tmp_path):
    p=tmp_path/'x.json'
    pilot.freeze(p,{'version':1})
    pilot.freeze(p,{'version':1})
    with pytest.raises(ValueError,match='frozen_content_changed'):
        pilot.freeze(p,{'version':2})


def test_query_is_readonly_and_complete(tmp_path):
    p=tmp_path/'db.sqlite'
    with sqlite3.connect(p) as db:
        db.execute('create table t(v integer)')
        db.execute('insert into t values(1)')
    before=file_hash(p)
    assert pilot.query(p,'select v from t')==[[1]]
    for sql in ['delete from t','PRAGMA user_version=2',"ATTACH DATABASE ':memory:' AS other", "select load_extension('bad')"]:
        with pytest.raises(sqlite3.Error):
            pilot.query(p,sql)
    assert file_hash(p)==before


def test_control_named_rows_use_sql_columns_and_reject_missing_fields(tmp_path):
    p=tmp_path/'db.sqlite'
    with sqlite3.connect(p) as db:
        db.execute('create table t(a integer,b text)')
        db.execute("insert into t values(7,'x')")
    assert pilot.assertion_query(p,'select a,b from t',[{'b':'x','a':7}])==([[7,'x']],True)
    assert pilot.assertion_query(p,'select a,b from t',[[7,'x']])[1]
    assert not pilot.assertion_query(p,'select a,b from t',[[8,'x']])[1]
    with pytest.raises(ValueError,match='column_mismatch'):
        pilot.assertion_query(p,'select a,b from t',[{'a':7}])


def test_gate_counts_valid_unknown_and_does_not_require_actor_success():
    worlds=[dict(environment_passed=True,controls_passed=True,audit_passed=True,tasks=[
        dict(rubric_status='passed',evaluation=dict(evaluation_status='valid',business_label='fail'))
        for _ in range(10)]) for _ in range(2)]
    assert pilot.gate(worlds)['passed']
    worlds[0]['tasks'][0]['evaluation']['business_label']='unknown'
    worlds[0]['tasks'][1]['evaluation']['business_label']='unknown'
    assert pilot.gate(worlds)['passed']
    worlds[0]['tasks'][2]['evaluation']['business_label']='unknown'
    assert 'unknown_above_10_percent' in pilot.gate(worlds)['reasons']
    worlds[1]['tasks'][0]['rubric_status']='rejected'
    assert 'rubric_acceptance_incomplete' in pilot.gate(worlds)['reasons']


def test_request_cache_and_interruption_never_rebill(tmp_path):
    p=object.__new__(pilot.Pilot)
    p.run=tmp_path
    p.budget=Budget(tmp_path/'budget.sqlite',5,10000)
    p.sem=asyncio.Semaphore(3)
    p.gen=pilot.config().judge_api
    messages=[dict(role='user',content='test JSON')]
    async def exercise():
        key='test/completed'
        write_json(tmp_path/'requests'/key/'response.json',{'choices':[{'finish_reason':'stop','message':{'content':'{"ok":true}'}}]})
        with patch.object(pilot,'api_request',side_effect=AssertionError('must not send')):
            assert await p.request(key,messages,limit=100)=={'ok':True}
        p.budget.reserve('test/interrupted','test',messages,100)
        with pytest.raises(ValueError,match='interrupted_request_not_replayed'):
            await p.request('test/interrupted',messages,limit=100)
    asyncio.run(exercise())
    assert p.budget.summary()['physical_requests']==1


def test_shared_budget_accounts_unknown_and_embedding(tmp_path):
    b=Budget(tmp_path/'budget.sqlite',2,3000)
    rid=b.reserve('embedding/1','embedding',[{'role':'user','content':'x'}],0)
    b.finish(rid,{'total_tokens':10})
    rid=b.reserve('rubric/1','rubric',[{'role':'user','content':'x'}],100)
    b.finish(rid)
    with pytest.raises(BudgetExceeded):
        b.reserve('actor/1','actor',[],1)
    assert b.summary()['unknown_usage_requests']==1


def test_json_mode_fix_has_new_request_identity(tmp_path):
    p=object.__new__(pilot.Pilot)
    p.run=tmp_path
    p.budget=Budget(tmp_path/'budget.sqlite',5,10000)
    p.sem=asyncio.Semaphore(3)
    p.gen=pilot.config().judge_api
    captured=[]
    async def send(config,wire,budget,key,stage,dest):
        captured.append((wire,key))
        return {'choices':[{'finish_reason':'stop','message':{'content':'{"distinct":true}'}}]}
    async def exercise():
        with patch.object(pilot,'api_request',send):
            return await p.request('review/r0',[dict(role='system',content='Review the workflow.')],limit=100)
    assert asyncio.run(exercise())=={'distinct':True}
    wire,key=captured[0]
    assert key=='review/r0/json_contract_v1'
    assert 'JSON' in wire['messages'][0]['content']


def test_output_exhaustion_repairs_once_and_reuses_failed_response(tmp_path):
    p=object.__new__(pilot.Pilot)
    p.run=tmp_path
    p.budget=Budget(tmp_path/'budget.sqlite',5,100000)
    p.sem=asyncio.Semaphore(3)
    p.gen=pilot.config().judge_api
    write_json(tmp_path/'requests/assets/a/r0/response.json',
        {'choices':[{'finish_reason':'length','message':{'content':''}}]})
    pilot.freeze(tmp_path/'requests/assets/a/r0/identity.json',dict(messages=[dict(role='system',content='Return JSON')],
        model=pilot.MODEL,effort='high',code=False,limit=100))
    calls=[]
    async def send(config,wire,budget,key,stage,dest):
        calls.append((key,wire['max_completion_tokens']))
        return {'choices':[{'finish_reason':'length','message':{'content':''}}]}
    async def exercise():
        with patch.object(pilot,'api_request',send):
            with pytest.raises(ValueError,match='incomplete_generation'):
                await p.request('assets/a/r0',[dict(role='system',content='Return JSON')],limit=100)
    asyncio.run(exercise())
    assert calls==[('assets/a/r0/completion_repair',32768)]


def test_adapter_rejects_task_binding_mismatch(tmp_path):
    dbpath=tmp_path/'initial.db'
    with sqlite3.connect(dbpath) as db:
        db.execute('create table t(id integer primary key)')
    write_json(tmp_path/'environment.json',{'scenario':'test','full_code':'pass'})
    write_json(tmp_path/'verifiers/00.json',{'scenario':'test','task_idx':0,'task':'different task','verification':{'code':''}})
    with pytest.raises(ValueError,match='task_verifier_binding_mismatch'):
        pilot.task_spec({'data':str(tmp_path)},tmp_path,0,'original task')


def test_original_retains_schema_and_contract_verbatim():
    schema={'tables':[{'name':'t','ddl':'CREATE TABLE t(id INTEGER PRIMARY KEY);','indexes':[]}]}
    contract={'description':'Explicit 24 months; all rows returned.'}
    original=pilot.original_input('test:0','Create a term for 24 months.',contract,schema)
    assert original['sources'][1]['text']==pilot.canonical(contract)
    assert schema['tables'][0]['ddl'] in original['sources'][2]['text']
    _,pack=pilot.d1.base.prepare_input(original)
    assert pack['candidates']


def test_multiline_engine_normalization_preserves_ast():
    raw='engine = create_engine(\n    DATABASE_URL,\n    connect_args={"check_same_thread":False},\n)\nvalue = 42\n'
    fixed=pilot.normalize_engine_layout(raw)
    assert ast.dump(ast.parse(raw))==ast.dump(ast.parse(fixed))
    assert len(fixed.splitlines())==2
    assert pilot.normalize_engine_layout(fixed)==fixed


def test_context_preflight_preserves_source_and_bounds_output(tmp_path):
    p=object.__new__(pilot.Pilot)
    p.run=tmp_path
    p.budget=Budget(tmp_path/'budget.sqlite',5,300000)
    p.sem=asyncio.Semaphore(3)
    p.gen=pilot.config().judge_api
    messages=[{'role':'user','content':'JSON '+('x'*96000)}]
    calls=[]
    async def send(config,wire,budget,key,stage,dest):
        assert wire['messages']==messages
        assert pilot.fits_context(messages,wire['max_completion_tokens'],128000)
        calls.append(key)
        return {'choices':[{'finish_reason':'stop','message':{'content':'{}'}}]}
    async def exercise():
        with patch.object(pilot,'api_request',send):
            await p.request('review/r1/completion_repair',messages,limit=32768)
    asyncio.run(exercise())
    assert calls==['review/r1/completion_repair/context_fit_v1']


def test_audited_failed_gate_prevents_redispatch(tmp_path):
    p=object.__new__(pilot.Pilot)
    p.run=tmp_path
    write_json(tmp_path/'pilot_gate.audited.json',{'passed':False})
    with patch.object(pilot,'report',return_value={'stopped':True}), patch.object(pilot.Pilot,'scenarios',side_effect=AssertionError('must not dispatch')):
        assert asyncio.run(p.run_all())=={'stopped':True}
