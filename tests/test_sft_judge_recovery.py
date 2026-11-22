import sqlite3
import pytest
from pathlib import Path
from harness_awm.io import write_json,seal,verify_seal,db_check
from scripts.sft_recovery_support import recovery_judge,physical_snapshot,copy_sealed_episode


def test_followup_resolvable_gap_is_bounded_and_preserves_failures():
    f=recovery_judge().should_followup
    d={'errors':['required_evidence_incomplete:R01']}
    assert f(d,True,1,1,12,0,False)
    assert not f(d,True,4,1,12,0,False)
    assert not f(d,True,1,1,12,2,False)
    assert not f(d,True,1,1,12,0,True)
    assert not f({'errors':[]},True,1,1,12,0,False)
    assert not f({'errors':['rubric_error']},True,1,1,12,0,False)


def test_post_action_logical_fk_violation_is_preserved_not_repaired(tmp_path):
    source=tmp_path/'working.db'
    with sqlite3.connect(source) as db:
        db.executescript('CREATE TABLE parent(id INTEGER PRIMARY KEY); CREATE TABLE child(id INTEGER PRIMARY KEY, pid INTEGER REFERENCES parent(id)); INSERT INTO child VALUES(1,99);')
    with pytest.raises(ValueError,match='foreign_keys'):db_check(source)
    target=tmp_path/'snapshot.db';result=physical_snapshot(source,target)
    assert result['physical_integrity']=='ok' and result['foreign_key_violations']
    with sqlite3.connect(target) as db:assert db.execute('SELECT pid FROM child').fetchone()[0]==99
    with pytest.raises(ValueError):physical_snapshot(source,target)


def test_copy_keeps_source_evidence_and_result_immutable(tmp_path):
    original=tmp_path/'original';copy=tmp_path/'copy'
    write_json(original/'trajectory.json',{'answer':'unchanged'})
    write_json(original/'episode_result.json',{'evaluation_valid':False})
    seal(original);before=verify_seal(original)
    audit=copy_sealed_episode(original,copy)
    write_json(copy/'episode_result.json',{'evaluation_valid':True})
    assert verify_seal(copy)==before==verify_seal(original)
    assert audit['original_result_sha256']
    assert 'false' in (original/'episode_result.json').read_text()

# Exercise the actual orchestration with a deterministic API substitute.
import asyncio
import copy
from test_judge_v2 import bundle,Responder,assessment,query
from harness_awm.backend import Budget
from harness_awm.contracts import RunConfig
from harness_awm.io import read_json


def run_recovery(bundle,tmp_path,monkeypatch,answers,rounds=12):
    runtime=recovery_judge();responder=Responder(answers)
    monkeypatch.setattr(runtime,'api_request',responder)
    config=RunConfig(judge='api',judge_protocol='rubric_v2',judge_v2={'query_rounds':rounds,'review_all_decisions':True})
    value=asyncio.run(runtime.evaluate_input(bundle,config,Budget(tmp_path/'b.sqlite',32,None),asyncio.Semaphore(1),tmp_path/'judgment'))
    return value,responder


def test_review_can_query_more_than_once_and_cache_first_valid(bundle,tmp_path,monkeypatch):
    gap=assessment(('unknown','pass'),refs=['E00002'],kind='evidence_gap')
    gap['items'][0]['unknown']['can_resolve_with_tools']=True
    result,responder=run_recovery(bundle,tmp_path,monkeypatch,[gap,query(),query(),assessment(refs=['E00008'])])
    assert result['evaluation_status']=='valid',result
    assert responder.calls==4
    again,responder=run_recovery(bundle,tmp_path,monkeypatch,[])
    assert again==result and responder.calls==0


def test_scope_expansion_does_not_reset_failed_objective_limit(bundle,tmp_path,monkeypatch):
    q1=query('SELECT absent FROM items');q2=copy.deepcopy(q1);q2['queries'][0]['affected_items']=['R1','Q1']
    valid=assessment(refs=['E00002'])
    result,_=run_recovery(bundle,tmp_path,monkeypatch,[q1,q2,q1,q2,valid,valid])
    assert result['evaluation_status']=='valid',result
    evidence=read_json(Path(result['attempt'])/'evidence.json')
    assert evidence['E00010']['error']=='objective_query_repair_limit'


def test_exhausted_review_cannot_turn_missing_evidence_into_success(bundle,tmp_path,monkeypatch):
    gap=assessment(('unknown','pass'),refs=['E00002'],kind='evidence_gap')
    gap['items'][0]['unknown']['can_resolve_with_tools']=True
    result,_=run_recovery(bundle,tmp_path,monkeypatch,[gap,*([query()]*4),gap])
    assert result['evaluation_status']=='evaluation_error',result
    assert 'query_round_budget_exhausted' in result['errors']


def test_overlay_only_valid_recovery_and_original_failures_retained(tmp_path):
    from scripts.sft_judge_recovery import overlay_repairs,write_manifest
    write_json(tmp_path/'candidates.json',{'model':{}})
    write_manifest(tmp_path,'model',{'candidates':[{'path':'valid_failed'},{'path':'repairable'},{'path':'unresolved'}]})
    overlay_repairs(tmp_path,[{'source_episode':'repairable','score':0,'episode':'fixed_failure'},
                            {'source_episode':'unresolved','score':None,'episode':'invalid_retry'}])
    assert [x['path'] for x in read_json(tmp_path/'model/manifest.json')['candidates']]==['valid_failed','fixed_failure','unresolved']


def test_adapter_keeps_bad_post_action_rows_and_strict_initial_check(tmp_path,monkeypatch):
    from harness_awm.environment import AWMEnvAdapter
    from scripts.sft_recovery_support import install_snapshot_fix
    old_audit,old_freeze=AWMEnvAdapter.audit,AWMEnvAdapter.freeze
    monkeypatch.setattr(AWMEnvAdapter,'audit',old_audit);monkeypatch.setattr(AWMEnvAdapter,'freeze',old_freeze)
    install_snapshot_fix()
    env=AWMEnvAdapter(None,None,tmp_path);env.runtime.mkdir();env.started=True
    with sqlite3.connect(env.runtime/'working.db') as db:
        db.executescript('CREATE TABLE p(id PRIMARY KEY); CREATE TABLE c(pid REFERENCES p(id)); INSERT INTO c VALUES(7);')
    assert env.audit('after')['foreign_key_violations']
    env.freeze()
    assert read_json(tmp_path/'audit/final_snapshot_integrity.json')['foreign_key_violations']
    with pytest.raises(ValueError,match='foreign_keys'):db_check(tmp_path/'final.db')


def test_format_hint_identifies_item_without_rewriting_proofs():
    runtime=recovery_judge()
    value=assessment(refs=['E00002']);value['items'][0]['satisfied']['established']=False
    before=copy.deepcopy(value)
    hints=runtime.validation_hints(value)
    assert [r['id'] for r in hints]==['R1']
    assert value==before


@pytest.mark.parametrize('ambiguous',[False,True])
def test_controller_selects_only_after_repair_and_holds_ambiguous_winner(tmp_path,monkeypatch,ambiguous):
    import scripts.sft_judge_recovery as driver
    source=tmp_path/'source';root=tmp_path/'recovery';source.mkdir();root.mkdir()
    monkeypatch.setattr(driver,'SOURCE',source);monkeypatch.setattr(driver,'OUTPUT',root)
    monkeypatch.setattr(driver,'ARCHIVE',tmp_path/'archive')
    checkpoints={'base':{},'a':{},'b':{}}
    plan={'sft_candidates':{'a':{},'b':{}},'minimum_coverage':.9,'selection_rule':'validation only'}
    monkeypatch.setattr(driver.selection,'checked_plan',lambda path:plan)
    events=[]
    class Tracker:
        url='offline-test'
        def __init__(self,*args,**kwargs):pass
        def log(self,values,step):pass
        def finish(self,**kwargs):pass
    monkeypatch.setattr(driver,'SwanLabTracker',Tracker)
    monkeypatch.setattr(driver,'monitor_repairs',lambda source,split,*args,**kwargs:events.append(('repair',split)) or [])
    monkeypatch.setattr(driver,'prepare_validation',lambda *args:checkpoints)
    monkeypatch.setattr(driver,'run_gpu_stage',lambda plan,split,tracker:events.append(('models',split)))
    monkeypatch.setattr(driver,'overlay_repairs',lambda *args:None)
    monkeypatch.setattr(driver,'collect_repairs',lambda:[])
    def reports(split,plan):
        result={}
        for name,succeeded,scored in [('base',0,100),('a',50,100),('b',45,90 if ambiguous else 100)]:
            result[name]={'evaluation_split':split,'budget':{},'evaluation':{'planned':100,'result_records':100,
                'confirmed_successes':succeeded,'scored':scored,'score_coverage':scored/100,'eval_reward':{'sum':succeeded}}}
        return result
    monkeypatch.setattr(driver,'final_reports',reports)
    def prepare(root,plan,candidates,split):
        assert read_json(root/'selection.json')['selected']=='a'
        assert list(candidates)==['base','a']
        events.append(('prepare',split))
    monkeypatch.setattr(driver.selection,'prepare_evaluation',prepare)
    if ambiguous:
        with pytest.raises(RuntimeError,match='could change winner'):driver.execute()
        assert not (root/'selection.json').exists()
        assert all(split!='test' for _,split in events)
    else:
        driver.execute()
        assert events==[('repair','validation'),('models','validation'),('repair','validation'),('prepare','test'),('models','test'),('repair','test')]
        assert read_json(root/'summary.json')['test_used_for_selection'] is False
    assert (source/'test/JUDGE_RECOVERY_PENDING.json').exists()
