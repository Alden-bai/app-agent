from __future__ import annotations
import asyncio,json
from pathlib import Path
from types import SimpleNamespace
import pytest
from awm_cleaning import assets
from awm_cleaning.budget import Budget,BudgetExhausted
from awm_cleaning.certification import GATES,decide,five_slots_pass,coverage
from awm_cleaning.common import FILES,immutable,read_json,save_json
from awm_cleaning.judge import judge_slot
from awm_cleaning.provenance import resolve,ProvenanceError

def source(tmp,monkeypatch,names=('one','two'),duplicate=False):
 monkeypatch.setattr(assets,'ROOT',tmp)
 scenarios=[{'scenario':s,'tasks':['Task '+s]} for s in names]
 old=tmp/'audit_runs/20261021_seed20261021/manifest.json'
 save_json(old,{'tasks':[{'scenario':r['scenario'],'task_idx':0,'task':r['tasks'][0]} for r in scenarios]})
 src=tmp/'source';src.mkdir()
 for name in FILES:
  if name=='gen_tasks.jsonl':data=scenarios
  elif name=='gen_verifier.jsonl':
   data=[{'scenario':r['scenario'],'task_idx':0,'task':r['tasks'][0],'verification':{'code':'not executed'}} for r in scenarios]
   if duplicate:data.append(data[0])
  else:data=[{'scenario':s,'data':s} for s in names]
  (src/name).write_text(''.join(json.dumps(r)+'\n' for r in data))
 return src

def test_prepare_preserves_duplicate_binding_and_every_raw_record(tmp_path,monkeypatch):
 src=source(tmp_path,monkeypatch,duplicate=True);m=assets.prepare(tmp_path/'run',src,expected_environments=2,expected_tasks=2)
 assert len(m['tasks'])==2 and len({t['ledger_id'] for t in m['tasks']})==2
 assert m['tasks'][0]['asset_indices']['gen_verifier.jsonl']==[0,2]
 assert any(p['count']==2 for p in m['tasks'][0]['asset_problems'])
 assert all('pure_code' not in name for name in m['files'])
 assert len(read_json(tmp_path/'run/manifest.json')['pilot100'])==2
 assert assets.validate(tmp_path/'run')['frame_tasks']==2

def test_normalization_collision_blocks_every_colliding_task(tmp_path,monkeypatch):
 src=source(tmp_path,monkeypatch,names=('A B','a_b'));m=assets.prepare(tmp_path/'run',src,expected_environments=2,expected_tasks=2)
 assert len(m['tasks'])==2 and m['normalization_collisions']
 assert all(any(p['kind']=='normalized_name_collision' for p in t['asset_problems']) for t in m['tasks'])

def test_frame_discrepancy_has_raw_evidence(tmp_path,monkeypatch):
 src=source(tmp_path,monkeypatch)
 with pytest.raises(ValueError,match='Frame mismatch'):assets.prepare(tmp_path/'run',src)
 assert read_json(tmp_path/'run/preparation_failure.json')['actual_tasks']==2
 assert len((tmp_path/'run/partial_original_tasks.jsonl').read_text().splitlines())==2

def test_source_runtime_asset_and_dependency_hashes_block_drift(tmp_path,monkeypatch):
 src=source(tmp_path,monkeypatch);m=assets.prepare(tmp_path/'run',src,expected_environments=2,expected_tasks=2)
 monkeypatch.setattr(assets,'source_freeze',lambda:{'changed.py':'new'})
 with pytest.raises(ValueError,match='source changed'):assets.validate(tmp_path/'run')

def test_cumulative_budget_includes_retries_and_survives_restart(tmp_path):
 b=Budget(tmp_path,2,'pilot100');assert b.reserve('hash','case:0',1)==1;assert b.reserve('hash','case:0',2)==2
 with pytest.raises(BudgetExhausted):Budget(tmp_path,2,'pilot100').reserve('hash','case:1',1)
 assert Budget(tmp_path,3,'pilot100').reserve('hash','case:1',1)==3
 with pytest.raises(ValueError,match='All scope requires'):Budget(tmp_path,100,'all',False)
 with pytest.raises(ValueError,match='All scope requires'):Budget(tmp_path,None,'all',True)
 assert Budget(tmp_path,100,'all',True)

def test_no_budget_does_not_consume_frozen_judge_slot(tmp_path):
 folder=tmp_path/'slot';out=asyncio.run(judge_slot(tmp_path,{'task':'t'},{},[],folder,Budget(tmp_path,0,'pilot100'),0))
 assert out['status']=='pending_budget' and not (folder/'reserved.json').exists()

def test_interrupted_request_is_not_resampled(tmp_path):
 folder=tmp_path/'slot';save_json(folder/'reserved.json',{'slot':0})
 result=asyncio.run(judge_slot(tmp_path,{'task':'t'},{},[],folder,Budget(tmp_path,5,'pilot100'),0))
 assert result['status']=='interrupted_unknown'

def test_valid_result_immutable_on_resume(tmp_path):
 folder=tmp_path/'slot';record={'status':'valid','accepted':False,'slot':0,'classification':'agent_error'};immutable(folder/'result.json',record)
 result=asyncio.run(judge_slot(tmp_path,{'task':'t'},{},[],folder,Budget(tmp_path,0,'pilot100'),0));assert result==record
 with pytest.raises(ValueError,match='Immutable evidence'):immutable(folder/'result.json',{**record,'accepted':True})

def test_model_failure_not_a_selection_gate_and_unknown_never_kept():
 cert={'gates':{k:True for k in GATES},'model_success':False};assert decide(cert)[0]=='keep'
 cert['gates']['sql_evidence']=False;assert decide(cert)[0]=='quarantine'
 cert['defects']=[{'kind':'false_accept','evidence':['raw.json'],'review':{'status':'evidence_reviewed'}}];assert decide(cert)[0]=='reject'
 cert['defects'][0]['review']['status']='candidate';assert decide(cert)[0]=='quarantine'

def test_five_slots_rejects_missing_duplicate_invalid_and_wrong():
 case={'independent':{'correct':True},'judge_slots':[{'slot':i,'status':'valid','accepted':True} for i in range(5)]}
 assert five_slots_pass([case]);case['judge_slots'][4]['status']='invalid';assert not five_slots_pass([case])
 case['judge_slots'][4]={'slot':3,'status':'valid','accepted':True};assert not five_slots_pass([case])
 case['judge_slots'][4]={'slot':4,'status':'valid','accepted':False};assert not five_slots_pass([case])

def test_requirement_negative_must_effectively_violate_reachable_case():
 spec={'requirements':[{'id':'answer','quote':'give value'}]}
 cases=[{'reachable':True,'independent':{'status':'valid','correct':True,'checks':[{'requirement_id':'answer','passed':True}]}}]
 assert not coverage(spec,cases)['passed']
 cases.append({'reachable':False,'violates':['answer'],'independent':{'status':'valid','correct':False,'checks':[{'requirement_id':'answer','passed':False}]}})
 assert not coverage(spec,cases)['passed'];cases[-1]['reachable']=True;assert coverage(spec,cases)['passed']

def test_public_constants_prior_returns_and_computation_are_reproducible():
 log=[];constant={'$public':1,'citation':'contract:lower'};registry={'contract:lower':1}
 value={'$compute':'subtract','args':[constant,constant]}
 assert resolve(value,'task',[],log,public_contracts=registry)==0
 assert resolve({'$result':0,'path':['id']},'task',[{'id':42}],log)==42
 with pytest.raises(ProvenanceError):resolve({'$sql':'SELECT id'},'task',[],[])

@pytest.mark.parametrize('kind,expected_requests',[('transport',3),('auth',1),('malformed',1),('server_error',1),('truncation',1),('agent_error',1)])
def test_official_transport_and_semantic_outputs_have_distinct_retry_rules(tmp_path,monkeypatch,kind,expected_requests):
 import httpx,openai
 import awm_cleaning.judge as module
 from awm_audit.config import LLMConfig
 monkeypatch.setattr(module,'load_config',lambda:LLMConfig('https://example.invalid/v1','deepseek-flash','test-no-secret'))
 original_openai=openai.AsyncOpenAI;count=0
 class Response:
  def __init__(self,classification,finish='stop'):
   self.content='{' if classification=='malformed' else json.dumps({'classification':classification})
   self.choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))];self.finish=finish
  def model_dump(self,**kwargs):return {'choices':[{'message':{'content':self.content},'finish_reason':self.finish}],'usage':{'total_tokens':10}}
 async def create(**kwargs):
  nonlocal count;count+=1
  request=httpx.Request('POST','https://example.invalid/v1')
  if kind=='transport':raise openai.APIConnectionError(request=request)
  if kind=='auth':raise openai.AuthenticationError('no authorization',response=httpx.Response(401,request=request),body={})
  return Response('complete' if kind=='truncation' else kind,'length' if kind=='truncation' else 'stop')
 async def close():pass
 monkeypatch.setattr(openai,'AsyncOpenAI',lambda **kwargs:SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),close=close))
 async def no_sleep(_):pass
 monkeypatch.setattr(module.asyncio,'sleep',no_sleep)
 result=asyncio.run(module.judge_slot(tmp_path,{'task':'t'},{},[],tmp_path/'slot',Budget(tmp_path,5,'pilot100'),0))
 assert count==expected_requests
 assert result['status']==('valid' if kind=='agent_error' else 'invalid')
 assert result['accepted'] is (False if kind=='agent_error' else None)
 if kind=='auth':assert read_json(tmp_path/'evidence/api_circuit.json')['blocked']

def test_empty_publication_is_loadable_checksummed_and_excludes_private_answers(tmp_path,monkeypatch):
 from awm_cleaning import report
 from awm_cleaning.public_loader import verify_release
 manifest={'revision':'test','tasks':[],'frame_tasks':0,'pilot100':[],'freeze':{'judge':{'formal_mode':'sql_llm'},'runtime':{}},'freeze_sha256':'test'}
 save_json(tmp_path/'manifest.json',manifest)
 for name in FILES:save_json(tmp_path/'assets'/(name+'.json'),[])
 save_json(tmp_path/'evidence/private_answer.json',{'answer':'not-public'})
 monkeypatch.setattr(assets,'validate',lambda *a,**k:manifest)
 result=report.publish(tmp_path);release=Path(result['path']);assert result['tasks']==0
 assert verify_release(release)['tasks']==0
 assert not (release/'evidence').exists() and not list(release.rglob('task_spec.json'))
 assert 'empty' in (release/'README.md').read_text()
 (release/'tasks.jsonl').write_text('tampered')
 with pytest.raises(ValueError,match='checksum'):verify_release(release)
