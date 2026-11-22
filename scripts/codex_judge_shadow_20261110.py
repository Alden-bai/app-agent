"""Evidence-bound shadow-only Codex pilot; never writes production selections."""
import argparse,fcntl,json,sys
from pathlib import Path
ROOT=Path('/root/awm-agent-train');sys.path.insert(0,str(ROOT))
from harness_awm.io import read_json,write_json,file_hash,now
from harness_awm.judge_evidence import EvidenceStore,model_view
from harness_awm.judge_v2 import QueryAction,validate_assessment,aggregate
BASE=ROOT/'runs/codex_judge_shadow_20261110_v1'

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['view','query','finish']);p.add_argument('--case',default='case001');a=p.parse_args()
 case=(BASE/a.case).resolve()
 if case.parent!=BASE.resolve():raise ValueError('case path')
 with (case/'adapter.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  manifest=read_json(case/'manifest.json')
  for name,sha in manifest['immutable_hashes'].items():
   if file_hash(case/name)!=sha:raise ValueError('changed input:'+name)
  bundle=read_json(case/'input.json');settings=manifest['settings']
  store=EvidenceStore(case/'initial.db',case/'final.db',folder=case/'evidence',seconds=settings['query_seconds'],max_rows=settings['query_rows'],max_bytes=settings['query_bytes'])
  store.records={p.stem:read_json(p) for p in sorted((case/'evidence').glob('E*.json'))}
  state=read_json(case/'state.json')
  if a.mode=='view':
   print(json.dumps({'original':bundle['original'],'rubric':bundle['normalized_rubric'],'evidence':model_view(store.records,max_bytes=40000),'state':state},ensure_ascii=False));return
  action=json.load(sys.stdin)
  if a.mode=='query':
   parsed=QueryAction.model_validate(action)
   if state['query_rounds']>=12:raise ValueError('shadow_query_round_budget_exhausted')
   if len(parsed.queries)>settings['queries_per_round']:raise ValueError('query_batch_limit')
   known={x['id'] for x in bundle['normalized_rubric']['items']}
   if any(not set(q.affected_items)<=known for q in parsed.queries):raise ValueError('unknown_item')
   state['query_rounds']+=1;write_json(case/'state.json',state)
   write_json(case/'calls'/f"query_{state['query_rounds']:02d}.json",action)
   results=[store.call(q.tool,q.arguments,objective=q.objective,affected_items=q.affected_items) for q in parsed.queries]
   store.verify();print(json.dumps(results,ensure_ascii=False));return
  checked=validate_assessment(action,bundle['normalized_rubric'],store.records);result=aggregate(checked);store.verify()
  write_json(case/'assessment.json',action)
  write_json(case/'shadow_result.json',{'shadow_only':True,'eligible_for_model_selection':False,'eligible_for_training_reward':False,'assessment':checked,'decision':result,'model_requested':'gpt-6-luna','reasoning_effort_requested':'high','model_identity_attested':False,'provider_tokens':None,'provider_tokens_status':'not_exposed_by_collaboration_tool','at':now()})
  print(json.dumps({'saved':str(case/'shadow_result.json'),'decision':result},ensure_ascii=False))
if __name__=='__main__':main()
