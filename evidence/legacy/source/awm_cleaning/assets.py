"""Full raw-record ledger; no sampled preparation or duplicate selection."""
from __future__ import annotations
import ast,collections,importlib.metadata,json,os,platform,re,sys,time
from pathlib import Path
from awm_audit.assets import normalize
from awm_audit.config import load_config
from awm_audit.network import open_remote
from .common import *

def prepare(run,source=None,revision=REVISION,expected_environments=1000,expected_tasks=10000):
 run=Path(run).resolve();src=Path(source).resolve() if source else run/'source';src.mkdir(parents=True,exist_ok=True)
 if revision!=REVISION:raise ValueError('Use a separately configured release for another revision')
 if (run/'manifest.json').exists():
  m=read_json(run/'manifest.json');validate(run,m);return m
 assets={};metadata={}
 for name in FILES:
  p=src/name
  if not p.exists():
   temp=p.with_suffix('.partial')
   with open_remote(f'https://huggingface.co/datasets/Snowflake/AgentWorldModel-1K/resolve/{revision}/{name}',120) as response,temp.open('wb') as out:
    while chunk:=response.read(1024*1024):out.write(chunk)
   temp.rename(p)
  data=rows(p);assets[name]=data;metadata[name]={'sha256':digest(p),'records':len(data),'bytes':p.stat().st_size,'path':str(p)}
  # Existing AWM instrumented runtime reads these exact unmodified records.
  save_json(run/'assets'/(name+'.json'),data)
 taskrows=assets['gen_tasks.jsonl'];total=sum(len(r['tasks']) for r in taskrows)
 if len(taskrows)!=expected_environments or total!=expected_tasks:
  save_json(run/'preparation_failure.json',{'kind':'frame_mismatch','actual_environments':len(taskrows),'actual_tasks':total,'expected_environments':expected_environments,'expected_tasks':expected_tasks,'files':metadata})
  write_rows(run/'partial_original_tasks.jsonl',[{'raw_scenario_row':i,'scenario':r['scenario'],'task_idx':j,'task':t} for i,r in enumerate(taskrows) for j,t in enumerate(r['tasks'])])
  raise ValueError(f'Frame mismatch {len(taskrows)} environments/{total} tasks')
 scenario_rows=collections.defaultdict(list)
 for i,r in enumerate(taskrows):scenario_rows[normalize(r['scenario'])].append((i,r['scenario']))
 collisions={k:v for k,v in scenario_rows.items() if len(v)!=1}
 by_file={}
 for name,data in assets.items():
  groups=collections.defaultdict(list)
  for i,r in enumerate(data):
   key=(normalize(r.get('scenario',r.get('name',''))),r.get('task_idx')) if name=='gen_verifier.jsonl' else normalize(r.get('scenario',r.get('name','')))
   groups[key].append(i)
  by_file[name]=groups
 tasks=[]
 for row_idx,r in enumerate(taskrows):
  norm=normalize(r['scenario'])
  for idx,text in enumerate(r['tasks']):
   errors=[];indices={};hashes={}
   if norm in collisions:errors.append({'kind':'normalized_name_collision','records':collisions[norm]})
   for name in FILES:
    key=(norm,idx) if name=='gen_verifier.jsonl' else norm
    matches=by_file[name][key];indices[name]=matches
    if len(matches)!=1:errors.append({'file':name,'kind':'record_count','count':len(matches)});continue
    item=assets[name][matches[0]];hashes[name]=digest(item)
    if name=='gen_verifier.jsonl' and item.get('task')!=text:errors.append({'file':name,'kind':'task_text_mismatch'})
   tasks.append({'ledger_id':f'row{row_idx}:task{idx}','raw_task_id':r.get('id',r['scenario'])+':'+str(idx),'scenario':r['scenario'],'task_idx':idx,'raw_scenario_row':row_idx,'normalized_scenario':norm,'task':text,'task_sha256':digest(text),'asset_indices':indices,'asset_hashes':hashes,'asset_problems':errors})
 old=read_json(ROOT/'audit_runs/20261021_seed20261021/manifest.json');pilot=[]
 for t in old['tasks']:
  found=[x for x in tasks if x['scenario']==t['scenario'] and x['task_idx']==t['task_idx'] and x['task']==t['task']]
  if len(found)!=1:raise ValueError('Original pilot no longer matches exact raw task')
  pilot.append(found[0]['ledger_id'])
 packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
 config={**load_config().public(),'temperature':1.0,'max_tokens':4096,'formal_mode':'sql_llm'}
 freeze={'source':source_freeze(),'runtime':{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'timezone':os.environ.get('TZ','system'),'system_timezone':list(time.tzname),'utc_offset_seconds':time.timezone,'clock_policy':{'kind':'recorded_wall_clock','temporal_tasks_require_additional_clock_control':True},'dependencies':dict(sorted(packages.items()))},'protocol':PROTOCOL,'protocol_document_sha256':digest(ROOT/'awm_cleaning_quality_plan_v2.md') if (ROOT/'awm_cleaning_quality_plan_v2.md').exists() else None,'judge':config}
 m={'schema_version':1,'created_at':now(),'revision':revision,'dataset':'Snowflake/AgentWorldModel-1K','tasks':tasks,'scenarios':[r['scenario'] for r in taskrows],'frame_environments':len(taskrows),'frame_tasks':len(tasks),'files':metadata,'source':str(src),'normalization_collisions':collisions,'pilot100':pilot,'old_manifest_sha256':digest(ROOT/'audit_runs/20261021_seed20261021/manifest.json'),'freeze':freeze,'freeze_sha256':digest(freeze),'assets_complete':True}
 save_json(run/'manifest.json',m);write_rows(run/'tasks.ledger.jsonl',tasks);save_json(run/'evidence/frozen_runtime.json',freeze)
 # Every record has a transparent initial disposition, explicitly unprocessed.
 for group in ('keep','reject'):write_rows(run/f'tasks.{group}.jsonl',[])
 write_rows(run/'tasks.quarantine.jsonl',[{**t,'decision':'quarantine','processing_status':'unprocessed','reasons':['not_processed']} for t in tasks])
 return m

def validate(run,m=None):
 run=Path(run);m=m or read_json(run/'manifest.json')
 for name,meta in m['files'].items():
  p=Path(meta['path'])
  if digest(p)!=meta['sha256']:raise ValueError('Frozen input changed: '+name)
  loaded=read_json(run/'assets'/(name+'.json'))
  # Verify normalized runtime mirrors are exactly original records.
  if digest(loaded)!=digest(rows(p)):raise ValueError('Runtime asset mirror changed: '+name)
 current_packages=dict(sorted({d.metadata['Name']:d.version for d in importlib.metadata.distributions()}.items()))
 if current_packages!=m['freeze']['runtime']['dependencies']:raise ValueError('Frozen runtime dependencies changed')
 if sys.version!=m['freeze']['runtime']['python'] or sys.executable!=m['freeze']['runtime']['executable']:raise ValueError('Frozen Python runtime changed')
 if source_freeze()!=m['freeze']['source']:raise ValueError('Frozen executable source changed; new release run required')
 if os.environ.get('TZ','system')!=m['freeze']['runtime']['timezone'] or list(time.tzname)!=m['freeze']['runtime']['system_timezone'] or time.timezone!=m['freeze']['runtime']['utc_offset_seconds']:raise ValueError('Frozen timezone configuration changed')
 if m['freeze'].get('protocol_document_sha256') and digest(ROOT/'awm_cleaning_quality_plan_v2.md')!=m['freeze']['protocol_document_sha256']:raise ValueError('Frozen protocol document changed')
 if digest(m['freeze'])!=m['freeze_sha256']:raise ValueError('Manifest freeze hash mismatch')
 if len(m['tasks'])!=m['frame_tasks'] or len({t['ledger_id'] for t in m['tasks']})!=len(m['tasks']):raise ValueError('Ledger duplicated or incomplete')
 config={**load_config().public(),'temperature':1.0,'max_tokens':4096,'formal_mode':'sql_llm'}
 if config!=m['freeze']['judge']:raise ValueError('Frozen Judge configuration changed')
 return m
