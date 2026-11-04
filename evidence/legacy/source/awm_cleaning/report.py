from __future__ import annotations
import collections,json
from pathlib import Path
from .common import *
from .certification import decide,GATES


from .evidence import validate_certificate

def rebuild(run):
 run=Path(run).resolve();m=read_json(run/'manifest.json');certs={}
 for p in sorted((run/'evidence/certificates').glob('*.json')):
  c=read_json(p)
  if c['ledger_id'] in certs:raise ValueError('Duplicate task certificate')
  if c.get('freeze_sha256')!=m['freeze_sha256']:raise ValueError('Certificate freeze mismatch')
  validate_certificate(run,m,c)
  c['decision'],c['reasons']=decide(c);certs[c['ledger_id']]=c
 dispositions=collections.defaultdict(list);approved=[]
 for t in m['tasks']:
  c=certs.get(t['ledger_id']);decision=c['decision'] if c else 'quarantine'
  row={**t,'decision':decision,'processing_status':c['processing_status'] if c else 'unprocessed','reasons':c['reasons'] if c else ['not_processed'],'certificate':str(run/'evidence/certificates'/f"{t['ledger_id'].replace(':','_')}.json") if c else None}
  if decision=='keep':
   row['verifier']={'mode':'sql_llm','config':m['freeze']['judge'],'config_sha256':digest(m['freeze']['judge']),'entry_sha256':t['asset_hashes']['gen_verifier.jsonl']};approved.append({'ledger_id':t['ledger_id'],**row['verifier']})
  dispositions[decision].append(row)
 for name in ('keep','reject','quarantine'):write_rows(run/f'tasks.{name}.jsonl',dispositions[name])
 write_rows(run/'task_certificates.jsonl',certs.values());write_rows(run/'approved_verifiers.jsonl',approved)
 envs=[read_json(p) for p in sorted((run/'evidence/environments').glob('*/check.json'))];write_rows(run/'environment_checks.jsonl',envs);write_rows(run/'tool_checks.jsonl',[{'scenario':e['scenario'],**t} for e in envs for t in e.get('tools',[])])
 api=rows(run/'evidence/api_requests.jsonl') if (run/'evidence/api_requests.jsonl').exists() else [];starts=[r for r in api if r['event']=='request_started'];ends=[r for r in api if r['event']=='request_finished'];tokens=collections.Counter()
 for r in ends:
  u=r.get('usage',{});
  for k in ('prompt_tokens','completion_tokens','total_tokens'):tokens[k]+=u.get(k,0) or 0
  details=u.get('prompt_tokens_details') or {};tokens['cached_tokens']+=details.get('cached_tokens',0) or 0
 processed=len(certs);pilot=[certs.get(i) for i in m['pilot100']];judged_cases=sum(bool(c.get('judge_slots')) for cert in certs.values() for c in cert.get('cases',[]));requests_per_case=len(starts)/judged_cases if judged_cases else None
 # Projection records assumptions and observed ranges, never currency pricing.
 observed_case_counts=[len(c.get('cases',[])) for c in certs.values() if c.get('cases')]
 projection={'status':'conditional_interval_only','whole_population':m['frame_tasks'],'requires_per_task_case_discovery':True,'assumption':'5 new slots per frozen case; transport requests per slot 1..3; unknown number of applicable cases and upstream defect exclusions','observed_task_case_count_range':[min(observed_case_counts),max(observed_case_counts)] if observed_case_counts else None,'requests_if_all_tasks_have_observed_case_range':[m['frame_tasks']*min(observed_case_counts)*5,m['frame_tasks']*max(observed_case_counts)*5*3] if observed_case_counts else None,'price_status':'not_estimated_no_verified_provider_price','not_confidence_interval':True}
 per_task=[]
 for task_id,cert in certs.items():
  marker=task_id.replace(':','_');task_starts=[r for r in starts if '/'+marker+'/' in r.get('slot','')];ids={r['request_id'] for r in task_starts};task_ends=[r for r in ends if r['request_id'] in ids];task_tokens=collections.Counter()
  for e in task_ends:
   for key in ('prompt_tokens','completion_tokens','total_tokens'):task_tokens[key]+=(e.get('usage',{}) or {}).get(key,0) or 0
  task_dir=run/'evidence/tasks'/marker
  per_task.append({'ledger_id':task_id,'cases':len(cert.get('cases',[])),'requests':len(task_starts),'retries':sum(r['attempt']>1 for r in task_starts),'tokens':dict(task_tokens),'api_elapsed_seconds':sum(r.get('elapsed_seconds',0) for r in task_ends),'storage_bytes':sum(p.stat().st_size for p in task_dir.rglob('*') if p.is_file())})
 token_samples=[(r.get('usage',{}) or {}).get('total_tokens',0) for r in ends if (r.get('usage',{}) or {}).get('total_tokens')]
 projection['observed_tokens_per_response_range']=[min(token_samples),max(token_samples)] if token_samples else None
 request_interval=projection['requests_if_all_tasks_have_observed_case_range']
 projection['tokens_if_all_requests_return_observed_range']=[request_interval[0]*min(token_samples),request_interval[1]*max(token_samples)] if token_samples and request_interval else None
 latencies=[r['elapsed_seconds'] for r in ends if r.get('elapsed_seconds') is not None]
 projection['api_seconds_if_all_requests_have_observed_range']=[request_interval[0]*min(latencies),request_interval[1]*max(latencies)] if request_interval and latencies else None
 storage_samples=[r['storage_bytes'] for r in per_task if r['cases']]
 projection['task_evidence_bytes_if_all_tasks_have_observed_full_case_storage_range']=[m['frame_tasks']*min(storage_samples),m['frame_tasks']*max(storage_samples)] if storage_samples else None
 projection['runtime_and_storage_limits']='API time interval excludes environment/tool checks, local computation and transport backoff. Task evidence storage excludes fixed source/runtime assets and depends on future cases.'
 events=rows(run/'evidence/execution_events.jsonl') if (run/'evidence/execution_events.jsonl').exists() else []
 costs={'recorded_execution_elapsed_seconds':sum(e.get('elapsed_seconds',0) for e in events),'per_task':per_task,'physical_api_requests':len(starts),'completed_requests':len(ends),'retries':sum(r['attempt']>1 for r in starts),'tokens':dict(tokens),'api_elapsed_seconds':sum(r.get('elapsed_seconds',0) for r in ends),'storage_bytes':sum(p.stat().st_size for p in run.rglob('*') if p.is_file()),'currency_cost':None,'projection':projection};save_json(run/'cost_report.json',costs)
 counts={k:len(dispositions[k]) for k in ('keep','reject','quarantine')};statuses=collections.Counter(c['processing_status'] for c in certs.values());pilot_checked=sum(c is not None for c in pilot);pilot_all_gates=sum(bool(c) and all(c['gates'].get(g) for g in GATES) for c in pilot)
 summary={'original_denominator':len(m['tasks']),'counts':counts,'processed_tasks':processed,'unprocessed_tasks':len(m['tasks'])-processed,'processing_statuses':dict(statuses),'pilot_inspection_complete':pilot_checked==100,'full_inspection_complete':processed==len(m['tasks']),'pilot_exact_original100_checked':pilot_checked,'pilot_all_gates_completed':pilot_all_gates,'pilot_certification_complete':all(c is not None and (c['decision'] in ('keep','reject')) for c in pilot),'full_cleaning_complete':processed==len(m['tasks']) and all(c['decision'] in ('keep','reject') for c in certs.values()),'formal_mode':'sql_llm'};save_json(run/'summary.json',summary)
 lines=['# AWM frozen release cleaning',f"Release `{m['revision']}`; official SQL+LLM mode only.",'',f"Original tasks = keep + reject + quarantine: {len(m['tasks'])} = {counts['keep']} + {counts['reject']} + {counts['quarantine']}.",f"Checked {processed}; unprocessed {len(m['tasks'])-processed}. Checked-insufficient and upstream-blocked are separate statuses: `{dict(statuses)}`.",'',f"Original pilot inspected: {pilot_checked}/100. All required certification gates passed: {pilot_all_gates}/100. Pilot certification complete: {summary['pilot_certification_complete']}. Full cleaning complete: {summary['full_cleaning_complete']}.",'','Inspection completion does not mean certification completion. Legacy labels and single historical judge outputs never upgrade a task. Ambiguous assets remain blocked. Model solve success rate is not a selection criterion.','',f"Requests {costs['physical_api_requests']}; retries {costs['retries']}; tokens `{dict(tokens)}`; bytes {costs['storage_bytes']}. Currency cost unavailable. See cost_report.json for conditional full-population interval and its assumptions.",'','## Remaining evidence gaps','']
 for reason,n in collections.Counter(r for c in certs.values() if c['decision']=='quarantine' for r in c['reasons']).most_common():lines.append(f'- {reason}: {n}')
 lines+=['','## Reviewed hard defects','']
 for c in certs.values():
  if c['decision']=='reject':lines.append(f"- {c['scenario']}:{c['task_idx']}: "+', '.join(c['reasons'])+f"; certificate `evidence/certificates/{c['ledger_id'].replace(':','_')}.json`.")
 lines+=['','## Resume and audit stages','','`prepare` freezes all original assets. `run --scope pilot100 --resume` reuses immutable completed evidence and resumes pending judge budget slots. `run --scope all --max-api-requests N --authorize-all-model-budget` separately authorizes a cumulative physical request ceiling. `report` rebuilds decisions; `publish` exports only keep records.','','Certificates record the reached gate; checked-insufficient tasks still require instantiated requirements, independent reviewed rules, legal tool replays, task-specific semantics, and complete five-slot SQL Judge evidence.']
 (run/'report.md').write_text('\n'.join(lines)+'\n');return summary

def publish(run):
 from .assets import validate
 run=Path(run).resolve();m=validate(run);rebuild(run);keep=rows(run/'tasks.keep.jsonl');keys=[r['ledger_id'] for r in keep]
 if len(keys)!=len(set(keys)):raise ValueError('Duplicate keep task instance')
 for row in keep:
  cert=read_json(row['certificate'])
  if decide(cert)[0]!='keep' or row.get('verifier',{}).get('mode')!='sql_llm':raise ValueError('Uncertified or mode-mismatched publication')
  if row['verifier']['config_sha256']!=digest(m['freeze']['judge']):raise ValueError('Judge version mismatch')
 public=[{k:r[k] for k in ('ledger_id','scenario','task_idx','task','task_sha256','verifier')} for r in keep]
 version=digest(public);dest=run/'release'/version;dest.mkdir(parents=True,exist_ok=True);write_rows(dest/'tasks.jsonl',public)
 # Public runtime contains only selected official records, exact modified AWM
 # source, initial state and public judge configuration. No audit rules/answers.
 scenarios={r['scenario'] for r in keep};bindings={(r['scenario'],r['task_idx']) for r in keep}
 for name in FILES:
  data=read_json(run/'assets'/(name+'.json'))
  selected=[r for r in data if (r.get('scenario',r.get('name')) in scenarios if name!='gen_verifier.jsonl' else (r.get('scenario'),r.get('task_idx')) in bindings)]
  if name=='gen_tasks.jsonl':selected=[{'scenario':r['scenario'],'task_idx':i,'task':text} for r in selected for i,text in enumerate(r['tasks']) if (r['scenario'],i) in bindings]
  write_rows(dest/'assets'/name,selected)
 import shutil
 for path in sorted((AWM/'awm').rglob('*.py')):
  target=dest/'runtime/awm'/path.relative_to(AWM/'awm');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
 save_json(dest/'judge_config.json',m['freeze']['judge']);save_json(dest/'runtime_dependencies.json',m['freeze']['runtime'])
 for scenario in scenarios:
  cert=next(read_json(r['certificate']) for r in keep if r['scenario']==scenario)
  env=read_json(run/'evidence/environments'/scenario/'check.json');backup(env['initial_databases'][0],dest/'initial'/f'{scenario}.db')
 loader=Path(__file__).with_name('public_loader.py');shutil.copy2(loader,dest/'load_release.py')
 (dest/'README.md').write_text('Official AWM SQL+LLM certified subset. Tasks: '+str(len(public))+'.\nRun Python load_release.py to verify every public artifact checksum. Use load_task(root, ledger_id, output_database) to obtain isolated original initial state, environment, verifier, and frozen Judge config. Install the exact frozen dependencies in runtime_dependencies.json; runtime/awm is the actual frozen AWM implementation. No credentials are exported.\n'+('This release is empty: no task passed every admission gate.\n' if not public else ''))
 checksums={str(p.relative_to(dest)):digest(p) for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='manifest.json'}
 immutable(dest/'manifest.json',{'release_id':m['revision'],'source_manifest_sha256':digest(run/'manifest.json'),'tasks_sha256':digest(dest/'tasks.jsonl'),'tasks':len(public),'mode':'sql_llm','private_answers_exported':False,'checksums':checksums});return {'path':str(dest),'tasks':len(public)}
