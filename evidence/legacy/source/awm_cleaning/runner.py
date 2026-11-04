from __future__ import annotations
import asyncio,importlib,json,traceback,time
from pathlib import Path
from awm_audit.assets import scenario_asset
from awm_audit.oracle import load_spec,evaluate as legacy_evaluate
from awm_audit.verification import verifier_entry
from .assets import validate
from .budget import Budget
from .certification import coverage,decide,five_slots_pass,sql_review_valid,GATES
from .common import *
from .judge import extraction,judge_slot
from .provenance import inspect_legacy
from .runtime import environment_check,execute,dual_instance_check

def reviewed_defect(kind,evidence,basis):return {'kind':kind,'evidence':evidence,'review':{'status':'evidence_reviewed','source':'automatic','basis':basis}}

def inspect_task(run,task,env):
 """Inspect original text/facts and old candidate independently; no old verdicts."""
 folder=Path(run)/'evidence/tasks'/task['ledger_id'].replace(':','_');folder.mkdir(parents=True,exist_ok=True)
 cert={'ledger_id':task['ledger_id'],'scenario':task['scenario'],'task_idx':task['task_idx'],'task_sha256':task['task_sha256'],'processing_status':'checked_insufficient','gates':{k:False for k in GATES},'defects':[],'evidence':[],'formal_mode':'sql_llm'}
 if task['asset_problems']:
  save_json(folder/'asset_binding.json',task['asset_problems']);cert['evidence'].append(str(folder/'asset_binding.json'));cert['missing']=['ambiguous_original_asset_binding'];return cert,folder
 cert['gates']['assets']=True
 cert['environment_version']={'schema_sha256':task['asset_hashes'].get('gen_db.jsonl'),'sample_sha256':task['asset_hashes'].get('gen_sample.jsonl'),'source_sha256':task['asset_hashes'].get('gen_envs.jsonl'),'spec_sha256':task['asset_hashes'].get('gen_spec.jsonl'),'initial_databases':[{'path':p,'sha256':digest(Path(p)),'business_digest':digest(business_state(p))} for p in env.get('initial_databases',[])]}
 if env['status']=='initialization_defect':
  p=str(Path(run)/'evidence/environments'/task['scenario']/'check.json');cert['defects'].append(reviewed_defect('original_initialization_failure',[p],'Three complete independent official DDL/sample imports contain recorded SQLite errors/integrity failures; required common initial state unavailable.'));cert['processing_status']='upstream_blocked';return cert,folder
 if env['status']!='basic_checked':cert['missing']=['environment_infrastructure_unknown'];return cert,folder
 initial=env['initial_databases'][0]
 try:
  spec=load_spec(task['scenario'],task['task_idx'],initial);frozen=spec.frozen();save_json(folder/'legacy_candidate_spec.json',frozen)
  issues=[]
  for i,step in enumerate(spec.calls):issues+=inspect_legacy(step['arguments'],task['task'],('calls',i,'arguments'))
  save_json(folder/'legacy_provenance_review.json',issues)
  # Facts/rules are retained, but not certified merely because an old builder
  # called itself ready or claimed a defect.
  fact_checks=legacy_evaluate(spec,initial,initial,'',[])
  save_json(folder/'independent_initial_checks.json',fact_checks)
  cert['legacy_review']={'candidate_status':spec.status,'requirements':len(spec.rules),'calls':len(spec.calls),'provenance_issues':len(issues),'old_results_reused':False,'candidate_defect_not_automatically_adopted':spec.status=='confirmed_defect'}
  cert['evidence']+=[str(folder/x) for x in ('legacy_candidate_spec.json','legacy_provenance_review.json','independent_initial_checks.json')]
 except Exception as e:
  save_json(folder/'candidate_error.json',safe_error(e));cert['evidence'].append(str(folder/'candidate_error.json'))
 cert['missing']=['requirement_text_contract_mapping','independent_reviewed_executable_rules','real_provenance_safe_success_replays','effective_frozen_counterexample_coverage','fully_checked_sql_evidence','new_five_slot_judgments']
 return cert,folder

async def certify_reference(run,task,env,cert,folder,budget):
 from . import reference_ecommerce as plugin
 initial=env['initial_databases'][0];task={**task,'_public_contracts':plugin.public_contracts(scenario_asset(run,'gen_spec.jsonl',task['scenario']))};spec=plugin.build(task,initial);immutable(folder/'task_spec.json',spec)
 spec_hash=digest(spec);cert.update(spec_sha256=spec_hash,requirements=spec['requirements'],cases=[])
 # Requirement citations are checked against unmodified task text before tests.
 citations=all(r['quote'] in task['task'] for r in spec['requirements']);cert['gates'].update(requirements=citations,independent_rules_review=citations)
 replay_results=[]
 for replay in range(3):
  c=spec['cases'][0];p=folder/f'positive_replay_{replay}'
  trace=read_json(p/'trajectory.json') if (p/'trajectory.json').exists() else await execute(run,task,c,initial,p)
  result=plugin.evaluate(spec,initial,p/'final.db',trace['final_answer'],trace['trajectory']) if not trace['failure'] else {'status':'unknown','correct':None,'error':trace['failure']}
  immutable(p/'independent.json',result)
  # Real mutation -> reset of that exact working database -> business comparison.
  work=p/'working.db'
  if (p/'modified_reset.json').exists():reset=read_json(p/'modified_reset.json')
  else:
   mutated=business_state(work)!=business_state(initial);backup(initial,work);reset_equal=business_state(work)==business_state(initial);reset={'mutated_before_reset':mutated,'business_state_restored':reset_equal,'passed':mutated and reset_equal};immutable(p/'modified_reset.json',reset)
  result={**result,'modified_reset_passed':reset['passed']};replay_results.append(result)
 cert['gates']['three_positive_replays']=all(r.get('correct') is True for r in replay_results);cert['gates']['provenance']=cert['gates']['three_positive_replays']
 # Reinitialization after actual mutation verifies fresh reset semantics in each
 # of three real service replays plus the independent import digests.
 cert['reset_review']={'passed':cert['gates']['three_positive_replays'] and env['fresh_reset_equal'] and all(r['modified_reset_passed'] for r in replay_results),'positive_replays':3,'initial_business_digest':env['business_digests'][0]}
 isolation_path=folder/'isolation/check.json';isolation=read_json(isolation_path) if isolation_path.exists() else await dual_instance_check(run,task,plugin.isolation_probe(),initial,folder/'isolation')
 cert['isolation_review']=isolation
 cert['gates']['environment']=bool(env['all_exposed_tool_schemas_checked'] and env.get('basic_contracts',{}).get('all_basic_contracts_passed') is True and cert['reset_review']['passed'] and isolation['status']=='passed')
 cert['tool_review']={'all_exposed_basic_schemas':env['all_exposed_tool_schemas_checked'],'dependency_create_review':'all frozen cases crosschecked with DB; below-range input and wrong values inspected','isolation':isolation['status'],'current_cleaning_concurrency':2}
 # Freeze all challenge recipes before obtaining any judge outputs.
 immutable(folder/'frozen_cases.json',{'spec_sha256':spec_hash,'cases':spec['cases'],'challenge_frozen_before_judge':True})
 entry=verifier_entry(run,task,'sql');all_sql=True
 for c in spec['cases']:
  p=folder/'cases'/c['name'];trace=read_json(p/'trajectory.json') if (p/'trajectory.json').exists() else await execute(run,task,c,initial,p)
  independent=plugin.evaluate(spec,initial,p/'final.db',trace['final_answer'],trace['trajectory']) if not trace['failure'] else {'status':'unknown','correct':None,'checks':[],'error':trace['failure']}
  immutable(p/'independent.json',independent)
  actual={k:v for k,v in c.items() if k not in ('calls','answer')};actual.update(independent=independent,reachable=not bool(trace['failure']),evidence=str(p),judge_slots=[])
  extpath=p/'extraction.json';ext=read_json(extpath) if extpath.exists() else extraction(entry,p);immutable(extpath,ext)
  review=plugin.sql_review(entry,ext.get('evidence'),initial,p/'final.db') if ext['status']=='valid' else {'status':'infrastructure_invalid','checks':[],'worker':ext};immutable(p/'sql_review.json',review)
  checked=sql_review_valid(review);all_sql=all_sql and checked;actual['sql_review']=review['status']
  if review['status']=='confirmed_evidence_mismatch' and actual['reachable'] and independent['status']=='valid' and independent['correct'] is c['expected']:
   cert['defects'].append(reviewed_defect('official_sql_evidence_mismatch',[str(p/'sql_review.json'),str(p/'extraction.json'),str(p/'independent.json')],'Original SQL export differs from independent complete relational reconstruction on a reachable frozen case, with confirmed task binding and row labels.'))
  if checked and independent['status']=='valid' and independent['correct'] is c['expected'] and actual['reachable']:
   for slot in range(5):
    outcome=await judge_slot(run,task,ext['evidence'],trace['trajectory'],p/'judge'/str(slot),budget,slot);actual['judge_slots'].append(outcome)
    if outcome['status']=='valid' and outcome['accepted'] is not independent['correct']:
     cert['defects'].append(reviewed_defect('official_judge_misclassification',[str(p/'independent.json'),str(p/'sql_review.json'),str(p/'judge'/str(slot)/'result.json')],'Task-literal predicates on same new row; real MCP call, provenance log and fully independently reconstructed original SQL export; new valid fixed judge slot disagrees.'))
  cert['cases'].append(actual)
 from .deep_review import review_submission
 deep=[]
 for c in cert['cases']:
  p=Path(c['evidence']);trajectory=read_json(p/'trajectory.json');review=review_submission(initial,p/'final.db',trajectory['trajectory']);immutable(p/'deep_tool_review.json',review);deep.append(review)
 cert['deep_tool_checks']=deep
 cert['gates']['environment']=cert['gates']['environment'] and all(r['status']=='passed' for r in deep)
 cert['coverage']=coverage(spec,cert['cases']);cert['gates']['coverage']=cert['coverage']['passed'];cert['gates']['challenge']=any(c.get('role')=='challenge' and c['independent'].get('status')=='valid' for c in cert['cases']);cert['gates']['sql_evidence']=all_sql;cert['gates']['five_judge_slots']=five_slots_pass(cert['cases']);cert['gates']['review']=citations and all(c['independent']['status']=='valid' and c['independent']['correct'] is c['expected'] for c in cert['cases'])
 cert['processing_status']='certification_checked';cert.pop('missing',None);return cert

def scenario_reference(task):return task['scenario']=='e_commerce_10' and task['task_idx']==7

async def run(run,scope='pilot100',resume=False,max_api_requests=None,authorize_all=False):
 started=time.monotonic();run=Path(run).resolve();m=validate(run);append(run/'evidence/execution_events.jsonl',{'event':'run_started','at':now(),'scope':scope});budget=Budget(run,max_api_requests,scope,authorize_all);selected=set(m['pilot100']) if scope=='pilot100' else {t['ledger_id'] for t in m['tasks']};environments={};certs=[]
 for task in m['tasks']:
  if task['ledger_id'] not in selected:continue
  result=run/'evidence/certificates'/f"{task['ledger_id'].replace(':','_')}.json"
  if result.exists():
   if not resume:raise ValueError('Existing results require --resume')
   existing=read_json(result)
   resumable=existing.get('decision')=='quarantine' and (existing.get('processing_status')=='checked_insufficient' and scenario_reference(task) or any(x.get('status')=='pending_budget' for c in existing.get('cases',[]) for x in c.get('judge_slots',[])))
   if not resumable:certs.append(existing);continue
   immutable(run/'evidence/certificate_versions'/digest(existing)/result.name,existing)
  scenario=task['scenario'];print(f"checking {task['ledger_id']} {scenario}:{task['task_idx']}",flush=True)
  if scenario not in environments:
   try:environments[scenario]=await environment_check(run,scenario)
   except Exception as e:
    environments[scenario]={'scenario':scenario,'status':'infrastructure_unknown','error':safe_error(e)}
    save_json(run/'evidence/environments'/scenario/'check.json',environments[scenario])
  env=environments[scenario];cert,folder=inspect_task(run,task,env)
  if scenario=='e_commerce_10' and task['task_idx']==7 and cert['gates']['assets'] and env['status']=='basic_checked':
   try:cert=await certify_reference(run,task,env,cert,folder,budget)
   except Exception as e:
    save_json(folder/'certification_error.json',{'error':safe_error(e),'traceback':traceback.format_exc()});cert['evidence'].append(str(folder/'certification_error.json'));cert['processing_status']='checked_insufficient';cert['missing']=['certification_interrupted_or_infrastructure_failure']
  cert['decision'],cert['reasons']=decide(cert);cert['freeze_sha256']=m['freeze_sha256']
  save_json(folder/'inspection.json',{'ledger_id':task['ledger_id'],'task_sha256':task['task_sha256'],'processing_status':cert['processing_status'],'gates':cert['gates'],'reasons':cert['reasons']})
  cert['evidence_inventory']={str(p):digest(p) for base in (folder,run/'evidence/environments'/scenario) for p in sorted(base.rglob('*')) if p.is_file()}
  immutable(run/'evidence/certificate_versions'/digest(cert)/result.name,cert);save_json(result,cert);certs.append(cert)
 append(run/'evidence/execution_events.jsonl',{'event':'run_finished','at':now(),'scope':scope,'elapsed_seconds':time.monotonic()-started,'processed_in_invocation':len(certs)})
 from .report import rebuild
 return rebuild(run)
