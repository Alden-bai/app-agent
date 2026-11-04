from __future__ import annotations
import asyncio,ast,json,sqlite3
from pathlib import Path
from awm_audit import runtime as legacy
from awm_audit.assets import scenario_asset
from .common import *
from .provenance import resolve

async def execute(run,task,case,initial,folder):
 log=[];original=legacy.resolve
 final_answer=object()
 def strict(value,_work,results):
  if value is final_answer:return case.get('answer','')
  return resolve(value,task['task'],results,log,public_contracts=task.get('_public_contracts',{}))
 # Legacy runtime is intentionally serial: its single adapter global cannot be
 # used across concurrent task executions. Isolation uses separate processes.
 legacy.resolve=strict
 try:
  actual={**case,'answer':final_answer}
  payload=await legacy.execute_case(run,task,actual,initial,folder)
 finally:legacy.resolve=original
 save_json(Path(folder)/'argument_provenance.json',log)
 return payload

def tool_schema_review(tools):
 import jsonschema
 out=[]
 for tool in tools:
  try:
   jsonschema.Draft202012Validator.check_schema(tool.get('inputSchema',{}))
   out.append({'tool':tool['name'],'schema_sha256':digest(tool.get('inputSchema',{})),'schema_valid':True,'semantic_status':'not_checked','reason':'JSON schema validity alone does not establish business behavior'})
  except Exception as e:out.append({'tool':tool.get('name'),'schema_valid':False,'error':safe_error(e),'semantic_status':'unknown'})
 return out

async def environment_check(run,scenario):
 folder=Path(run)/'evidence/environments'/scenario
 if (folder/'check.json').exists():return read_json(folder/'check.json')
 folder.mkdir(parents=True,exist_ok=True);initials=[];errors=[];digests=[]
 for i in range(3):
  p=folder/f'reset_{i}.db'
  try:
   report=legacy.initialize(run,scenario,p);state=business_state(p);digests.append(digest(state));initials.append(str(p))
  except Exception as e:
   evidence=p.with_suffix('.initialization.json');report=read_json(evidence) if evidence.exists() else {'error':safe_error(e)}
   errors.append({'cycle':i,'report':report,'evidence':str(evidence)})
 if errors:
  check={'scenario':scenario,'status':'initialization_defect','review':{'status':'evidence_reviewed','basis':'original DDL/sample errors or SQLite integrity/foreign-key failures in three independent complete imports'},'errors':errors,'initial_databases':initials,'all_tasks_affected':True}
  immutable(folder/'check.json',check);return check
 t={'scenario':scenario,'task_idx':None,'task':'Discover all formally exposed tools.'}
 payload=await legacy.execute_case(run,t,{'name':'discover','calls':[],'answer':'Discovery complete.','expected':True},initials[0],folder/'discovery')
 if payload['failure']:
  check={'scenario':scenario,'status':'infrastructure_unknown','errors':[payload['failure']],'initial_databases':initials,'discovery':str(folder/'discovery/trajectory.json')}
 else:
  tools=read_json(folder/'discovery/tools.json');schemas=tool_schema_review(tools)
  from .tool_contracts import basic_contracts
  basic=await basic_contracts(run,scenario,initials[0],folder/'basic_contracts')
  spec=scenario_asset(run,'gen_spec.jsonl',scenario)
  check={'scenario':scenario,'status':'basic_checked','initial_databases':initials,'business_digests':digests,'fresh_reset_equal':len(set(digests))==1,'modified_reset_checked':False,'isolation_checked':False,'all_exposed_tool_schemas_checked':all(x['schema_valid'] for x in schemas),'tool_count':len(tools),'tools':schemas,'basic_contracts':basic,'spec_sha256':digest(spec),'discovery':str(folder/'discovery/trajectory.json'),'missing':['task_dependency_business_semantics','modify_then_reset','dual_instance_write_read_isolation','current_cleaning_concurrency_isolation']}
 immutable(folder/'check.json',check);return check

async def dual_instance_check(run,task,probe,initial,folder):
 """Two live official MCP servers; real A write, then B read and audit snapshot."""
 import os,subprocess,sys
 from contextlib import AsyncExitStack
 import httpx
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);processes=[];logs=[];ports=[];dbs=[]
 before=business_state(initial);evidence={'status':'unknown','instances':[]}
 try:
  for label in ('A','B'):
   root=folder/label;root.mkdir(exist_ok=True);work=root/'working.db';backup(initial,work);port=legacy.free_port();ports.append(port);dbs.append(work)
   script=legacy.server_script(run,task['scenario'],work,root,port);log=(root/'server.log').open('w');logs.append(log)
   env={k:v for k,v in os.environ.items() if k in ('PATH','LANG','LC_ALL','TZ')};env.update(HOST='127.0.0.1',PORT=str(port),DATABASE_PATH=f'sqlite:///{work}')
   p=subprocess.Popen([sys.executable,script],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);processes.append(p);evidence['instances'].append({'label':label,'pid':p.pid,'port':port,'db':str(work)})
  async with httpx.AsyncClient(timeout=1,trust_env=False) as http:
   for port,p in zip(ports,processes):
    for _ in range(120):
     if p.poll() is not None:raise RuntimeError('Isolation server exited')
     try:
      if (await http.get(f'http://127.0.0.1:{port}/openapi.json')).status_code==200:break
     except (httpx.HTTPError,OSError):pass
     await asyncio.sleep(.25)
    else:raise TimeoutError('Isolation service startup')
  async with AsyncExitStack() as stack:
   a=await stack.enter_async_context(legacy.session(f'http://127.0.0.1:{ports[0]}/mcp'));b=await stack.enter_async_context(legacy.session(f'http://127.0.0.1:{ports[1]}/mcp'))
   ta=(await a.list_tools()).tools;tb=(await b.list_tools()).tools;returns=[];calls=[];provenance=[]
   for step in probe['calls']:
    aliases=read_json(folder/'A/tool_aliases.json');name=legacy.select_tool(ta,aliases.get(step['tool'],step['tool']));args=resolve(step['arguments'],task['task'],returns,provenance,public_contracts=task.get('_public_contracts',{}))
    props=next(t for t in ta if t.name==name).inputSchema.get('properties',{})
    for wrapper in ('body','payload'):
     if wrapper in args and wrapper not in props:args={**args.pop(wrapper),**args}
    result=await a.call_tool(name,args);plain=legacy.unpack_result(result);returns.append(plain);calls.append({'name':name,'arguments':args,'response':plain,'is_error':result.isError})
   # Target review has no public read interface. B reads the same SKU through
   # its actual MCP session and the auditor reads B's complete business state.
   name=legacy.select_tool(tb,'get_sku_by_code');args={'sku_code':resolve(probe['calls'][0]['arguments']['body']['sku_code'],task['task'],[],provenance,public_contracts=task.get('_public_contracts',{}))}
   result=await b.call_tool(name,args);bread={'name':name,'arguments':args,'response':legacy.unpack_result(result),'is_error':result.isError}
   aftera=business_state(dbs[0]);afterb=business_state(dbs[1]);changed=aftera!=before;isolated=afterb==before
   evidence.update(status='passed' if changed and isolated and not any(c['is_error'] for c in calls) and not result.isError else 'unknown',both_live=all(p.poll() is None for p in processes),A_calls=calls,B_read=bread,A_changed=changed,B_business_unchanged=isolated,A_before_digest=digest(before),A_after_digest=digest(aftera),B_after_digest=digest(afterb),provenance=provenance)
 except Exception as e:evidence['error']=safe_error(e)
 finally:
  for p in processes:
   if p.poll() is None:
    p.terminate()
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)
  for log in logs:log.close()
 immutable(folder/'check.json',evidence);return evidence
