"""Unchanged official SQL extractor/prompt/parser, with durable fixed-slot transport."""
from __future__ import annotations
import asyncio,sys,time,json
from pathlib import Path
from types import SimpleNamespace
from awm_audit.config import load_config,validate_config
from awm_audit.verification import execute_verifier
from .common import *
from .budget import BudgetExhausted

class LoggedClient:
 def __init__(self,config,budget,slot):
  from openai import AsyncOpenAI
  self.config=config;self.budget=budget;self.slot=slot;self.log=[]
  self.client=AsyncOpenAI(base_url=config.base_url,api_key=config.api_key,timeout=120,max_retries=0)
  self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
 async def create(self,**kwargs):
  from openai import APIConnectionError,APITimeoutError,APIStatusError
  prior=sum(r['event']=='request_started' and r.get('slot')==self.slot for r in self.budget.records())
  for attempt in range(prior,3):
   try:request_id=self.budget.reserve(digest(kwargs),self.slot,attempt+1)
   except BudgetExhausted:
    self.log.append({'pending_budget':True,'attempt':attempt+1});raise
   started=time.monotonic();record={'request_id':request_id,'attempt':attempt+1,'request':kwargs,'started_at':now()};self.log.append(record)
   try:
    response=await self.client.chat.completions.create(**kwargs);data=response.model_dump(mode='json');record.update(response=data,finished_at=now())
    usage=data.get('usage',{}) or {};self.budget.finish(request_id,status='response',elapsed_seconds=time.monotonic()-started,usage=usage,finish_reason=(data.get('choices') or [{}])[0].get('finish_reason'));return response
   except Exception as e:
    status=getattr(e,'status_code',None);record.update(error=self.config.redact(e),error_type=type(e).__name__,status_code=status,finished_at=now())
    self.budget.finish(request_id,status='transport_error',status_code=status,error_type=type(e).__name__,elapsed_seconds=time.monotonic()-started)
    if status in (401,402,403):save_json(self.budget.path.parent/'api_circuit.json',{'blocked':True,'status_code':status,'at':now()})
    transient=isinstance(e,(APIConnectionError,APITimeoutError)) or isinstance(e,APIStatusError) and (status==429 or 500<=status<=599)
    if not transient or attempt==2:raise
    await asyncio.sleep(2**attempt)

async def judge_slot(run,task,extraction,trajectory,folder,budget,slot):
 folder=Path(folder);resultpath=folder/'result.json'
 if resultpath.exists():return read_json(resultpath)
 if (folder/'reserved.json').exists() and not (folder/'pending_budget.json').exists():return {'status':'interrupted_unknown','accepted':None,'slot':slot,'reason':'Reserved judge slot interrupted; no semantic resampling on resume'}
 if (budget.path.parent/'api_circuit.json').exists():return {'status':'authorization_error','accepted':None,'slot':slot}
 if not budget.available():return {'status':'pending_budget','accepted':None,'slot':slot,'reason':'No physical request dispatched; resumable after explicit budget increase'}
 config=load_config();validate_config(config);folder.mkdir(parents=True,exist_ok=True)
 immutable(folder/'reserved.json',{'slot':slot,'task_sha256':digest(task['task']),'extraction_sha256':digest(extraction),'trajectory_sha256':digest(trajectory)})
 sys.path.insert(0,str(AWM)) if str(AWM) not in sys.path else None
 from awm.core import verify
 client=LoggedClient(config,budget,str(folder.resolve()));original=verify.AsyncOpenAI;verify.AsyncOpenAI=lambda **kwargs:client
 try:classification,details=await verify.run_llm_judge(task['task'],extraction,config.base_url,config.api_key,config.model,trajectory)
 finally:
  verify.AsyncOpenAI=original;await client.client.close()
  previous=read_json(folder/'requests.json') if (folder/'requests.json').exists() else []
  save_json(folder/'requests.json',previous+client.log)
 if client.log and client.log[-1].get('pending_budget'):
  save_json(folder/'pending_budget.json',{'status':'pending_budget','slot':slot,'attempt':client.log[-1]['attempt']});return {'status':'pending_budget','accepted':None,'slot':slot}
 finish=[r['response']['choices'][0].get('finish_reason') for r in client.log if 'response' in r]
 responses=[r['response'] for r in client.log if 'response' in r]
 strict_format=False
 if responses:
  try:
   raw=json.loads(responses[-1]['choices'][0]['message'].get('content') or '')
   strict_format=isinstance(raw,dict) and raw.get('classification')==classification
  except (ValueError,KeyError,TypeError):pass
 valid=classification in ('complete','incomplete','agent_error') and finish and finish[-1]=='stop' and strict_format
 result={'status':'valid' if valid else 'invalid','classification':classification,'accepted':classification=='complete' if valid else None,'details':details,'slot':slot,'request_count':len(client.log),'finish_reason':finish[-1] if finish else None,'strict_json_format':strict_format}
 immutable(resultpath,result);return result

def extraction(entry,folder):
 result=execute_verifier(entry,'sql',folder)
 # Worker exception, mutation, empty extraction are infrastructure failures.
 if result.get('status')!='valid' or result.get('verdict')=='judge_error':return {'status':'invalid','accepted':None,'worker':result}
 return {'status':'valid','evidence':result['details'],'worker':result}
