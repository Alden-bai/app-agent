"""Cumulative physical requests, reserved durably before network dispatch."""
from __future__ import annotations
import contextlib,fcntl,time
from pathlib import Path
from .common import *
class BudgetExhausted(RuntimeError):pass
class Budget:
 def __init__(self,run,limit,scope,authorize_all=False):
  self.path=Path(run)/'evidence/api_requests.jsonl';self.limit=limit;self.scope=scope
  if scope=='all' and (limit is None or not authorize_all):raise ValueError('All scope requires --max-api-requests and --authorize-all-model-budget')
  if limit is not None and limit<0:raise ValueError('Budget must be nonnegative')
 def records(self):return rows(self.path) if self.path.exists() else []
 def available(self):
  return self.limit is None or sum(r['event']=='request_started' for r in self.records())<self.limit
 def reserve(self,request_hash,slot,attempt):
  self.path.parent.mkdir(parents=True,exist_ok=True)
  with (self.path.parent/'api.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   started=[r for r in self.records() if r['event']=='request_started']
   if self.limit is not None and len(started)>=self.limit:raise BudgetExhausted('Cumulative physical API request budget exhausted')
   request_id=len(started)+1;append(self.path,{'event':'request_started','request_id':request_id,'at':now(),'request_sha256':request_hash,'slot':slot,'attempt':attempt,'scope':self.scope});return request_id
 def finish(self,request_id,**fields):append(self.path,{'event':'request_finished','request_id':request_id,'at':now(),**fields})
