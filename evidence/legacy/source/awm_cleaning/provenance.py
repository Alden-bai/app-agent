"""Strict task/public/return/computation references. No executor SQL or hidden facts."""
from __future__ import annotations
import json,operator,re
class ProvenanceError(ValueError):pass

def resolve(value,task,results,log,path=(),public_contracts=None):
 public_contracts=public_contracts or {}
 if isinstance(value,dict):
  if '$sql' in value:raise ProvenanceError('Executor SQL references are forbidden')
  if '$literal' in value:
   quote=value.get('quote','');v=value['$literal']
   if type(v) not in (str,int,float,bool,type(None)):raise ProvenanceError('Task literals must be primitive scalar values')
   if not quote or quote not in task:raise ProvenanceError('Literal citation absent from original task')
   if isinstance(v,str) and v not in quote:raise ProvenanceError('Literal value absent from quoted task')
   if type(v) in (int,float) and not re.search(r'(?<![\w.])'+re.escape(str(v))+r'(?![\w.])',quote):raise ProvenanceError('Numeric literal absent as full token in quoted task')
   if type(v) is bool and not re.search(r'\b'+str(v).lower()+r'\b',quote,re.I):raise ProvenanceError('Boolean literal absent from quoted task')
   if v is None and not re.search(r'\b(null|none)\b',quote,re.I):raise ProvenanceError('Null literal absent from quoted task')
   log.append({'path':list(path),'kind':'task_literal','quote':quote,'value':v});return v
  if '$public' in value:
   if not value.get('citation'):raise ProvenanceError('Public constant needs contract citation')
   if value['citation'] not in public_contracts or (type(public_contracts[value['citation']]) is not type(value['$public']) or public_contracts[value['citation']]!=value['$public']):raise ProvenanceError('Public constant is not in the frozen verified contract registry')
   log.append({'path':list(path),'kind':'public_contract','citation':value['citation'],'value':value['$public']});return value['$public']
  if '$result' in value:
   idx=value['$result']
   if type(idx) is not int or not -len(results)<=idx<len(results):raise ProvenanceError('Reference is not prior tool output')
   result=results[idx]
   for key in value.get('path',[]):result=result[key]
   log.append({'path':list(path),'kind':'prior_tool_return','index':idx,'return_path':value.get('path',[]),'value':result});return result
  if '$compute' in value:
   op=value['$compute'];args=[resolve(v,task,results,log,path+('args',i),public_contracts) for i,v in enumerate(value.get('args',[]))]
   functions={'add':operator.add,'subtract':operator.sub,'multiply':operator.mul,'divide':operator.truediv,'concat':lambda *xs:''.join(map(str,xs)),'list':lambda *xs:list(xs)}
   if op not in functions:raise ProvenanceError('Unregistered reproducible computation')
   result=functions[op](*args);log.append({'path':list(path),'kind':'computation','op':op,'args':args,'value':result});return result
  if any(k.startswith('$') for k in value):raise ProvenanceError('Unknown reference operator')
  return {k:resolve(v,task,results,log,path+(k,),public_contracts) for k,v in value.items()}
 if isinstance(value,list):return [resolve(v,task,results,log,path+(i,),public_contracts) for i,v in enumerate(value)]
 raise ProvenanceError('Every scalar executor argument requires an explicit provenance reference')

def inspect_legacy(value,task,path=()):
 """Diagnose old recipes; never silently upgrade DB-derived constants."""
 issues=[]
 if isinstance(value,dict):
  if '$sql' in value:issues.append({'path':list(path),'reason':'executor_sql_reference'})
  elif '$compute' in value:issues.append({'path':list(path),'reason':'legacy_computation_not_provenance_validated'})
  elif '$result' not in value:
   for k,v in value.items():issues+=inspect_legacy(v,task,path+(k,))
 elif isinstance(value,list):
  for i,v in enumerate(value):issues+=inspect_legacy(v,task,path+(i,))
 elif (value is None or isinstance(value,bool)) or isinstance(value,(str,int,float)) and str(value) not in task:issues.append({'path':list(path),'reason':'literal_not_in_task_or_prior_tool_response','value':value})
 return issues
