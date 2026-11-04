"""Independent response/persistence checks for review submission dependencies."""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from .common import business_state,query

def review_submission(initial,final,trace):
 checks=[]
 try:
  before,after=business_state(initial),business_state(final)
  checks.append({'name':'no_collateral_business_table_changes','passed':all(before[t]==after[t] for t in before if t!='reviews')})
  old={r['id']:r for r in query(initial,'SELECT * FROM reviews')};new={r['id']:r for r in query(final,'SELECT * FROM reviews')}
  checks.append({'name':'existing_reviews_preserved','passed':all(new.get(k)==v for k,v in old.items())})
  creates=[]
  for event in trace:
   for call in event.get('tool_calls',[]):
    if call.get('name')!='call_tool' or event.get('is_error'):continue
    name=call['arguments']['tool_name'];arguments=call['arguments']['arguments'];arguments=json.loads(arguments) if isinstance(arguments,str) else arguments
    response=event.get('tool_response',{}).get('content');response=json.loads(response) if isinstance(response,str) else response
    if name.startswith('create_review'):
     rid=response.get('review_id');row=new.get(rid);creates.append(rid)
     checks.append({'name':'create_return_matches_persisted_row','review_id':rid,'passed':row is not None and all(type(response.get(k)) is type(row.get(k)) and response.get(k)==row.get(k) for k in ('user_id','sku_id','rating','review_text','status')) and datetime.fromisoformat(response['created_at'])==datetime.fromisoformat(row['created_at'])})
     sku=query(final,'SELECT id FROM sku WHERE sku_code=?',[arguments['sku_code']]);expected_status=arguments.get('status','pending')
     checks.append({'name':'create_contract_fields_and_default','review_id':rid,'passed':bool(row) and len(sku)==1 and row['sku_id']==sku[0]['id'] and row['rating']==arguments['rating'] and row['review_text']==arguments['review_text'] and row['status']==expected_status})
    elif name.startswith('search_skus'):
     expected=query(initial,'SELECT id,sku_code FROM sku WHERE sku_code=? ORDER BY id',[arguments['sku_code']]);actual=[{'id':r['id'],'sku_code':r['sku_code']} for r in response.get('skus',[])]
     checks.append({'name':'search_exact_filter_complete_target_identity','passed':actual==expected and response.get('total')==len(expected),'expected':expected,'actual':actual})
  checks.append({'name':'all_new_review_rows_have_real_create_responses','passed':set(new)-set(old)==set(creates)})
  return {'status':'passed' if all(c['passed'] for c in checks) else 'needs_review','checks':checks,'scope':'POST review legal fields/default/returned values/new rows/existing rows/collateral tables; exact SKU search identity/filter/omission/count where used'}
 except Exception as e:return {'status':'unknown','checks':checks,'error':f'{type(e).__name__}: {e}'}
