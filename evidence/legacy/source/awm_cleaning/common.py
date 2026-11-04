from __future__ import annotations
import hashlib,json,os,sqlite3,tempfile
from pathlib import Path
from awm_audit.common import ROOT,AWM,backup,digest,now,query,read_json,save_json,rows
REVISION='dde80a0283fe781bdc51656bce57063dc5650213'
FILES=('gen_scenario.jsonl','gen_tasks.jsonl','gen_db.jsonl','gen_sample.jsonl','gen_spec.jsonl','gen_envs.jsonl','gen_verifier.jsonl')
PROTOCOL={'version':1,'formal_mode':'sql_llm','slots':5,'positive_replays':3,'temperature':1.0,'max_tokens':4096,'timeout':120,'transport_retries':2,'default_concurrency':1,'full_budget_requires_authorization':True}
def immutable(path,value):
 path=Path(path)
 if path.exists():
  if read_json(path)!=value:raise ValueError(f'Immutable evidence conflict: {path}')
  return
 save_json(path,value)
def append(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a') as f:f.write(json.dumps(value,ensure_ascii=False,default=str)+'\n');f.flush();os.fsync(f.fileno())
def write_rows(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.partial-')
 with os.fdopen(fd,'w') as f:
  for row in data:f.write(json.dumps(row,ensure_ascii=False,default=str)+'\n')
 os.replace(tmp,path)
def business_state(path):
 with sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro',uri=True) as db:
  tables=[r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
  return {t:sorted([list(r) for r in db.execute('SELECT * FROM "'+t.replace('"','""')+'"')],key=lambda r:json.dumps(r,sort_keys=True,default=str)) for t in tables}
def source_freeze():
 paths=sorted(p for folder in (ROOT/'awm_cleaning',ROOT/'awm_audit',AWM/'awm') for p in folder.rglob('*.py') if '__pycache__' not in p.parts and 'tests' not in p.parts)
 return {str(p.relative_to(ROOT)):digest(p) for p in paths}
def safe_error(e):return {'type':type(e).__name__,'message':str(e)}
