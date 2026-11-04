"""Standalone loader shipped only with public runtime assets, never audit recipes."""
from __future__ import annotations
import hashlib,json,sqlite3,sys
from pathlib import Path

def verify_release(root):
 root=Path(root).resolve();manifest=json.loads((root/'manifest.json').read_text())
 for name,expected in manifest['checksums'].items():
  path=root/name
  if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Release artifact checksum mismatch: '+name)
 return manifest

def load_task(root,ledger_id,output_database):
 root=Path(root).resolve();verify_release(root);tasks=[json.loads(line) for line in (root/'tasks.jsonl').read_text().splitlines() if line]
 selected=[t for t in tasks if t['ledger_id']==ledger_id]
 if len(selected)!=1:raise ValueError('Task instance not uniquely published')
 task=selected[0];target=Path(output_database).resolve()
 if target.exists():raise ValueError('Refusing to overwrite task database')
 source=root/'initial'/(task['scenario']+'.db');target.parent.mkdir(parents=True,exist_ok=True)
 with sqlite3.connect(f'file:{source}?mode=ro',uri=True) as src,sqlite3.connect(target) as dst:src.backup(dst)
 def entry(name,verifier=False):
  entries=[json.loads(line) for line in (root/'assets'/name).read_text().splitlines() if line]
  matches=[r for r in entries if r.get('scenario',r.get('name'))==task['scenario'] and (not verifier or r.get('task_idx')==task['task_idx'])]
  if len(matches)!=1:raise ValueError('Published asset binding ambiguous')
  return matches[0]
 return {'task':task,'database':str(target),'environment':entry('gen_envs.jsonl'),'verifier':entry('gen_verifier.jsonl',True),'judge_config':json.loads((root/'judge_config.json').read_text()),'runtime_source':str(root/'runtime')}
if __name__=='__main__':print(json.dumps(verify_release(Path(__file__).resolve().parent),indent=2))
