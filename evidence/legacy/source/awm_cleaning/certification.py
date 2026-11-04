"""Declarative admission gates; unknown evidence never silently passes."""
from __future__ import annotations
from .common import digest
ASPECTS=('object','join','filter','aggregate','null','omission')
GATES=('assets','environment','provenance','requirements','independent_rules_review','three_positive_replays','coverage','challenge','sql_evidence','five_judge_slots','review')
def sql_review_valid(review):
 return review.get('status')=='fully_checked' and all(review.get('aspects',{}).get(a) is True for a in ASPECTS) and bool(review.get('checks')) and all(c.get('passed') is True for c in review['checks'])
def coverage(spec,cases):
 requirements=spec.get('requirements',[]);missing=[]
 if not requirements:missing.append('requirements_absent')
 for r in requirements:
  if not r.get('id') or not r.get('quote'):missing.append('uncited_requirement')
  valid=[c for c in cases if c.get('independent',{}).get('status')=='valid' and c.get('reachable')]
  if not any(c['independent']['correct'] is True and any(ch.get('requirement_id')==r['id'] and ch.get('passed') is True for ch in c['independent'].get('checks',[])) for c in valid):missing.append('positive:'+r['id'])
  if r.get('hard',True) and not any(c['independent']['correct'] is False and r['id'] in c.get('violates',[]) and any(ch.get('requirement_id')==r['id'] and ch.get('passed') is False for ch in c['independent'].get('checks',[])) for c in valid):missing.append('effective_counterexample:'+r['id'])
 return {'passed':not missing,'missing':missing}
def decide(cert):
 hard=[d for d in cert.get('defects',[]) if d.get('review',{}).get('status')=='evidence_reviewed' and d.get('evidence')]
 if hard:return 'reject',['reviewed_hard_defect:'+d['kind'] for d in hard]
 missing=[g for g in GATES if cert.get('gates',{}).get(g) is not True]
 return ('keep',[]) if not missing else ('quarantine',missing)
def five_slots_pass(cases):
 return bool(cases) and all(len(c.get('judge_slots',[]))==5 and {x.get('slot') for x in c['judge_slots']}==set(range(5)) and all(x.get('status')=='valid' and x.get('accepted') is c.get('independent',{}).get('correct') for x in c['judge_slots']) for c in cases)
