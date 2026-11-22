"""Freeze and categorize unscored attempts without modifying model outputs."""
from pathlib import Path
from collections import Counter
from datetime import datetime
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from harness_awm.io import read_json,write_json,file_hash,now
from harness_awm.evaluation_metrics import evaluation_score


def inspect_result(folder):
    result=read_json(folder/'episode_result.json')
    candidates=sorted((folder/'evaluations_v2'/str(result.get('evaluation_id'))).glob('attempt_*/result.json'))
    if not candidates:candidates=sorted((folder/'evaluations_v2').glob('**/result.json'))
    assessment=read_json(candidates[-1]) if candidates else {}
    error=result.get('error')
    errors=assessment.get('errors',[])
    if error and 'database_foreign_keys_failed' in error:kind='snapshot_foreign_keys'
    elif result.get('evaluation_status')=='rubric_error':kind='rubric_error'
    elif any('query_round_budget_exhausted' in x for x in errors):kind='query_round_budget_exhausted'
    elif any('required_evidence_incomplete' in x for x in errors):kind='required_evidence_incomplete'
    elif error:kind='execution_error'
    else:kind='other_judge_error'
    return {'kind':kind,'errors':errors,'execution_error':error,'assessment_path':str(candidates[-1]) if candidates else None,
        'rubric_issues':assessment.get('rubric_issues',[]),'evaluation_status':result.get('evaluation_status'),
        'execution_integrity':result.get('execution_integrity'),'terminal_state_status':result.get('terminal_state_status'),
        'termination_reason':result.get('termination_reason'),'result_sha256':file_hash(folder/'episode_result.json'),
        'seal_sha256':file_hash(folder/'evidence_seal.json')}


def main():
    source=ROOT/'runs/qwen35_sft_eval_local_20261109_v1';out=ROOT/'runs/sft_judge_recovery_20261109_v1'
    out.mkdir(exist_ok=True)
    cutoff=datetime.fromisoformat('2026-11-09T12:07:21.245645+00:00').timestamp()
    with sqlite3.connect(source/'validation/queue.sqlite') as db:
        done={str(Path(x[0])/'episodes/t00000/a0') for x in db.execute("SELECT output FROM jobs WHERE state='done'")}
    rows=[];recorded=prior_records=0
    for name in read_json(source/'validation/candidates.json'):
        manifest=read_json(source/'validation'/name/'manifest.json')
        for item in manifest['candidates']:
            folder=Path(item['path'])
            if str(folder) not in done and not item.get('reused_original_attempt'):continue
            result=read_json(folder/'episode_result.json');recorded+=1
            prior=(folder/'episode_result.json').stat().st_mtime<=cutoff
            prior_records+=prior
            if evaluation_score(result) is not None:continue
            rows.append({'candidate':name,'task_instance_id':item['task_instance_id'],'episode':str(folder),
                         'in_previous_116':prior,**inspect_result(folder)})
    report={'at':now(),'source':str(source),'previous_status_cutoff':cutoff,'recorded':recorded,
        'previous_recorded':prior_records,'unscored':len(rows),'previous_unscored':sum(r['in_previous_116'] for r in rows),
        'categories':dict(Counter(r['kind'] for r in rows)),
        'previous_categories':dict(Counter(r['kind'] for r in rows if r['in_previous_116'])), 'attempts':rows}
    write_json(out/'initial_audit.json',report)
    print({k:v for k,v in report.items() if k!='attempts'})
    for kind in report['categories']:
        examples=[r for r in rows if r['kind']==kind][:2]
        for r in examples:print('EXAMPLE',r)


if __name__=='__main__':main()
