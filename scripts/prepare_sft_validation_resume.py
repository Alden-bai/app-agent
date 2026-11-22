"""Audit frozen protocol, recover committed results, and requeue killed episodes."""
import json
import os
from pathlib import Path
import signal
import sqlite3
import time

from harness_awm.io import read_json,write_json,file_hash,verify_seal,now
from harness_awm.training.selection import sources
from harness_awm.evaluation_metrics import evaluation_score
from harness_awm.backend import Budget
from scripts.sft_eval_local_recovery import alive,tree_hashes
from scripts.resume_sft_validation import SOURCE,REPAIRS,CONTROL,ARCHIVE,ROOT


def audit_source_additions(old,current):
    changed=[k for k in old if old[k]!=current.get(k)]
    added=[k for k in current if k not in old]
    if changed or any(not k.startswith('harness_awm/training/verl/') for k in added):
        raise ValueError('existing evaluation implementation changed: '+str(changed+added))
    return added


def main():
    CONTROL.mkdir(exist_ok=True)
    if (CONTROL/'resume_audit.json').exists():raise RuntimeError('already prepared')
    for root in [SOURCE,REPAIRS]:
        launch=read_json(root/'launch.json')
        if alive(launch['pid']):raise RuntimeError('old supervisor still alive')
    for pid in read_json(SOURCE/'validation/launch.json')['worker_pids']:
        if alive(pid):raise RuntimeError('old evaluation worker still alive')
    originals={}
    current=sources()
    for root in [SOURCE,REPAIRS]:
        plan=read_json(root/'plan.json')
        assert file_hash(root/'plan.json')==read_json(root/'plan.sha256.json')['sha256']
        added=audit_source_additions(plan['source_hashes'],current)
        originals[root.name]={'plan':plan,'plan_sha256':file_hash(root/'plan.json'),'added_training_only_files':added}
    assert 'No space left on device' in read_json(SOURCE/'status.json')['error']
    write_json(CONTROL/'original_plans.json',originals)
    with sqlite3.connect(SOURCE/'validation/queue.sqlite') as src,sqlite3.connect(CONTROL/'queue_before.sqlite') as dst:
        src.backup(dst)
    with sqlite3.connect(CONTROL/'queue_before.sqlite') as db:
        db.row_factory=sqlite3.Row
        claimed=[dict(r) for r in db.execute("SELECT * FROM jobs WHERE state='claimed'")]
    killed=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:parts=(entry/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        if any(any(part.startswith((row['output']+'/episodes/').encode()) and part.endswith(b'/runtime/server.py') for part in parts) for row in claimed):
            pid=int(entry.name)
            if os.getpgid(pid)==pid:os.killpg(pid,signal.SIGTERM)
            else:os.kill(pid,signal.SIGTERM)
            killed.append(pid)
    time.sleep(1)
    audit=[];manifests={}
    for row in claimed:
        output=Path(row['output']);ep=output/'episodes/t00000/a0'
        item={'job':row,'files_sha256':tree_hashes(output)}
        if (ep/'episode_result.json').exists():
            result=read_json(ep/'episode_result.json');verify_seal(ep)
            assert result['task_instance_id']==row['task']
            item['action']='reuse_complete_result'
        else:
            if (ep/'evidence_seal.json').exists():raise RuntimeError('sealed episode without final result needs separate audit')
            new=output.with_name(output.name+'_storage_retry1')
            assert not new.exists()
            item.update(action='retry_infrastructure_interruption',new_output=str(new))
            manifest=manifests.setdefault(row['candidate'],read_json(SOURCE/'validation'/row['candidate']/'manifest.json'))
            found=[x for x in manifest['candidates'] if x['task_instance_id']==row['task']]
            assert len(found)==1 and found[0]['path']==str(ep)
            found[0].update(path=str(new/'episodes/t00000/a0'),original_interrupted_attempt=str(ep),recovery_action='supervisor_inode_failure_killed_worker')
        audit.append(item)
    # Persist the complete decision before modifying queue/manifest records.
    write_json(CONTROL/'claim_audit.json',{'claims':audit,'terminated_orphan_servers':killed,'at':now()})
    for name,manifest in manifests.items():
        path=SOURCE/'validation'/name/'manifest.json'
        write_json(CONTROL/f'manifests_before/{name}.json',read_json(path))
        write_json(path,manifest);write_json(path.with_name('manifest.sha256.json'),{'sha256':file_hash(path)})
    with sqlite3.connect(SOURCE/'validation/queue.sqlite') as db:
        for item in audit:
            row=item['job']
            if item['action']=='reuse_complete_result':db.execute("UPDATE jobs SET state='done' WHERE id=? AND state='claimed'",(row['id'],))
            else:db.execute("UPDATE jobs SET state='pending',owner=NULL,output=? WHERE id=? AND state='claimed'",(item['new_output'],row['id']))
    restored=[]
    for job in sorted((REPAIRS/'judge_jobs_v1').iterdir()):
        if (job/'recovery.json').exists():continue
        started=read_json(job/'started.json');row=started['source'];ep=job/'episode'
        result=read_json(ep/'episode_result.json');verify_seal(ep)
        assert result['task_instance_id']==row['task_instance_id']
        assert file_hash(Path(row['episode'])/'episode_result.json')==started['original_result_sha256']
        assert file_hash(Path(row['episode'])/'evidence_seal.json')==started['original_seal_sha256']
        caches=list((ep/'evaluations_v2'/result['evaluation_id']).glob('attempt_*/result.json'))
        assert len(caches)==1
        assessment=read_json(caches[0])
        assert assessment['evaluation_id']==result['evaluation_id']
        assert assessment['evaluation_status']==result['evaluation_status']
        value={k:started[k] for k in ['original_episode','original_result_sha256','original_seal_sha256']}
        value.update(key=job.name,candidate=row['candidate'],task_instance_id=row['task_instance_id'],
            source_episode=row['episode'],episode=str(ep),reason=row['kind'],at=now(),split=row.get('split','validation'),
            in_previous_116=row.get('in_previous_116',False),evaluation_status=result['evaluation_status'],
            task_status=result.get('task_status'),score=evaluation_score(result),errors=assessment['errors'],
            usage=Budget(job/'budget.sqlite',32,None).summary(),restored_cached_assessment=True)
        write_json(job/'recovery.json',value);restored.append(value)
    for root in [SOURCE,REPAIRS]:
        plan=read_json(root/'plan.json');plan['source_hashes']=current
        plan['resume_source_audit']=str(CONTROL/'original_plans.json')
        write_json(root/'plan.json',plan);write_json(root/'plan.sha256.json',{'sha256':file_hash(root/'plan.json')})
    ARCHIVE.mkdir(parents=True,exist_ok=True)
    stopped=ROOT/'runs/qwen35_verl_sft_20261109_v4'
    status=read_json(stopped/'status.json');status.update(stage='stopped_by_user',reason='reuse existing four SFT checkpoints for validation',updated_at=now())
    write_json(stopped/'status.json',status)
    summary={'at':now(),'reused_complete_claims':sum(x['action']=='reuse_complete_result' for x in audit),
        'requeued_killed_episodes':sum(x['action']=='retry_infrastructure_interruption' for x in audit),
        'restored_judge_results_without_api_calls':len(restored),'archive':str(ARCHIVE),'protocol_changed':False}
    write_json(CONTROL/'resume_audit.json',summary);print(json.dumps(summary))

if __name__=='__main__':main()
