#!/usr/bin/env python3
"""Complete reference-format and pre-dispatch length failures without resampling Judge."""
import asyncio
import fcntl
import json
import re
import shutil
import sqlite3
from pathlib import Path
import judge_accuracy100 as base
import judge_accuracy100_openai as original
from judge_accuracy100_openai_v3 import Study
from harness_awm.evaluation import apply_evaluation, evaluate_episode
from harness_awm.io import canonical, digest, file_hash, now, read_json, verify_seal, write_json


def normalize_evidence(candidate, evidence, scenario, tables):
    """Resolve explicit source aliases or IDs already written in a reason; no label edits."""
    candidate=json.loads(json.dumps(candidate));changes=[]
    for item in candidate.get('items',[]):
        refs=item.get('evidence_ids')
        if refs is None:
            reason=item.get('reason','')
            refs=[k for k in evidence if k.startswith('q') and re.search(r'(?<![\w])'+re.escape(k)+r'(?![\w])',reason)]
            if re.search(r'\btrajectory\b',reason,re.I):refs.append('trajectory')
            if re.search(r'\braw tool results?\b',reason,re.I):refs.append('runtime')
            if not refs:raise ValueError('missing_evidence_not_recoverable_from_explicit_references')
            changes.append({'item':item['id'],'operation':'extract_explicit_existing_evidence_ids_from_reason','ids':refs})
        mapped=[]
        for ref in refs:
            target=ref
            if ref=='contract:'+scenario:target='contract'
            elif ref.startswith('schema:') and ref.split(':',1)[1] in tables:target='schema'
            elif ref=='raw_tool_results':target='runtime'
            if target not in evidence:raise ValueError('unresolved_evidence_id:'+ref)
            if target!=ref:changes.append({'item':item['id'],'from':ref,'to':target})
            if target not in mapped:mapped.append(target)
        item['evidence_ids']=mapped
    return candidate,changes


class Finish(Study):
    def __init__(self,run):
        super().__init__(run)
        # The old 128000 limit counted serialized UTF-8 BYTES, not model tokens.
        # 512000 bytes conservatively bounds input below the model's documented
        # 1050000-token context, without truncating any original Judge evidence.
        self.config=self.config.model_copy(update={
            'actor_api':self.config.actor_api.model_copy(update={'context_limit':512000}),
            'judge_api':self.config.judge_api.model_copy(update={'context_limit':512000})})
        p=self.run/'completion_amendment.json'
        if not p.exists():
            write_json(p,{'at':now(),'implementation':str(Path(__file__).resolve()),'sha256':file_hash(__file__),
                'scope':'Repair source-ID formatting from saved candidates and fresh blind reference review; first official dispatch for requests previously rejected by local length guard only.',
                'length_guard':'512000 serialized UTF-8 input/output-reservation bytes for reference/Judge; actual Actor execution remains unchanged.',
                'budget_cap':base.CAP,'api':self.config.judge_api.model_dump(),
                'no_resampling':'Never call official Judge again if any physical request exists for this task, including unknown or invalid results.',
                'label_policy':'Preserve old labels. Fresh reference contexts exclude official results. All posthoc repairs reported separately.'})
        else:assert read_json(p)['sha256']==file_hash(__file__)
        self.finish_sem=asyncio.Semaphore(4)

    def judge_request_count(self,rid):
        with sqlite3.connect(self.run/'budget.sqlite') as db:
            return db.execute("SELECT count(*) FROM requests WHERE stage='judge' AND request_key LIKE ?",(rid+'/judge/%',)).fetchone()[0]

    async def repair_reference(self,entry,out,result,refpath):
        ref=read_json(refpath);error=ref.get('error','')
        allowed=['invalid_reference_evidence','api_context_limit_exceeded','incomplete_reference_output','review_evidence_invalid']
        if not any(x in error for x in allowed):return result
        dest=out/'reference_completion';frozen=dest/'reference_frozen.json'
        if not frozen.exists():
            replies=[read_json(p) for p in sorted((refpath.parent/'reference').glob('*/parsed.json'))]
            candidate=next((p for p in reversed(replies) if p.get('action')=='finish'),None)
            try:
                verify_seal(out/'episode')
                if candidate is None:
                    task=base.TaskSpec.model_validate(read_json(out/'task_spec.json'))
                    new=await self.reference(entry,task,out/'episode',dest)
                else:
                    payload=read_json(refpath.parent/'reference_input.json')
                    queries=read_json(refpath.parent/'queries.json') if (refpath.parent/'queries.json').exists() else []
                    evidence={k:{'valid':True} for k in ['task','contract','schema','trajectory','runtime']}
                    evidence.update({q['id']:q for q in queries})
                    with sqlite3.connect((out/'episode/final.db').resolve().as_uri()+'?mode=ro',uri=True) as db:
                        tables={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
                    normalized,changes=normalize_evidence(candidate,evidence,entry['scenario'],tables)
                    rubric=read_json(Path(entry['input'])/'rubric.json')
                    base.validate_reference(normalized,rubric,evidence)
                    write_json(dest/'normalization.json',{'original':candidate,'normalized':normalized,'changes':changes})
                    write_json(dest/'reference_input.json',payload)
                    write_json(dest/'queries.json',queries)
                    fresh=await self.llm(dest/'reference_review','reference_review',[
                        {'role':'system','content':self.m['review_prompt']+' Allowed evidence_ids: '+','.join(evidence)+'.'},
                        {'role':'user','content':canonical({'evidence':payload,'queries':[v for k,v in evidence.items() if k not in {'task','contract','schema','trajectory','runtime'}],'candidate':normalized})}])
                    if fresh.get('label') not in {'pass','fail','unknown'} or not fresh.get('reason'):raise ValueError('invalid_review')
                    # Review IDs may use the same exact source aliases; normalize
                    # through the same auditable mapping, never infer its label.
                    reviewed,review_changes=normalize_evidence({'items':[{'id':'review','reason':fresh['reason'],'evidence_ids':fresh.get('evidence_ids')}]},evidence,entry['scenario'],tables)
                    fresh['evidence_ids']=reviewed['items'][0]['evidence_ids']
                    if not fresh['evidence_ids'] or all(not evidence[k].get('valid',True) for k in fresh['evidence_ids']):
                        raise ValueError('review_has_no_valid_evidence')
                    write_json(dest/'review_source_mapping.json',review_changes)
                    label=normalized['task_label'] if normalized['task_label']==fresh['label'] else 'unknown'
                    new={'label':label,'candidate':normalized,'review':fresh,'reference_sql_revision':3,
                         'rubric_sha256':entry['rubric_sha256'],'normalization_sha256':file_hash(dest/'normalization.json')}
                verify_seal(out/'episode')
            except Exception as exc:new={'label':'unknown','error':f'{type(exc).__name__}: {exc}','reference_sql_revision':3}
            new.update(frozen_at=now(),blind_to_official=True,posthoc_reference_repair=True,prior_reference_sha256=file_hash(refpath))
            write_json(frozen,new)
        new=read_json(frozen)
        result.update(reference_label=new['label'],active_reference=str(frozen),reference_hash=file_hash(frozen),
            reference_sql_revision=3,reference_repaired_after_judge=self.judge_request_count(entry['id'])>0)
        for k in ['agreement','dispute','dispute_error']:result.pop(k,None)
        if new['label'] in {'pass','fail'} and result['judge_valid']:
            result['agreement']=(new['label']=='pass')==result['official_pass']
        return result

    async def finish_one(self,entry):
        async with self.finish_sem:
            out=self.run/'tasks'/entry['id'];result=read_json(out/'result.json')
            marker=out/'completion_check.json'
            if marker.exists():return result
            history=out/'assessment_history/before_completion';history.mkdir(parents=True,exist_ok=True)
            if not (history/'result.json').exists():shutil.copyfile(out/'result.json',history/'result.json')
            refpath=Path(result.get('active_reference',out/'reference_frozen.json'))
            if refpath.exists():result=await self.repair_reference(entry,out,result,refpath)
            evaluation_path=out/'official_evaluation.json'
            if evaluation_path.exists():
                prior=read_json(evaluation_path)
                if not prior['evaluation_valid'] and 'api_context_limit_exceeded' in prior.get('error','') and self.judge_request_count(entry['id'])==0:
                    task=base.TaskSpec.model_validate(read_json(out/'task_spec.json'))
                    write_json(out/'judge_first_dispatch_after_guard.json',{'at':now(),'reference_hash':result['reference_hash'],'prior_physical_judge_requests':0})
                    evaluation=await evaluate_episode(out/'episode',task,self.config,self.budget,self.judge_sem)
                    apply_evaluation(out/'episode',evaluation)
                    write_json(out/'official_evaluation_completion.json',evaluation)
                    result.update(judge_valid=evaluation['evaluation_valid'],official_pass=evaluation['official_pass'],classification=evaluation['outcome_classification'],active_official_evaluation=str(out/'official_evaluation_completion.json'))
                    result.pop('agreement',None)
                    if result['reference_label'] in {'pass','fail'} and result['judge_valid']:
                        result['agreement']=(result['reference_label']=='pass')==result['official_pass']
            write_json(out/'result.json',result)
            write_json(marker,{'at':now(),'reference_label':result['reference_label'],'judge_valid':result['judge_valid']})
            print(canonical({'completed_evidence_check':entry['task_id'],'reference':result['reference_label'],'official_pass':result['official_pass']}),flush=True)
            self.report();return result

    async def dispute_one(self,entry):
        async with self.finish_sem:
            out=self.run/'tasks'/entry['id'];result=read_json(out/'result.json')
            if result.get('agreement') is not False:return
            destination=out/'final_dispute'
            if (destination/'outcome.json').exists():return
            refpath=Path(result.get('active_reference',out/'reference_frozen.json'))
            refroot=refpath.parent
            # Fallback for the original reference stored at the task root.
            payload=read_json(refroot/'reference_input.json')
            queries=read_json(refroot/'queries.json') if (refroot/'queries.json').exists() else []
            evaluation=read_json(result.get('active_official_evaluation',out/'official_evaluation.json'))
            try:
                answer=await self.llm(destination,'dispute',[
                    {'role':'system','content':self.m['dispute_prompt']},
                    {'role':'user','content':canonical({'reference_input':payload,'queries':queries,'frozen_reference':read_json(refpath),'official_evaluation':evaluation,'actual_official_request':read_json(out/'episode'/evaluation['attempt']/'official_request.json')})}])
                allowed={'judge_error','reference_error','rubric_error','adapter_error','environment_issue','insufficient_evidence'}
                if answer.get('attribution') not in allowed or type(answer.get('confirmed')) is not bool:raise ValueError('invalid_dispute')
                result['dispute']=answer
            except Exception as exc:
                answer={'confirmed':False,'attribution':'insufficient_evidence','error':f'{type(exc).__name__}: {exc}'}
                result['dispute']=answer
            write_json(destination/'outcome.json',answer);write_json(out/'result.json',result)

    async def complete(self):
        assert len(list((self.run/'tasks').glob('*/result.json')))==100
        await asyncio.gather(*(self.finish_one(t) for t in self.m['tasks']))
        await asyncio.gather(*(self.dispute_one(t) for t in self.m['tasks']))
        result=self.report()
        write_json(self.run/'final_completion.json',{'at':now(),'terminal_tasks':100,'summary_sha256':file_hash(self.run/'summary.json')})
        return result


if __name__=='__main__':
    study=Finish(original.RUN)
    with (study.run/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(canonical(asyncio.run(study.complete())),flush=True)
