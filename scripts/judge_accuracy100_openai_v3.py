#!/usr/bin/env python3
"""Qualified SQL and recoverable failed queries; immutable actor/Judge evidence."""
import asyncio
import fcntl
import json
import shutil
from pathlib import Path
import judge_accuracy100 as base
import judge_accuracy100_openai as original
from judge_accuracy100_openai_v2 import strict_sql_evidence
from harness_awm.io import canonical, digest, file_hash, now, read_json, verify_seal, write_json


def query_id_allowed(query, evidence):
    return (isinstance(query.get('id'),str) and isinstance(query.get('sql'),str)
            and (query['id'] not in evidence or evidence[query['id']].get('valid') is False))


class Study(original.OpenAIStudy):
    def __init__(self, run):
        super().__init__(run)
        base.sql_evidence=strict_sql_evidence
        self.m['reference_prompt'] += '\nSQL EXECUTION CONTRACT: Qualify EVERY real table with initial. or final., including JOINs and subqueries. Unqualified reads are rejected. Put exactly ONE SQL statement in each queries entry. You may reuse an ID only to repair a previously invalid query; otherwise use a new ID. Query errors are recoverable within the same reference session and are never business failures.'
        path=self.run/'reference_sql_amendment_v3.json'
        if not path.exists():
            write_json(path,{'at':now(),'implementation':str(Path(__file__).resolve()),'sha256':file_hash(__file__),
                'strict_sql_implementation_sha256':file_hash(Path(__file__).with_name('judge_accuracy100_openai_v2.py')),
                'reference_prompt':self.m['reference_prompt'],
                'query_id_policy':'Failed queries may reuse an ID; successful evidence IDs are immutable. Query history retains each attempt; labels resolve to latest successful query result.',
                'actor_judge_policy':'Reuse every actual actor attempt and first official result. Never rerun business actions or select a preferred Judge response.',
                'posthoc_labels':'Repair invalid reference evidence in fresh contexts that exclude official results. Preserve all earlier labels; report posthoc corrections separately.'})
        else:assert read_json(path)['sha256']==file_hash(__file__)

    async def reference(self, entry, task, episode, out):
        verify_seal(episode)
        inputs=Path(entry['input']);rubric=read_json(inputs/'rubric.json')
        trace=read_json(episode/'trajectory.json');runtime=read_json(episode/'episode_result.json')
        events=[json.loads(l) for l in (episode/'events.jsonl').read_text().splitlines()]
        raw=[{'step':e['turn_id'],'result':read_json(episode/e['raw_result_ref']['path'])}
             for e in events if e['event_type']=='tool_completed']
        audit_trace=[]
        for step in trace['trajectory']:
            item={k:v for k,v in step.items() if k!='tool_response'}
            if 'tool_response' in step:
                item['response_location']=('raw_tool_results for this iteration' if any(t['name']=='call_tool' for t in step.get('tool_calls',[])) else 'tool discovery or observation paging; original preserved in trajectory.json')
            audit_trace.append(item)
        payload={'task':task.instruction,'contract':base.compact_contract(read_json(inputs/'gen_spec.json')),
            'schema':read_json(inputs/'gen_db.json')['db_schema'],'rubric':rubric,
            'trajectory':audit_trace,'raw_tool_results':raw,
            'runtime':{k:runtime[k] for k in ['termination_reason','execution_integrity','terminal_state_status']},
            'actual_start_utc':events[0]['time'],'actual_end_utc':events[-1]['time'],
            'timezone':self.config.timezone,'max_reference_turns':6}
        write_json(out/'reference_input.json',payload)
        evidence={k:{'valid':True} for k in ['task','contract','schema','trajectory','runtime']}
        history=[];candidate=None
        messages=[{'role':'system','content':self.m['reference_prompt']},{'role':'user','content':canonical(payload)}]
        for turn in range(6):
            answer=await self.llm(out/'reference'/f'turn_{turn}','reference',messages)
            messages.append({'role':'assistant','content':canonical(answer)})
            if answer.get('action')=='finish':
                candidate=base.validate_reference(answer,rubric,evidence);break
            if answer.get('action')!='query' or not 1<=len(answer.get('queries',[]))<=12:
                raise ValueError('invalid_reference_query_action')
            results=[]
            for query in answer['queries']:
                if not query_id_allowed(query,evidence):raise ValueError('invalid_query_or_overwrite_of_valid_evidence')
                result=strict_sql_evidence(episode/'initial.db',episode/'final.db',query)
                evidence[query['id']]=result
                history.append({'turn':turn,**result});results.append(result)
            write_json(out/'queries.json',history)
            messages.append({'role':'user','content':canonical({'query_results':results,'turns_remaining':5-turn})})
        if candidate is None:raise ValueError('reference_turn_limit')
        write_json(out/'reference_candidate.json',candidate)
        # The fresh reviewer sees the latest result for each query ID; full
        # failed-query history is retained separately and never presented as state.
        queries=[v for k,v in evidence.items() if k not in {'task','contract','schema','trajectory','runtime'}]
        reviewed=await self.llm(out/'reference_review','reference_review',[
            {'role':'system','content':self.m['review_prompt']},
            {'role':'user','content':canonical({'evidence':payload,'queries':queries,'candidate':candidate})}])
        if reviewed.get('label') not in {'pass','fail','unknown'} or not reviewed.get('reason'):
            raise ValueError('invalid_reference_review')
        ids=reviewed.get('evidence_ids')
        if not isinstance(ids,list) or not ids or not set(ids)<=set(evidence):raise ValueError('review_evidence_invalid')
        label=candidate['task_label'] if candidate['task_label']==reviewed['label'] else 'unknown'
        verify_seal(episode)
        return {'label':label,'frozen_at':now(),'candidate':candidate,'review':reviewed,
            'rubric_sha256':file_hash(inputs/'rubric.json'),'evidence_seal_sha256':file_hash(episode/'evidence_seal.json'),
            'queries_sha256':digest(history),'blind_to_official':True,'reference_sql_revision':3,
            'reference_type':'rubric_and_evidence_based_two_pass_gpt6_luna_reference_not_human_gold'}

    def needs_repair(self,out):
        result=read_json(out/'result.json')
        if result.get('reference_sql_revision')==3:return False
        p=Path(result.get('active_reference',out/'reference_frozen.json'))
        if not p.exists():return False
        ref=read_json(p)
        if ref.get('reference_sql_revision')==3:return False
        if any(k in ref.get('error','') for k in ['duplicate_query_id','invalid_query']):return True
        if (out/'queries.json').exists():
            for q in read_json(out/'queries.json'):
                if q.get('valid') and 'Unqualified table reference' in strict_sql_evidence(out/'episode/initial.db',out/'episode/final.db',q).get('error',''):
                    return True
        return False

    async def task(self,entry):
        result=await super().task(entry)
        out=self.run/'tasks'/entry['id']
        if not self.needs_repair(out):return result
        async with self.env_sem:
            history=out/'assessment_history/before_v3';history.mkdir(parents=True,exist_ok=True)
            if not (history/'result.json').exists():shutil.copyfile(out/'result.json',history/'result.json')
            repair=out/'reference_sql_v3';frozen=repair/'reference_frozen.json'
            if not frozen.exists():
                try:
                    task=base.TaskSpec.model_validate(read_json(out/'task_spec.json'))
                    reference=await self.reference(entry,task,out/'episode',repair)
                except Exception as exc:reference={'label':'unknown','error':f'{type(exc).__name__}: {exc}'}
                reference.update(frozen_at=now(),blind_to_official=True,posthoc_reference_repair=True,reference_sql_revision=3)
                write_json(frozen,reference)
            reference=read_json(frozen)
            result.update(reference_label=reference['label'],reference_hash=file_hash(frozen),active_reference=str(frozen),
                reference_sql_revision=3,reference_repaired_after_judge=(out/'official_evaluation.json').exists())
            for key in ['agreement','dispute','dispute_error']:result.pop(key,None)
            if reference['label'] in {'pass','fail'} and result['judge_valid']:
                result['agreement']=(reference['label']=='pass')==result['official_pass']
            write_json(out/'result.json',result)
            print(canonical({'reference_repaired':entry['task_id'],'label':reference['label'],'official_pass':result['official_pass']}),flush=True)
            self.report();return result

    def report(self):
        report=super().report()
        records=[read_json(p) for p in (self.run/'tasks').glob('*/result.json')]
        report['posthoc_reference_repair_count']=sum(r.get('reference_repaired_after_judge',False) for r in records)
        report['pending_reference_sql_repairs']=[r['task_id'] for r in records if self.needs_repair(self.run/'tasks'/r['id'])]
        report['limitations'].append('Reference SQL revisions 2/3 repair ambiguous initial/final queries and failed-query ID reuse. Posthoc reference repairs remain model-blind to official results; original Judge outputs are reused.')
        write_json(self.run/'summary.json',report)
        return report

    async def run_all(self):
        # Previously started tasks settle or repair first; new tasks follow only after this checkpoint.
        await asyncio.gather(*(self.task(t) for t in self.m['tasks'][:23]))
        print(canonical({'reference_repair_checkpoint':self.report()['comparable']}),flush=True)
        await asyncio.gather(*(self.task(t) for t in self.m['tasks'][23:]))
        report=self.report()
        write_json(self.run/'completion.json',{'at':now(),'terminal_tasks':report['terminal_tasks'],'summary_sha256':file_hash(self.run/'summary.json'),'reference_sql_revision':3})
        return report


if __name__=='__main__':
    study=Study(original.RUN)
    with (study.run/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(canonical(asyncio.run(study.run_all())),flush=True)
