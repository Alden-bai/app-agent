#!/usr/bin/env python3
"""Complete truncated review responses without regenerating any rubric.

No semantic rejection is retried. Only a recorded length termination at the
final review stage qualifies; the compiled draft and source remain unchanged.
"""
import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from awm_new_pilot import DEFAULT_RUN,Pilot,BOUNDARIES,d1,freeze
from harness_awm.io import read_json


async def repair(run,pilot=None):
    pilot=pilot or Pilot(run)
    completed=[]
    for path in sorted(pilot.data.glob('worlds/*/rubrics/*/result.json')):
        old=read_json(path)
        error=old.get('feedback',{}).get('validation_error','')
        if not error.startswith('incomplete_generation:rubric_review/'):
            continue
        key=error.split(':',1)[1]
        if '/completion_repair' in key:
            continue
        response=read_json(pilot.run/'requests'/key/'response.json')
        if response['choices'][0]['finish_reason']!='length':
            raise ValueError('review_not_truncated')
        folder=path.parent
        revision=int(key.rsplit('/r',1)[1])
        inp=read_json(folder/'original.json')
        selected=read_json(folder/f'selection.r{revision}.json')
        _,pack=d1.base.prepare_input(inp)
        compiled=d1.compile_rubric(selected,inp,pack)
        request=d1.review_request(inp,selected)
        request['messages'][0]['content']+='\n'+BOUNDARIES.read_text()
        # Canonical storage changes object key order. Reconstruct and compare
        # semantic input, then replay the exact frozen wire representation.
        identity=read_json(pilot.run/'requests'/key/'identity.json')
        messages=identity['messages']
        if messages[0]!=request['messages'][0] or json.loads(messages[1]['content'])!=json.loads(request['messages'][1]['content']):
            raise ValueError('review_source_or_draft_changed')
        review=await pilot.request(key,messages,limit=identity['limit'])
        validation=d1.base.evaluate(review,inp['task_id'],{r['id'] for r in compiled['rubric']},
            {r['candidate_id'] for r in pack['candidates']})
        freeze(folder/f'review.r{revision}.completed.json',dict(response=review,validation=validation))
        old_path=folder/'result.before_review_completion.json'
        if old_path.exists():
            raise ValueError('review_completion_already_installed')
        path.rename(old_path)
        if validation['status']=='passed':
            freeze(folder/'rubric.json',compiled)
            freeze(path,dict(status='passed',revision=revision,scores=validation['scores'],
                            unresolved=len(compiled['unresolved']),review_completion_repair=True))
        else:
            freeze(path,dict(status=validation['status'],feedback=dict(review=review,validation=validation),
                            review_completion_repair=True))
        completed.append(dict(task=inp['task_id'],status=validation['status']))
    print(completed,flush=True)
    return completed


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN)
    a=parser.parse_args()
    if not os.environ.get('OPENAI_API_KEY'):
        from dotenv import dotenv_values
        os.environ['OPENAI_API_KEY']=dotenv_values(Path(__file__).resolve().parents[1]/'.env',interpolate=False)['OPENAI_API_KEY']
    with (a.run/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        asyncio.run(repair(a.run))
