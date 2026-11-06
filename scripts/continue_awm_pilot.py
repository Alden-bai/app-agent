#!/usr/bin/env python3
"""Resume only repaired startup and incomplete reviews; preserve all episodes."""
import asyncio
import fcntl
import os
from pathlib import Path
from awm_new_pilot import DEFAULT_RUN,Pilot,freeze,gate,report
from repair_awm_pilot_reviews import repair
from harness_awm.io import read_json,write_json


def archive(path):
    if not path.exists():
        return
    dest=path.with_name(path.stem+'.before_continuation_v1'+path.suffix)
    if dest.exists():
        raise ValueError('continuation_already_applied:'+str(path))
    path.rename(dest)


async def resume():
    p=Pilot(DEFAULT_RUN)
    scenes=read_json(p.data/'scenarios.json')
    first_result=p.run/'worlds'/scenes[0]['name']/'result.json'
    if first_result.exists():
        old=read_json(first_result)
        proof=read_json(p.data/'worlds'/scenes[0]['name']/'adapter_repair.json')
        if old.get('error')!='ValueError: api_context_limit_exceeded' or not proof['ast_equivalent'] or proof['paid_requests']!=0:
            raise ValueError('startup_repair_not_proven')
        archive(first_result)
    outcomes=await asyncio.gather(p.world(scenes[0]),repair(DEFAULT_RUN,p),return_exceptions=True)
    errors=[f'{type(x).__name__}: {x}' for x in outcomes if isinstance(x,BaseException)]
    if errors:
        write_json(p.run/'continuation_errors.json',errors)
    # A new summary can include newly accepted unchanged rubrics. Existing sealed
    # episodes are cache hits; the Agent is never rerun to improve its score.
    archive(p.run/'worlds'/scenes[1]['name']/'result.json')
    second=await p.world(scenes[1])
    first=read_json(first_result) if first_result.exists() else {'scenario':scenes[0]['name'],'error':'continuation_failed'}
    decision=gate([first,second])
    archive(p.run/'pilot_gate.json')
    freeze(p.run/'pilot_gate.json',decision)
    if decision['passed']:
        for start in range(2,10,2):
            await asyncio.gather(*(p.world(s) for s in scenes[start:start+2]))
            report(p.run)
    print(report(p.run),flush=True)


if __name__=='__main__':
    if not os.environ.get('OPENAI_API_KEY'):
        from dotenv import dotenv_values
        os.environ['OPENAI_API_KEY']=dotenv_values(Path(__file__).resolve().parents[1]/'.env',interpolate=False)['OPENAI_API_KEY']
    with (DEFAULT_RUN/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            asyncio.run(resume())
        finally:
            report(DEFAULT_RUN)
