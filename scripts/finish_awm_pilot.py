#!/usr/bin/env python3
"""Finish bounded review/control protocol fixes and refresh cached pilot results."""
import asyncio
import fcntl
import os
from pathlib import Path
from awm_new_pilot import DEFAULT_RUN,Pilot,freeze,gate,report
from repair_awm_pilot_reviews import repair as repair_reviews
from repair_awm_pilot_controls import repair as repair_controls
from harness_awm.io import read_json,write_json


async def finish():
    p=Pilot(DEFAULT_RUN)
    repairs=await asyncio.gather(repair_reviews(DEFAULT_RUN,p),repair_controls(DEFAULT_RUN,p),return_exceptions=True)
    errors=[f'{type(x).__name__}: {x}' for x in repairs if isinstance(x,BaseException)]
    freeze(p.run/'finalization_repairs.json',dict(results=[x for x in repairs if not isinstance(x,BaseException)],errors=errors))
    scenes=read_json(p.data/'scenarios.json')
    for scene in scenes[:2]:
        path=p.run/'worlds'/scene['name']/'result.json'
        saved=path.with_name('result.before_finalization_v1.json')
        if saved.exists():
            raise ValueError('finalization_already_applied')
        path.rename(saved)
    worlds=await asyncio.gather(*(p.world(s) for s in scenes[:2]))
    path=p.run/'pilot_gate.json'
    path.rename(p.run/'pilot_gate.before_finalization_v1.json')
    decision=gate(worlds)
    freeze(path,decision)
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
            asyncio.run(finish())
        finally:
            report(DEFAULT_RUN)
