#!/usr/bin/env python3
"""Complete budget-censored evidence in a NEW capped run; never replay an Actor request."""
import argparse
import asyncio
import fcntl
from pathlib import Path
import sqlite3

import harness_v2_experiment as experiment
from harness_v2_experiment import ROOT, load_credentials, load_manifest
from harness_awm.backend import APIBackend, Budget
from harness_awm.contracts import RunConfig, TaskSpec
from harness_awm.io import canonical, file_hash, read_json, verify_seal, write_json
from harness_awm.judge_inputs import prepare_input
from harness_awm.judge_runtime import evaluate_input
from harness_awm.runner import EpisodeRunner


def verify_source(source):
    m=load_manifest(source)
    for path,expected in m.get('implementation_hashes',{}).items():
        if file_hash(path)==expected: continue
        # Only the driver has a documented reporting-only fix. Actor/Judge runtime
        # must still match the exact executed code, and the old driver is retained.
        archived=source/'implementation_snapshot/harness_v2_experiment.py'
        if Path(path).resolve()!=Path(experiment.__file__).resolve() or not archived.exists() or file_hash(archived)!=expected:
            raise ValueError('source_runtime_changed:'+path)
    return m


def prepare(source, output, max_tokens, max_requests):
    source, output=Path(source).resolve(),Path(output).resolve()
    m=verify_source(source)
    entries=[]
    with sqlite3.connect((source/'budget.sqlite').as_uri()+'?mode=ro',uri=True) as db:
        if db.execute("SELECT count(*) FROM requests WHERE status='in_flight'").fetchone()[0]:
            raise ValueError('source_requests_still_running')
        for arm in m['configs']:
            for t in m['tasks']:
                path=source/arm/t['id']/'result.json'
                if not path.exists(): raise ValueError('source_run_not_finished')
                r=read_json(path); e=r.get('execution')
                if not e: continue
                prefix=arm+'/'+t['id']+'/actor/'
                requests=db.execute('SELECT count(*) FROM requests WHERE substr(request_key,1,?)=?',(len(prefix),prefix)).fetchone()[0]
                episode=path.parent/'episode'
                verify_seal(episode)
                mode=None
                if e['termination_reason']=='global_budget_limit' and requests==0 and e['tool_calls']==0:
                    mode='first_actor_request'
                else:
                    ev=r.get('evaluation',{})
                    errors=[r.get('error') or '',*ev.get('errors',[])]
                    if not ev.get('evaluation_valid') and any('global_budget' in x for x in errors):
                        mode='judge_only'
                if mode:
                    entries.append({'arm':arm,'id':t['id'],'mode':mode,'source_result':str(path),
                        'source_result_hash':file_hash(path),'source_episode':str(episode),
                        'source_actor_requests':requests,'source_tool_calls':e['tool_calls']})
    output.mkdir(parents=True,exist_ok=False)
    manifest={'source_run':str(source),'source_manifest_hash':file_hash(source/'manifest.json'),
        'entries':entries,'max_total_tokens':max_tokens,'max_physical_requests':max_requests,
        'implementation_hash':file_hash(__file__),'driver_hash':file_hash(experiment.__file__),
        'policy':'Only previously undispatched Actor requests may start; all other episodes are Judge-only on the same sealed state. No valid evaluation is replaced. Source run remains unchanged.'}
    write_json(output/'manifest.json',manifest)
    write_json(output/'manifest.sha256.json',{'sha256':file_hash(output/'manifest.json')})
    return manifest


async def run_all(output):
    output=Path(output).resolve(); m=read_json(output/'manifest.json')
    if file_hash(output/'manifest.json')!=read_json(output/'manifest.sha256.json')['sha256'] or file_hash(__file__)!=m['implementation_hash'] or file_hash(experiment.__file__)!=m['driver_hash']:
        raise ValueError('supplement_changed')
    source=Path(m['source_run']); original=verify_source(source)
    if file_hash(source/'manifest.json')!=m['source_manifest_hash']: raise ValueError('source_changed')
    load_credentials(original)
    budget=Budget(output/'budget.sqlite',m['max_physical_requests'],m['max_total_tokens'])
    workers, models, judges=asyncio.Semaphore(4),asyncio.Semaphore(4),asyncio.Semaphore(2)
    judge_config=RunConfig.model_validate(original['judge_config'])

    async def one(entry):
        out=output/entry['arm']/entry['id']
        if (out/'result.json').exists(): return
        async with workers:
            if (out/'started.json').exists(): raise ValueError('interrupted_supplement_not_replayed')
            if file_hash(entry['source_result'])!=entry['source_result_hash']: raise ValueError('source_result_changed')
            old=read_json(entry['source_result']); result=dict(old)
            write_json(out/'started.json',entry)
            try:
                episode=Path(entry['source_episode'])
                if entry['mode']=='first_actor_request':
                    if entry['source_actor_requests'] or entry['source_tool_calls']:
                        raise ValueError('actor_replay_forbidden')
                    config=RunConfig.model_validate(original['configs'][entry['arm']])
                    task=TaskSpec.model_validate(read_json(source/'inputs'/entry['id']/'task.json'))
                    rid='supplement/'+entry['arm']+'/'+entry['id']
                    episode=out/'episode'
                    result={'execution':await EpisodeRunner(config,budget,judges).run(task,episode,rid,
                            APIBackend(config,budget,models,rid),evaluate_after=False)}
                    result['empty_answer']=not read_json(episode/'trajectory.json')['final_answer'].strip()
                result['episode']=str(episode)
                result.pop('error',None)
                if result['execution']['terminal_state_status']=='trusted' and result['execution']['termination_reason']!='cancelled':
                    bundle=prepare_input(episode,read_json(source/'inputs'/entry['id']/'original.json'),
                                         read_json(source/'inputs'/entry['id']/'rubric.json'))
                    result['evaluation']=await evaluate_input(bundle,judge_config,budget,judges,out/'evaluations')
            except Exception as exc:
                result['error']=f'{type(exc).__name__}: {exc}'
            result['supplement_mode']=entry['mode']
            write_json(out/'result.json',result)
            print(canonical({'arm':entry['arm'],'id':entry['id'],'mode':entry['mode'],
                             'status':result.get('evaluation',{}).get('task_status'),'error':result.get('error')}),flush=True)
    try:
        await asyncio.gather(*(one(e) for e in m['entries']))
    finally:
        results=[read_json(output/e['arm']/e['id']/'result.json') for e in m['entries'] if (output/e['arm']/e['id']/'result.json').exists()]
        write_json(output/'summary.json',{'planned':len(m['entries']),'recorded':len(results),'budget':budget.summary()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--source',type=Path)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--max-tokens',type=int,default=0)
    parser.add_argument('--max-requests',type=int,default=0)
    args=parser.parse_args()
    if args.action=='prepare':
        m=prepare(args.source,args.run,args.max_tokens,args.max_requests)
        print(canonical({'entries':m['entries'],'max_total_tokens':m['max_total_tokens']}))
    else:
        with (args.run/'runner.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            asyncio.run(run_all(args.run))
