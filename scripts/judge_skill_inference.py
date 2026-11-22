"""Manually score frozen skill inference later; dry-run unless --execute is given."""
import argparse
import asyncio
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness_awm.backend import Budget
from harness_awm.contracts import RunConfig, TaskSpec
from harness_awm.io import file_hash, read_json, write_json
from harness_awm.judge_inputs import input_from_manifest, prepare_input
from harness_awm.judge_runtime import evaluate_input
from harness_awm.stages import check_manifest


async def run(a):
    manifest = check_manifest(a.run)
    if not a.execute:
        print({'planned': len(manifest['candidates']), 'judge_started': False,
               'note': 'Use --execute --config <judge-config.json> --output <new-directory> when funded.'})
        return
    if not a.config or not a.output or a.output.exists():
        raise ValueError('Execution requires --config and a new --output directory')
    config = RunConfig.model_validate({**read_json(a.config), 'skills': None,
        'purpose': 'evaluation', 'actor': 'training', 'judge': 'api', 'judge_protocol': 'rubric_v2',
        'judge_inputs_manifest': manifest['config']['judge_inputs_manifest']})
    if any(x is not None and x <= 0 for x in (config.max_physical_requests,config.max_total_tokens)):
        raise ValueError('Judge config must specify positive budgets, or both null')
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env', override=False)
    if not os.environ.get(config.judge_api.api_key_env):
        raise ValueError('Missing API credential: '+config.judge_api.api_key_env)
    a.output.mkdir(parents=True)
    write_json(a.output/'source.json',dict(run=str(a.run),manifest_sha256=file_hash(a.run/'manifest.json'),
        config=config.model_dump(),training_updates=False,source_episodes_modified=False))
    budget = Budget(a.output/'budget.sqlite',config.max_physical_requests,config.max_total_tokens)
    semaphore = asyncio.Semaphore(config.judge_concurrency)
    records=[]
    sources=None
    if (a.run/'judge_sources.json').exists():
        if file_hash(a.run/'judge_sources.json') != read_json(a.run/'judge_sources.sha256.json')['sha256']:
            raise ValueError('Frozen Judge sources changed')
        sources=read_json(a.run/'judge_sources.json')
    for c in manifest['candidates']:
        folder=a.run/c['path']
        task=TaskSpec.model_validate(read_json(folder/'task.json'))
        if sources is None:
            bundle=input_from_manifest(folder,task,config.judge_inputs_manifest)
        else:
            source=sources[task.task_instance_id]
            bundle=prepare_input(folder,source['original'],source['rubric'])
        target=a.output/c['rollout_id']
        result=await evaluate_input(bundle,config,budget,semaphore,target)
        records.append(dict(task_instance_id=task.task_instance_id,result=result))
        write_json(a.output/'progress.json',dict(completed=len(records),planned=len(manifest['candidates']),
                                              budget=budget.summary()))
        if 'global_budget_exhausted' in str(result.get('errors',[])):
            break
    confirmed=sum(r['result'].get('evaluation_valid') is True and
                  r['result'].get('task_status')=='confirmed_pass' for r in records)
    write_json(a.output/'summary.json',dict(planned=len(manifest['candidates']),completed=len(records),
        confirmed_successes=confirmed,success_over_planned=confirmed/len(manifest['candidates']),
        scored=sum(r['result'].get('evaluation_valid') is True for r in records),budget=budget.summary()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--config',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    a.run=a.run.resolve()
    asyncio.run(run(a))
