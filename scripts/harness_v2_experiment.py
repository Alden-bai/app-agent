#!/usr/bin/env python3
"""Paired v2 replay from frozen initial states; no retries of started episodes."""
import argparse
import asyncio
from collections import Counter
import fcntl
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness_awm.backend import APIBackend, Budget
from harness_awm.contracts import RunConfig, TaskSpec
from harness_awm.io import canonical, file_hash, read_json, verify_seal, write_json
from harness_awm.judge_inputs import prepare_input
from harness_awm.judge_runtime import evaluate_input
from harness_awm.protocol_v2 import PROTOCOL, SYSTEM_PROMPT
from harness_awm.runner import EpisodeRunner

PILOT = ['t000', 't011', 't018', 't024', 't044', 't049', 't054', 't066', 't079', 't097']
DEFAULT_SOURCE = ROOT/'runs/openai_gpt6_luna_judge_accuracy100_20261103'
DEFAULT_JUDGE = ROOT/'runs/judge_v2_regression100_20261104_v4'


def sources():
    return [*sorted((ROOT/'harness_awm').glob('*.py')), Path(__file__).resolve()]


def prepare(source, judge, run, max_requests, max_tokens):
    source, judge, run = map(lambda p: Path(p).resolve(), (source, judge, run))
    for root in (source, judge):
        if file_hash(root/'manifest.json') != read_json(root/'manifest.sha256.json')['sha256']:
            raise ValueError('source_manifest_changed')
    sm, jm = read_json(source/'manifest.json'), read_json(judge/'manifest.json')
    run.mkdir(parents=True, exist_ok=False)
    old = RunConfig.model_validate(sm['config']).model_dump()
    old.update(protocol_id=PROTOCOL, max_physical_requests=max_requests, max_total_tokens=max_tokens,
               observation_chars=8000, max_read_chars=4000)
    expanded = read_json(ROOT/'configs/harness/v2-expanded.json')
    expanded.pop('max_physical_requests'); expanded.pop('max_total_tokens')
    configs = {'original_budget': RunConfig.model_validate(old).model_dump(),
               'expanded_budget': RunConfig.model_validate({**old, **expanded}).model_dump()}
    tasks = []
    for entry in sm['tasks']:
        tid = entry['id']
        dest = run/'inputs'/tid
        dest.mkdir(parents=True)
        result = {'id': tid, 'task_id': entry['task_id'], 'pilot': tid in PILOT,
                  'baseline': read_json(judge/'cases'/tid/'result.json')}
        try:
            ep = source/'tasks'/tid/'episode'
            verify_seal(ep)
            task = TaskSpec.model_validate(read_json(ep/'task.json'))
            if file_hash(ep/'initial.db') != task.initial_state_hash:
                raise ValueError('frozen_initial_mismatch')
            for original, name in [(ep/'initial.db', 'initial.db'), (Path(task.environment_ref), 'environment.json'),
                                   (Path(task.verifier_ref), 'verifier.json'),
                                   (judge/'cases'/tid/'original.json', 'original.json'),
                                   (judge/'cases'/tid/'rubric.json', 'rubric.json')]:
                shutil.copyfile(original, dest/name)
            task = task.model_copy(update={'initial_state_ref': str(dest/'initial.db'),
                'environment_ref': str(dest/'environment.json'), 'verifier_ref': str(dest/'verifier.json'),
                'protocol_id': PROTOCOL})
            write_json(dest/'task.json', task.model_dump())
            result['baseline_execution'] = read_json(ep/'episode_result.json')
            result['baseline_empty_answer'] = not read_json(ep/'trajectory.json')['final_answer'].strip()
            result['ready'] = True
        except (OSError, ValueError, KeyError) as exc:
            result.update(ready=False, preflight_error=f'{type(exc).__name__}: {exc}')
        tasks.append(result)
    manifest = {'schema_version':'awm-harness-v2-experiment', 'source':str(source), 'judge_source':str(judge),
        'source_manifest_sha256':file_hash(source/'manifest.json'), 'judge_manifest_sha256':file_hash(judge/'manifest.json'),
        'configs':configs, 'judge_config':jm['config'], 'actor_system_prompt':SYSTEM_PROMPT,
        'max_physical_requests':max_requests, 'max_total_tokens':max_tokens, 'tasks':tasks,
        'implementation_hashes':{str(p):file_hash(p) for p in sources()},
        'input_hashes':{str(p.relative_to(run)):file_hash(p) for p in sorted((run/'inputs').rglob('*')) if p.is_file()},
        'policy':'one new episode per arm/task; no replay of interrupted writes; one shared capped budget across both arms and judges; historical labels never sent to Actor or Judge'}
    write_json(run/'manifest.json', manifest)
    write_json(run/'manifest.sha256.json', {'sha256':file_hash(run/'manifest.json')})
    return report(run)


def load_manifest(run, verify_code=False):
    m = read_json(run/'manifest.json')
    if file_hash(run/'manifest.json') != read_json(run/'manifest.sha256.json')['sha256']:
        raise ValueError('experiment_manifest_changed')
    for rel, expected in m['input_hashes'].items():
        if file_hash(run/rel) != expected:
            raise ValueError('frozen_input_changed:' + rel)
    if verify_code:
        for path, expected in m['implementation_hashes'].items():
            if file_hash(path) != expected:
                raise ValueError('implementation_changed_prepare_new_run:' + path)
    return m


def report(run):
    run = Path(run).resolve()
    m = load_manifest(run)
    result = {'planned_tasks':len(m['tasks']), 'frozen_ready':sum(t['ready'] for t in m['tasks']),
              'preflight_blocked':[{'id':t['id'],'reason':t['preflight_error']} for t in m['tasks'] if not t['ready']], 'arms':{},
              'limitations':['Historical Judge labels are not human gold.',
                             'Actual wall clock is recorded, not frozen; relative-time tasks can differ.',
                             'Model sampling is not deterministic.',
                             'A final answer after a resource stop is not proof of business completion.']}
    for arm in m['configs']:
        records = []
        for t in m['tasks']:
            path = run/arm/t['id']/'result.json'
            if path.exists():
                r = read_json(path)
                e = r.get('execution') or {}
                records.append({'id':t['id'], 'baseline_status':t['baseline']['task_status'],
                    'baseline_zero_calls': t.get('baseline_execution',{}).get('tool_calls') == 0,
                    'baseline_empty_answer':t.get('baseline_empty_answer'),
                    'new_status':r.get('evaluation',{}).get('task_status', 'undetermined'),
                    'termination':e.get('termination_reason'), 'tool_calls':e.get('tool_calls'),
                    'empty_answer':r.get('empty_answer'), 'error':r.get('error'),
                    'metrics':e.get('execution_metrics',{})})
        failed = [r for r in records if r['baseline_status']=='failed']
        result['arms'][arm] = {'recorded':len(records),
            'statuses':dict(Counter(r['new_status'] for r in records)),
            'terminations':dict(Counter(r['termination'] or 'not_executed' for r in records)),
            'finalization_triggers':dict(Counter(r['metrics']['finalization_trigger'] for r in records if r['metrics'].get('finalization_trigger'))),
            'original_failed_zero_calls':sum(r['tool_calls']==0 for r in failed),
            'original_failed_empty_answers':sum(r['empty_answer'] is True for r in failed),
            'regressions':[r['id'] for r in records if r['baseline_status']=='confirmed_pass' and r['new_status']!='confirmed_pass'],
            'intersections':dict(Counter(f"zero={r['tool_calls']==0},empty={r['empty_answer']},termination={r['termination']}" for r in records)),
            'tasks':records}
    if (run/'budget.sqlite').exists():
        result['budget'] = Budget(run/'budget.sqlite', m['max_physical_requests'], m['max_total_tokens']).summary()
    write_json(run/'summary.json', result)
    return result


def load_credentials(m):
    from dotenv import dotenv_values
    values = dotenv_values(ROOT/'.env', interpolate=False)
    # The original study explicitly uses this workspace credential source.
    for config in [*m['configs'].values(), m['judge_config']]:
        for name in ('actor_api','judge_api'):
            api = config[name]
            env = api['api_key_env']
            if not os.environ.get(env):
                value = values.get(env) or (values.get('OPENAI_API_KEY') if env == 'AWM_OPENAI_STUDY_KEY' else None)
                if value:
                    os.environ[env] = value
    for config in [m['configs']['original_budget']['actor_api'],m['judge_config']['judge_api']]:
        if not os.environ.get(config['api_key_env']):
            raise ValueError('missing_experiment_credential:' + config['api_key_env'])


async def run_all(run, phase):
    run = Path(run).resolve()
    m = load_manifest(run, verify_code=True)
    load_credentials(m)
    budget = Budget(run/'budget.sqlite', m['max_physical_requests'], m['max_total_tokens'])
    model_sem, judge_sem, workers = asyncio.Semaphore(4), asyncio.Semaphore(2), asyncio.Semaphore(4)
    judge_config = RunConfig.model_validate(m['judge_config'])
    selected = [t for t in m['tasks'] if phase=='all' or t['pilot']]

    async def one(arm, t):
        out = run/arm/t['id']
        if (out/'result.json').exists():
            return
        async with workers:
            if not t['ready']:
                write_json(out/'result.json', {'error':t['preflight_error'], 'status':'preflight_blocked'})
                return
            # A started episode is never replayed, even after controller interruption.
            if (out/'started.json').exists():
                write_json(out/'result.json', {'error':'interrupted_attempt_not_replayed', 'status':'outcome_unknown'})
                return
            write_json(out/'started.json', {'arm':arm,'id':t['id']})
            config = RunConfig.model_validate(m['configs'][arm])
            task = TaskSpec.model_validate(read_json(run/'inputs'/t['id']/'task.json'))
            rid = arm + '/' + t['id']
            try:
                backend = APIBackend(config, budget, model_sem, rid)
                execution = await EpisodeRunner(config, budget, judge_sem).run(task, out/'episode', rid, backend, evaluate_after=False)
                trace = read_json(out/'episode/trajectory.json')
                result = {'execution':execution,'empty_answer':not trace['final_answer'].strip()}
                write_json(out/'result.json', result)
                if execution['terminal_state_status']=='trusted' and execution['termination_reason']!='cancelled':
                    bundle = prepare_input(out/'episode', read_json(run/'inputs'/t['id']/'original.json'),
                                           read_json(run/'inputs'/t['id']/'rubric.json'))
                    write_json(out/'judge_input.json', bundle)
                    result['evaluation'] = await evaluate_input(bundle, judge_config, budget, judge_sem, out/'evaluations')
            except Exception as exc:
                result = locals().get('result', {})
                result['error'] = f'{type(exc).__name__}: {exc}'
            write_json(out/'result.json', result)
            print(canonical({'arm':arm,'id':t['id'],'termination':result.get('execution',{}).get('termination_reason'),
                             'calls':result.get('execution',{}).get('tool_calls'),
                             'status':result.get('evaluation',{}).get('task_status'), 'error':result.get('error')}), flush=True)
    try:
        # Paired tasks alternate arms in the same scheduling and budget regime.
        await asyncio.gather(*(one(arm,t) for t in selected for arm in m['configs']))
    finally:
        data = report(run)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','run','report'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--judge-source', type=Path, default=DEFAULT_JUDGE)
    parser.add_argument('--max-requests', type=int, default=6000)
    parser.add_argument('--max-tokens', type=int, default=15000000)
    parser.add_argument('--phase', choices=['pilot','all'], default='pilot')
    args = parser.parse_args()
    if args.action=='prepare':
        result = prepare(args.source,args.judge_source,args.run,args.max_requests,args.max_tokens)
    elif args.action=='report':
        result = report(args.run)
    else:
        with (args.run/'runner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            result = asyncio.run(run_all(args.run,args.phase))
    print(canonical({k:v for k,v in result.items() if k!='arms'}), flush=True)


if __name__=='__main__': main()
