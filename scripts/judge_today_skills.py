"""Scheduled DeepSeek assessment of only the frozen 2026-11-17 skill run."""
import argparse
import asyncio
from collections import Counter
import csv
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import sys
import time

REPO = Path('/root/awm-agent-train')
sys.path.insert(0, str(Path(__file__).resolve().parent / 'source'))
sys.path.insert(1, str(REPO))
from harness_awm.io import read_json, write_json, file_hash, now
from harness_awm.contracts import RunConfig
from harness_awm.backend import Budget, _api_request_once
from harness_awm.judge_inputs import prepare_input
from harness_awm import judge_runtime
from harness_awm.judge_rate_limit import RateLimit, estimated_tokens
from harness_awm.judge_recovery import request_with_retries


def label(result):
    if not result or not result.get('evaluation_valid'):
        return '无法判定'
    return {'confirmed_pass': '正确', 'failed': '错误'}.get(result.get('task_status'), '无法判定')


def report(root, plan, records, stage, budget=None):
    rows = []
    for c in plan['candidates']:
        r = records.get(c['rollout_id'], {})
        result = r.get('result') or {}
        outcome = label(result) if r else ('推理未完成' if not c['complete'] else '待评分')
        reasons = [{'id': x.get('id', x.get('item_id')), 'status': x.get('status'),
                    'reason': x.get('reason', x.get('rationale', x))}
                   for x in result.get('items', []) if x.get('status') != 'pass']
        rows.append(dict(rollout_id=c['rollout_id'], task_instance_id=c['task_instance_id'],
                         skill_group=c['skill_group'], outcome=outcome,
                         termination_reason=c.get('termination_reason'),
                         result_path=r.get('result_path'), reasons=reasons,
                         errors=result.get('errors', r.get('errors', []))))
    write_json(root/'per_task.json', rows)
    tmp = root/'per_task.csv.tmp'
    with tmp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(root/'per_task.csv')
    groups = {}
    for name in ('all', 'injected', 'workflow', 'reminder_only', 'none'):
        selected = [r for r in rows if name == 'all' or r['skill_group'] == name or
                    (name == 'injected' and r['skill_group'] in ('workflow', 'reminder_only'))]
        counts = Counter(r['outcome'] for r in selected)
        groups[name] = dict(total=len(selected), counts=dict(counts),
                            confirmed_success_over_planned=counts['正确']/len(selected) if selected else None)
    write_json(root/'status.json', dict(at=now(), pid=os.getpid(), stage=stage,
               not_before=plan['not_before'], completed=len(records), planned=plan['eligible'],
               missing=589-plan['eligible'], groups=groups, budget=budget.summary() if budget else None,
               limitation='Only today’s trajectories; no matched no-skill control. Group differences are not causal skill effects.'))


async def execute(root, plan):
    from dotenv import dotenv_values
    settings = dotenv_values('/tmp/awm-deepseek-flash-20261113/workspace/.env')
    settings.update({k:v for k,v in dotenv_values(REPO/'.env').items() if k.startswith('DEEPSEEK_') and v})
    key = settings.get('DEEPSEEK_API_KEY') or os.environ.get('DEEPSEEK_API_KEY')
    if not key:
        raise RuntimeError('DeepSeek credential missing')
    os.environ['DEEPSEEK_API_KEY'] = key
    source = Path(plan['run'])
    for name, expected in plan['input_hashes'].items():
        if file_hash(source/name) != expected:
            raise ValueError('Frozen input changed: '+name)
    sources = read_json(source/'judge_sources.json')
    config = RunConfig.model_validate(plan['config'])
    budget = Budget(root/'budget.sqlite', config.max_physical_requests, config.max_total_tokens)
    limit = RateLimit(root/'rate.sqlite', 1000, 2_000_000)
    semaphore = asyncio.Semaphore(8)
    fatal = asyncio.Event()
    threshold = datetime.fromisoformat(plan['not_before']).timestamp()

    async def send(cfg, request, b, key, stage, folder, **kw):
        if time.time() < threshold:
            raise RuntimeError('API cannot start before scheduled time')
        if fatal.is_set():
            raise RuntimeError('DeepSeek transport halted')
        request = dict(request, extra_body={'thinking': {'type': 'disabled'}})
        try:
            return await _api_request_once(cfg, request, b, key, stage, folder, **kw)
        except Exception as exc:
            if getattr(exc, 'status_code', None) in (400, 401, 402, 403, 404):
                fatal.set()
                write_json(root/'transport_halted.json', dict(at=now(), status_code=exc.status_code,
                           error_type=type(exc).__name__))
            raise

    async def transport(cfg, request, b, key, stage, folder, **kw):
        return await request_with_retries(send, cfg, request, b, key, stage, folder,
                     limit=limit, tokens=estimated_tokens(request), **kw)
    judge_runtime.api_request = transport
    records = {p.stem: read_json(p) for p in (root/'records').glob('*.json')}
    queue = asyncio.Queue()
    for c in plan['candidates']:
        if c['complete'] and c['rollout_id'] not in records:
            queue.put_nowait(c)

    async def worker():
        while not queue.empty() and not fatal.is_set():
            c = queue.get_nowait()
            ident = c['rollout_id']
            folder = source/c['path']
            try:
                for name, expected in c['hashes'].items():
                    if file_hash(folder/name) != expected:
                        raise ValueError('Episode changed: '+name)
                original = sources[c['task_instance_id']]
                bundle = await asyncio.to_thread(prepare_input, folder, original['original'], original['rubric'])
                result = await judge_runtime.evaluate_input(bundle, config, budget, semaphore, root/'assessments'/ident)
                record = dict(result=result, result_path=str(Path(result['attempt'])/'result.json') if result.get('attempt') else None)
            except Exception as exc:
                record = dict(errors=[type(exc).__name__], result=None)
            write_json(root/'records'/f'{ident}.json', record)
            records[ident] = record
            report(root, plan, records, 'scoring', budget)
            print(now(), ident, label(record.get('result')), len(records), flush=True)
    await asyncio.gather(*(worker() for _ in range(8)))
    stage = 'needs_attention' if fatal.is_set() or any(not r.get('result') or
              r['result'].get('evaluation_status') == 'evaluation_error' for r in records.values()) else 'complete'
    report(root, plan, records, stage, budget)
    write_json(root/'summary.json', read_json(root/'status.json'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    root = args.output.resolve()
    with (root/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read_json(root/'plan.json')
        if file_hash(root/'plan.json') != read_json(root/'plan.sha256.json')['sha256']:
            raise ValueError('Plan changed')
        threshold = datetime.fromisoformat(plan['not_before']).timestamp()
        while time.time() < threshold:
            write_json(root/'scheduler.json', dict(pid=os.getpid(), stage='waiting', at=now(),
                       not_before=plan['not_before'], api_requests_started=False))
            time.sleep(min(20, threshold-time.time()))
        write_json(root/'scheduler.json', dict(pid=os.getpid(), stage='started', at=now(), not_before=plan['not_before']))
        try:
            asyncio.run(execute(root, plan))
        except Exception as exc:
            write_json(root/'fatal.json', dict(at=now(), error_type=type(exc).__name__, error=str(exc)))
            raise


if __name__ == '__main__':
    main()
