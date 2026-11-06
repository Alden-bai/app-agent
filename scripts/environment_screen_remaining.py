#!/usr/bin/env python3
"""Durable per-task Luna environment screening, extending the pilot MCP probe path."""
import argparse
import ast
import asyncio
from collections import Counter, defaultdict
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from harness_awm.io import db_fingerprint, digest, file_hash, now, read_json, write_json
from environment_screen_refs import resolve_ref
from screen_bank_environments import orm_checks, static_routes

DEFAULT = ROOT / 'runs/environment_screen_remaining5909_20261104_v1'
PILOT = ROOT / 'runs/environment_screen_luna100_20261104_v1'
RELEASE = ROOT / 'data/awm-dde80a0-repaired-v1'
BANK = ROOT / 'runs/rubric_luna_registry/bank_manifest.json'
EXECUTION = ROOT / 'runs/bank3000_execution_20261104_v1'
PYTHON = ROOT / 'agent-world-model/.venv/bin/python'
MODEL, EFFORT, WORKERS, SEED = 'gpt-6-luna', 'high', 12, 2026110401
FIELDS = ('input_tokens', 'cached_input_tokens', 'output_tokens', 'reasoning_output_tokens')


def payload(response):
    if response.get('structuredContent') is not None:
        return response['structuredContent']
    texts = [c.get('text', '') for c in response.get('content', []) if c.get('type') == 'text']
    if len(texts) == 1:
        try:
            return json.loads(texts[0])
        except ValueError:
            return texts[0]
    return texts


def resolve(value, results):
    if isinstance(value, dict):
        if '$result' in value:
            result = results[value['$result']]
            for key in value.get('path', []):
                result = result[key]
            return result
        return {k: resolve(v, results) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, results) for v in value]
    return value


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def jsonl(path, values):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as out:
        for v in values:
            out.write(json.dumps(v, ensure_ascii=False, sort_keys=True) + '\n')
    os.replace(tmp, path)


def exclusive_json(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n')


def check_hashes(mapping):
    for name, sha in mapping.items():
        if file_hash(name) != sha:
            raise ValueError('frozen input changed: ' + str(name))


def db_open(run):
    db = sqlite3.connect(run / 'queue.sqlite', timeout=30)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    return db


def add_job(db, cid, kind):
    db.execute('INSERT OR IGNORE INTO jobs(id,case_id,kind,status,attempts) VALUES(?,?,?,?,0)',
               (cid + '-' + kind, cid, kind, 'pending'))


def implementation_files():
    return [Path(__file__).resolve(), ROOT / 'scripts/environment_screen_pilot100.py',
            ROOT / 'scripts/screen_bank_environments.py',
            ROOT / 'scripts/environment_screen_refs.py',
            *sorted((ROOT / 'harness_awm').glob('*.py')),
            *sorted((ROOT / 'agent-world-model/awm').rglob('*.py'))]


def freeze_case(folder, cid, q, rubric, env, spec, schema, initial):
    folder.mkdir(parents=True)
    write_json(folder / 'task.json', dict(case_id=cid, task_id=q['task_id'], scenario=q['scenario'],
        index=q['index'], task=q['task'], runtime_clock_policy='actual_recorded',
        purpose='environment_verification_not_actor_score', rubric_accepted=True))
    write_json(folder / 'rubric.json', rubric)
    write_json(folder / 'environment.json', env)
    (folder / 'environment.py').write_text(env['full_code'])
    write_json(folder / 'contract.json', spec)
    write_json(folder / 'schema.json', schema)
    shutil.copyfile(initial, folder / 'initial.db')
    write_json(folder / 'asset_hashes.json', {p.name: file_hash(p) for p in folder.iterdir()})
    for p in folder.iterdir():
        p.chmod(0o444)


def prepare(run, instruction):
    if run.exists():
        raise ValueError('new directory required; resume existing runs with run/report')
    bank = read_json(BANK)
    check_hashes({v['path']: v['sha256'] for v in bank['exports'].values()})
    accepted = rows(bank['exports']['accepted_rubrics']['path'])
    questions = rows(bank['exports']['question_bank']['path'])
    executed = read_json(EXECUTION / 'manifest.json')['tasks']
    for name, values in [('accepted', accepted), ('questions', questions), ('executed', executed)]:
        if len({r['task_id'] for r in values}) != len(values):
            raise ValueError('duplicate task ids: ' + name)
    accepted_map = {r['task_id']: r for r in accepted}
    executed_ids = {r['task_id'] for r in executed}
    if executed_ids - accepted_map.keys():
        raise ValueError('executed ids absent from accepted rubric export')
    # Only execution status metadata is read, never trajectories or attribution.
    summary = read_json(EXECUTION / 'summary.json')
    if not summary['complete'] or summary['completed'] != len(executed):
        raise ValueError('execution list is not fully executed; reconcile before freezing')
    remaining = sorted(accepted_map.keys() - executed_ids)
    release = read_json(RELEASE / 'manifest.json')
    check_hashes({str(RELEASE / n): h for n, h in release['output_sha256'].items()})
    assets = {name: {r['scenario']: r for r in rows(RELEASE / (name + '.jsonl'))}
              for name in ['gen_envs', 'gen_spec', 'gen_db', 'gen_tasks']}
    qmap = {r['task_id']: r for r in questions}
    run.mkdir(parents=True)
    (run / 'private').mkdir()
    shutil.copyfile(instruction, run / 'orchestrator_instructions_zh.txt')
    base = ROOT / 'configs/environment_screen/pilot100_v2/prompt_zh.txt'
    supplement = ROOT / 'configs/environment_screen/remaining_v1/supplement_zh.txt'
    # Remove pilot-specific output ownership; the immutable prefix otherwise retains its standards.
    text = base.read_text().replace('首次输出同时保存 decision.first_pass.json 和 decision.json；首次文件不得覆盖。',
                                  '输出保存 decision.json；首次结论由调度器不可覆盖地封存。')
    (run / 'prompt_zh.txt').write_text(text + '\n' + supplement.read_text())
    snapshot = run / 'implementation_snapshot'
    for p in implementation_files():
        target = snapshot / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    sources = {str(BANK): file_hash(BANK), str(EXECUTION / 'manifest.json'): file_hash(EXECUTION / 'manifest.json'),
               str(RELEASE / 'manifest.json'): file_hash(RELEASE / 'manifest.json'),
               str(base): file_hash(base), str(supplement): file_hash(supplement),
               **{v['path']: v['sha256'] for v in bank['exports'].values()},
               **{str(RELEASE / n): h for n, h in release['output_sha256'].items()}}
    inventory = []
    with db_open(run) as db:
        db.executescript('''CREATE TABLE jobs(id TEXT PRIMARY KEY, case_id TEXT, kind TEXT, status TEXT,
          attempts INTEGER, result_path TEXT, error TEXT, started_at TEXT, finished_at TEXT);
          CREATE TABLE tasks(case_id TEXT PRIMARY KEY, task_id TEXT UNIQUE, scenario TEXT,
          status TEXT, first_ref TEXT, final_ref TEXT, reason TEXT);
          CREATE TABLE events(seq INTEGER PRIMARY KEY, at TEXT, job_id TEXT, event TEXT, detail TEXT);''')
        for i, tid in enumerate(remaining):
            q, a = qmap[tid], accepted_map[tid]
            check_hashes({q['original_ref']: q['original_sha256'], a['source_artifact_ref']: a['source_artifact_sha256']})
            sources[q['original_ref']] = q['original_sha256']
            sources[a['source_artifact_ref']] = a['source_artifact_sha256']
            original = read_json(q['original_ref'])
            if original['task_id'] != tid or original['task'] != q['task'] or not q['passed']:
                raise ValueError('original/rubric binding mismatch: ' + tid)
            scenario = q['scenario']
            if assets['gen_tasks'][scenario]['tasks'][q['index']] != q['task']:
                raise ValueError('release task binding mismatch: ' + tid)
            cid = f'T{i:05d}'
            freeze_case(run / 'inputs' / cid, cid, q, a['rubric'], assets['gen_envs'][scenario],
                        assets['gen_spec'][scenario], assets['gen_db'][scenario], RELEASE / 'databases' / (scenario + '.db'))
            inventory.append(dict(case_id=cid, task_id=tid, scenario=scenario,
                rubric_accepted=True, original_ref=q['original_ref'], original_sha256=q['original_sha256'],
                rubric_source_ref=a['source_artifact_ref'], rubric_source_sha256=a['source_artifact_sha256'],
                asset_hashes_sha256=file_hash(run / 'inputs' / cid / 'asset_hashes.json')))
            db.execute('INSERT INTO tasks VALUES(?,?,?,?,NULL,NULL,NULL)', (cid, tid, scenario, 'pending'))
            add_job(db, cid, 'initial')
        # Regression inputs contain only frozen materials; no pilot conclusions or labels.
        changed = read_json(PILOT / 'summary.json')['review_changed_cases']
        regression = []
        for old in changed:
            source = PILOT / 'cases' / old
            task = read_json(source / 'task.json')
            tid, scenario = task['task_id'], task['scenario']
            cid = 'R' + old[1:]
            freeze_case(run / 'inputs' / cid, cid, qmap[tid], accepted_map[tid]['rubric'],
                        assets['gen_envs'][scenario], assets['gen_spec'][scenario], assets['gen_db'][scenario],
                        RELEASE / 'databases' / (scenario + '.db'))
            regression.append(dict(case_id=cid, pilot_case_id=old, task_id=tid))
            add_job(db, cid, 'regression')
    jsonl(run / 'inventory.jsonl', inventory)
    write_json(run / 'regression_inventory.json', regression)
    sources.update({str(RELEASE / 'databases' / (s + '.db')): file_hash(RELEASE / 'databases' / (s + '.db'))
                    for s in {r['scenario'] for r in inventory} | {read_json(run / 'inputs' / r['case_id'] / 'task.json')['scenario'] for r in regression}})
    manifest = dict(version='remaining-environment-screen-v1', created_at=now(),
        accepted_count=len(accepted), executed_count=len(executed), frozen_count=len(remaining),
        scenario_count=len({r['scenario'] for r in inventory}), regression_count=len(regression),
        inventory_sha256=file_hash(run / 'inventory.jsonl'), regression_sha256=file_hash(run / 'regression_inventory.json'),
        model=MODEL, reasoning_effort=EFFORT, concurrency=WORKERS, sampling_seed=SEED,
        prompt_sha256=file_hash(run / 'prompt_zh.txt'), instructions_sha256=file_hash(run / 'orchestrator_instructions_zh.txt'),
        implementation_hashes={str(p): file_hash(p) for p in implementation_files()}, source_hashes=sources,
        codex_version=subprocess.check_output(['codex', '--version'], text=True).strip(),
        master_thread_id='01a1a6e0-6d00-7d63-8975-9210569a1458',
        usage_policy='response_id deduplication; input includes cache, output includes reasoning; unknown is null',
        limits='Finite task-relevant verification; no target label distribution, no savings claim without matched baseline.')
    write_json(run / 'manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ['accepted_count', 'executed_count', 'frozen_count', 'scenario_count', 'regression_count']}), flush=True)
    report(run)


def verify_run(run):
    m = read_json(run / 'manifest.json')
    check_hashes(m['source_hashes'])
    check_hashes(m['implementation_hashes'])
    check_hashes({str(run / 'inventory.jsonl'): m['inventory_sha256'],
                  str(run / 'regression_inventory.json'): m['regression_sha256'],
                  str(run / 'prompt_zh.txt'): m['prompt_sha256']})
    for r in rows(run / 'inventory.jsonl'):
        check_hashes({str(run / 'inputs' / r['case_id'] / 'asset_hashes.json'): r['asset_hashes_sha256']})
    return m


def task_spec(case):
    task = read_json(case / 'task.json')
    hashes = read_json(case / 'asset_hashes.json')
    check_hashes({str(case / n): h for n, h in hashes.items()})
    return SimpleNamespace(scenario=task['scenario'], initial_state_ref=str(case / 'initial.db'),
        initial_state_hash=hashes['initial.db'], initial_state_fingerprint=db_fingerprint(case / 'initial.db'),
        environment_ref=str(case / 'environment.json'), environment_code_hash=digest(read_json(case / 'environment.json')),
        tool_schema_hash=None, readiness_profile={'runtime_ready': True})


def config():
    from harness_awm.contracts import RunConfig
    return RunConfig(startup_timeout=45, tool_timeout=20, stop_timeout=3)


def is_quality(kind):
    return kind in {'exclude_review', 'pass_audit'} or kind.startswith('regression_quality')


def validate_plan(plan):
    if not isinstance(plan, dict) or not isinstance(plan.get('calls'), list) or not plan['calls']:
        raise ValueError('plan.calls must be a nonempty list of {tool,arguments} objects')
    for call in plan['calls']:
        if not isinstance(call, dict) or not isinstance(call.get('tool'), str) or not isinstance(call.get('arguments', {}), dict):
            raise ValueError('each call must have a tool string and arguments object')
    for name in ['before_queries', 'queries']:
        if not isinstance(plan.get(name, []), list):
            raise ValueError(name + ' must be a list of {sql,params} objects')
        for q in plan.get(name, []):
            if not isinstance(q, dict) or not isinstance(q.get('sql'), str) or not isinstance(q.get('params', []), list):
                raise ValueError(name + ': each query must be an object with sql string and params array')


def copy_raw_evidence(source, target):
    for name in ['probes', 'diagnostics', 'evidence']:
        if (source / name).exists():
            shutil.copytree(source / name, target / name, ignore=shutil.ignore_patterns(
                'decision*.json', 'independent*.json', 'review_note*.json', 'state.json'))


async def baseline(case, target):
    from harness_awm.environment import AWMEnvAdapter
    target.mkdir(parents=True)
    task = task_spec(case)
    result = dict(started_at=now(), scenario=task.scenario,
        database_sha256=task.initial_state_hash, environment_sha256=task.environment_code_hash,
        database_logical_fingerprint=task.initial_state_fingerprint,
        runner_hashes={str(p): file_hash(p) for p in implementation_files()},
        scope='database integrity/FK, ORM dates, route analysis, official MCP startup, complete tool catalog and schema validation; no task conclusions',
        parameters=dict(startup_timeout=45, tool_timeout=20, timezone=config().timezone),
        mutable_state_policy='fresh baseline DB, never used for business probes', mcp_ready=False)
    try:
        with sqlite3.connect((case / 'initial.db').as_uri() + '?mode=ro', uri=True) as db:
            result['integrity'] = db.execute('PRAGMA integrity_check').fetchall()
            result['foreign_keys'] = db.execute('PRAGMA foreign_key_check').fetchall()
            tree = ast.parse((case / 'environment.py').read_text())
            result['orm'] = orm_checks(tree, db)
            routes, defects = static_routes(tree)
            write_json(target / 'static.json', dict(routes=routes, route_defects=defects))
    except Exception as exc:
        result['static_error'] = repr(exc)
    adapter = AWMEnvAdapter(task, config(), target / 'mcp')
    try:
        await adapter.prepare()
        result.update(mcp_ready=True, tool_count=len(adapter.tools), tool_catalog_hash=digest(adapter.tools))
        write_json(target / 'tool_catalog.json', dict(tools=adapter.tools))
    except Exception as exc:
        result['mcp_error'] = repr(exc)
        import traceback
        result['mcp_traceback'] = traceback.format_exc()
    finally:
        # Preserve the actual complete catalog even if its input schema is invalid.
        if adapter.tools:
            write_json(target / 'tool_catalog.json', dict(tools=adapter.tools,
                schema_validation_passed=result['mcp_ready']))
        result['stopped'] = await adapter.close()
    result['finished_at'] = now()
    result['initial_unchanged'] = file_hash(case / 'initial.db') == task.initial_state_hash
    write_json(target / 'facts.json', result)


async def provision(run, job, attempt):
    cid = job['case_id']
    frozen = run / 'inputs' / cid
    # Review paths must not reveal whether the initial judgment was pass or exclude.
    public_kind = 'quality' if is_quality(job['kind']) else job['kind']
    case = run / 'work' / (cid + '-' + public_kind) / f'attempt-{attempt:02d}'
    shutil.copytree(frozen, case)
    task = read_json(case / 'task.json')
    # One baseline builder per environment; all copies are evidence-only, never a shared mutable DB.
    frozen_hashes = read_json(case / 'asset_hashes.json')
    baseline_key = digest(dict(inputs={n: frozen_hashes[n] for n in
        ['initial.db', 'environment.json', 'environment.py', 'contract.json', 'schema.json']},
        implementations={str(p): file_hash(p) for p in implementation_files()},
        config=config().model_dump(mode='json')))
    shared = run / 'shared_baseline' / baseline_key / task['scenario']
    shared.parent.mkdir(parents=True, exist_ok=True)
    with (shared.parent / (task['scenario'] + '.lock')).open('a') as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(.1)
        if not (shared / 'facts.json').exists():
            if shared.exists():
                shared.rename(shared.with_name(shared.name + '-interrupted-' + str(time.time_ns())))
            await baseline(case, shared)
        facts = read_json(shared / 'facts.json')
        if facts['database_sha256'] != file_hash(case / 'initial.db') or facts['environment_sha256'] != digest(read_json(case / 'environment.json')):
            raise ValueError('baseline version mismatch')
        for name, sha in facts['runner_hashes'].items():
            if file_hash(name) != sha:
                raise ValueError('baseline runner version mismatch')
        shutil.copytree(shared, case / 'baseline')
    # No historical labels, verdicts or traces are included in shared evidence.
    catalog = case / 'baseline/tool_catalog.json'
    if catalog.exists():
        tools = read_json(catalog)['tools']
        index = [dict(name=t['name'], description=t.get('description', '')[:180],
                      required=t['inputSchema'].get('required', [])) for t in tools]
    else:
        contract = read_json(case / 'contract.json')
        index = [dict(name=e['operation_id'], description=e.get('summary', ''), source='contract_only_MCP_unavailable')
                 for g in contract['api_spec']['api_groups'] for e in g['endpoints']]
    write_json(case / 'tool_index.json', index)
    for p in (case / 'baseline').rglob('*'):
        if p.is_file():
            p.chmod(0o444)
    # Explicitly authorized raw evidence only; conclusions stay hidden until reveal.
    if job['kind'].startswith('regression_quality'):
        with db_open(run) as db:
            prior = [dict(r) for r in db.execute("SELECT * FROM jobs WHERE case_id=? AND status='completed' AND kind NOT LIKE 'regression_quality%' ORDER BY finished_at", (cid,))]
        if not prior:
            raise ValueError('no same-task completed source for independent regression review')
        write_json(case / 'review_target.json', dict(job_id=prior[-1]['id']))
        index = []
        for n, old in enumerate(prior):
            old_case = Path(old['result_path']).parent
            dest = case / 'prior_raw' / f'round-{n:02d}'
            copy_raw_evidence(old_case, dest)
            original_assets = read_json(old_case / 'asset_hashes.json')
            if original_assets != read_json(case / 'asset_hashes.json'):
                raise ValueError('same-task raw evidence input version mismatch')
            dest.mkdir(parents=True, exist_ok=True)
            runtime_hashes = {p: h for p, h in read_json(run / 'versions/v1/manifest.json')['implementation_hashes'].items()
                              if '/harness_awm/' in p or '/agent-world-model/awm/' in p}
            check_hashes(runtime_hashes)
            write_json(dest / 'provenance.json', dict(source_asset_hashes=original_assets,
                actor_runtime_hashes=runtime_hashes, scope='same task only; copied raw probes and read-only diagnostics, no decisions',
                helper_revisions='Audit-only bootstrap/logging/query-plan validation changes are retained in controller version records; actor runtime and inputs unchanged.',
                raw_sha256={str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*') if p.is_file()}))
            for raw in sorted(dest.rglob('result.json')) if dest.exists() else []:
                value = read_json(raw)
                index.append(dict(ref=str(raw.relative_to(case)), setup_ok=value.get('setup_ok'),
                    error=value.get('error'), calls=[dict(tool=c['tool'], arguments=c['arguments']) for c in value.get('calls', [])]))
        write_json(case / 'same_task_raw_index.json', index)
        for raw in (case / 'prior_raw').rglob('*'):
            if raw.is_file():
                raw.chmod(0o444)
    elif job['kind'] in {'exclude_review', 'pass_audit', 'supplement'}:
        with db_open(run) as db:
            row = db.execute('SELECT first_ref FROM tasks WHERE case_id=?', (cid,)).fetchone()
        original_case = Path(row['first_ref']).parent
        source_assets = read_json(original_case / 'asset_hashes.json')
        if source_assets != read_json(case / 'asset_hashes.json'):
            raise ValueError('review source input version mismatch')
        copy_raw_evidence(original_case, case / 'prior_raw')
        write_json(case / 'prior_raw/provenance.json', dict(source_asset_hashes=source_assets,
            scope='same-task raw probes, diagnostics and evidence; initial conclusions not disclosed'))
        for p in (case / 'prior_raw').rglob('*') if (case / 'prior_raw').exists() else []:
            if p.is_file():
                p.chmod(0o444)
    write_json(case / 'state.json', dict(necessary_obligations=[], confirmed_facts=[],
        evidence_locations=['baseline/facts.json'], unresolved=[], next_step='read task and public tool index'))
    return case


async def probe(case, plan_path):
    from harness_awm.environment import AWMEnvAdapter
    task = task_spec(case)
    plan = read_json(plan_path)
    # Reject malformed audit plans before any startup or business writes.
    validate_plan(plan)
    calls = plan.get('calls', [])
    if not calls:
        raise ValueError('plan must include at least one public tool call')
    folder = case / 'probes' / ('p' + str(time.time_ns()))
    folder.mkdir(parents=True)
    write_json(folder / 'plan.json', plan)
    adapter = AWMEnvAdapter(task, config(), folder)
    result = dict(started_at=now(), calls=[], queries=[], setup_ok=False,
        initial_input_sha256=task.initial_state_hash, initial_logical_fingerprint=task.initial_state_fingerprint,
        environment_sha256=task.environment_code_hash, probe_helper_sha256=file_hash(__file__))
    values = []
    try:
        await adapter.prepare()
        result['setup_ok'] = True
        with sqlite3.connect((adapter.runtime / 'working.db').as_uri() + '?mode=ro', uri=True) as db:
            for q in plan.get('before_queries', []):
                result.setdefault('before_queries', []).append(sql_evidence(db, q, values))
        for call in calls:
            args = resolve(call.get('arguments', {}), values)
            response = await adapter.call(call['tool'], args)
            parsed = payload(response)
            values.append(parsed)
            result['calls'].append(dict(tool=call['tool'], arguments=args,
                argument_sources=call.get('argument_sources'), response=response, parsed=parsed))
            if response.get('isError') and not plan.get('continue_on_error', False):
                break
        with sqlite3.connect((adapter.runtime / 'working.db').as_uri() + '?mode=ro', uri=True) as db:
            db.execute('PRAGMA query_only=ON')
            for q in plan.get('queries', []):
                result['queries'].append(sql_evidence(db, q, values))
            result['integrity'] = db.execute('PRAGMA integrity_check').fetchall()
            result['foreign_keys'] = db.execute('PRAGMA foreign_key_check').fetchall()
    except Exception as exc:
        import traceback
        result['error'] = repr(exc)
        result['traceback'] = traceback.format_exc()
    finally:
        try:
            result['stopped'] = await adapter.close()
        except Exception as exc:
            from harness_awm.isolation import stop_process
            result['cleanup_error'] = repr(exc)
            result['stopped'] = await stop_process(adapter.process, 3)
        if adapter.started:
            try:
                result['final_sha256'] = adapter.freeze()
            except Exception as exc:
                result['freeze_error'] = repr(exc)
        result['initial_unchanged'] = file_hash(case / 'initial.db') == task.initial_state_hash
        if (folder / 'initial.db').exists():
            result['probe_snapshot_sha256'] = file_hash(folder / 'initial.db')
            result['probe_snapshot_logical_fingerprint'] = db_fingerprint(folder / 'initial.db')
            result['probe_snapshot_logical_matches'] = result['probe_snapshot_logical_fingerprint'] == task.initial_state_fingerprint
    result['finished_at'] = now()
    write_json(folder / 'result.json', result)
    preview = dict(result_ref=str(folder / 'result.json'), setup_ok=result['setup_ok'],
        error=result.get('error'), calls=[dict(tool=c['tool'], isError=c['response'].get('isError'),
        parsed=c['parsed']) for c in result['calls']], queries=result['queries'])
    text = json.dumps(preview, ensure_ascii=False)
    if len(text) > 18000:
        text = json.dumps(dict(result_ref=str(folder / 'result.json'), setup_ok=result['setup_ok'],
            error=result.get('error'), preview_omitted=True,
            instruction='完整原始结果已落盘；用过滤或分页读取所需字段，不能将此预览视为完整证据。'), ensure_ascii=False)
    print(text)


def sql_evidence(db, q, values):
    sql, params = q['sql'], resolve(q.get('params', []), values)
    db.execute('PRAGMA query_only=ON')
    cur = db.execute(sql, params)
    limit = q.get('limit', 200)
    records = cur.fetchmany(limit + 1)
    result = dict(sql=sql, params=params, columns=[c[0] for c in cur.description],
                  rows=records[:limit], truncated=len(records) > limit)
    if 'expected' in q:
        result['assertion_passed'] = records == [tuple(r) for r in q['expected']]
    return result


def evidence_refs(decision):
    refs = []
    for item in decision.get('requirements', []):
        refs.extend(item.get('evidence_refs', []))
    refs.extend(decision.get('probe_refs', []))
    if decision.get('defect'):
        refs.extend(decision['defect'].get('evidence_refs', []))
    return refs


def validate_decision(case, path=None):
    path = path or case / 'decision.json'
    d = read_json(path)
    task = read_json(case / 'task.json')
    if d.get('case_id') != task['case_id'] or d.get('decision') not in {'pass', 'exclude', 'review'}:
        raise ValueError('invalid decision identity or label')
    if d.get('verification_level') not in {'runtime', 'code_and_data', 'incomplete'} or not d.get('summary'):
        raise ValueError('missing verification level or summary')
    requirements = d.get('requirements', [])
    if not requirements or any(not r.get('requirement') or not r.get('evidence_refs') or
        r.get('status') not in {'supported', 'blocked', 'unresolved'} for r in requirements):
        raise ValueError('missing necessary obligations/evidence')
    if d['decision'] == 'pass' and any(r['status'] != 'supported' for r in requirements):
        raise ValueError('pass contains unsupported necessary obligation')
    if d['decision'] == 'exclude':
        defect = d.get('defect') or {}
        if not defect.get('mechanism') or defect.get('affects_required_outcome') is not True or not defect.get('alternatives_checked') or not defect.get('evidence_refs'):
            raise ValueError('exclude missing mechanism, causality, alternatives or evidence')
    if d['decision'] == 'review' and not d.get('limitations'):
        raise ValueError('review missing concrete gap')
    resolved = []
    for ref in evidence_refs(d):
        p = resolve_ref(case, ref)
        if p == path.resolve():
            raise ValueError('self-referential decision evidence')
        resolved.append(dict(ref=ref, sha256=file_hash(p)))
    task_spec(case)
    return d, resolved


def reveal(run, case):
    validate_decision(case, case / 'independent.json')
    with (case / 'independent.frozen.json').open('xb') as stream:
        stream.write((case / 'independent.json').read_bytes())
    (case / 'independent.frozen.json').chmod(0o444)
    validate_decision(case, case / 'independent.frozen.json')
    exclusive_json(case / 'independent.seal.json', dict(sha256=file_hash(case / 'independent.frozen.json'),
        frozen_ref='independent.frozen.json', at=now()))
    cid = read_json(case / 'task.json')['case_id']
    with db_open(run) as db:
        row = db.execute('SELECT first_ref FROM tasks WHERE case_id=?', (cid,)).fetchone()
        if row is None:
            target = read_json(case / 'review_target.json')['job_id']
            ref = db.execute("SELECT result_path FROM jobs WHERE id=? AND case_id=? AND status='completed'", (target, cid)).fetchone()
            if ref is None:
                raise ValueError('invalid same-task review target')
            initial_ref = ref[0]
        else:
            initial_ref = row['first_ref']
    d = read_json(initial_ref)
    write_json(case / 'initial_disclosed.json', d)
    print(json.dumps(dict(initial_decision_ref=str(case / 'initial_disclosed.json'),
        note='初审引用路径为初审本题的路径；原始探测的只读副本在 prior_raw/。不得读取初审原目录。'), ensure_ascii=False))


def stream_events(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def usage(thread_id):
    if not thread_id:
        return dict(actual_config_verified=False, missing='no thread id', requests=[], usage=None)
    with sqlite3.connect('file:/home/bjx/.codex/state_5.sqlite?mode=ro', uri=True) as db:
        row = db.execute('SELECT model,reasoning_effort,rollout_path FROM threads WHERE id=?', (thread_id,)).fetchone()
    if row is None:
        return dict(actual_config_verified=False, missing='no thread metadata', requests=[], usage=None)
    records, contexts, cumulative, compactions = {}, [], None, 0
    for e in stream_events(row[2]):
        p = e.get('payload', {})
        if e.get('type') == 'turn_context':
            contexts.append((p.get('model'), p.get('effort')))
        if e.get('type') == 'token_usage_record':
            rid = p.get('response_id')
            if rid:
                records[rid] = dict(response_id=rid, usage=p.get('usage'), thread_id=thread_id)
            cumulative = p.get('thread_token_usage', cumulative)
        if e.get('type') == 'compacted' or (e.get('type') == 'event_msg' and p.get('type') == 'context_compacted'):
            compactions += 1
    total = {k: sum(r['usage'].get(k, 0) for r in records.values()) for k in FIELDS} if records else None
    if total is not None:
        total['uncached_input_tokens'] = total['input_tokens'] - total['cached_input_tokens']
        total['total_tokens'] = total['input_tokens'] + total['output_tokens']
    return dict(thread_id=thread_id, model=row[0], reasoning_effort=row[1], rollout_ref=row[2],
        actual_config_verified=tuple(row[:2]) == (MODEL, EFFORT) and bool(contexts) and all(c == (MODEL, EFFORT) for c in contexts),
        requests=list(records.values()), request_count=len(records), usage=total,
        session_cumulative=cumulative, sum_matches_cumulative=bool(total and cumulative and
            all(total[k] == cumulative.get(k, 0) for k in FIELDS)), compaction_count=compactions,
        missing=None if records else 'per-request usage unavailable')


def message(run, job, case):
    extra = ''
    if is_quality(job['kind']):
        extra = f'''这是独立复核。先依据题面、本题材料及 prior_raw/ 允许复用的原始证据形成自己的判断，保存同格式 independent.json。
再运行 {PYTHON} {Path(__file__).resolve()} reveal --run {run} --case {case}，才可读取 initial_disclosed.json。
对照初审后保存 decision.json 和 review_note.json；后者记录独立判断、差异及理由。证据不足用新副本定向补查。不得提前查阅初审结论。'''
    elif job['kind'] == 'supplement':
        extra = f'''这是唯一一轮补查。先依据题面和原始证据形成 independent.json；再运行 {PYTHON} {Path(__file__).resolve()} reveal --run {run} --case {case} 查看初审具体缺口。
针对具体缺口补查并保存 decision.json、review_note.json；仍有必要信息缺失如实保留 review。'''
    task = read_json(case / 'task.json')
    facts = read_json(case / 'baseline/facts.json')
    summary = {k: facts.get(k) for k in ['mcp_ready', 'tool_count', 'mcp_error', 'integrity', 'started_at', 'parameters']}
    return (run / 'prompt_zh.txt').read_text() + f'''
当前唯一材料目录：{case}
当前题面：{task['task']}
先读 task.json 和 tool_index.json，并按需查看 rubric.json 中与身份、范围、条件有关的 unresolved 原文；已通过的 rubric 不代替环境证据，也不把 rubric 额外解释变成题面硬要求。环境代码、详细契约、schema、初态库按需检索。基础摘要（仅定位原始材料）：{json.dumps(summary, ensure_ascii=False)}
如有 same_task_raw_index.json，先据其定位本题先前的只读原始探测；它不含任何先前判定。适用的实际成功路径和失败记录均须纳入本题判断，不能忽略冲突。
公开工具完整索引 tool_index.json；正式目录和 schema 原始证据 baseline/tool_catalog.json（如启动失败按 facts.json 定位日志）。
探测命令：{PYTHON} {Path(__file__).resolve()} probe --case {case} --plan <本题计划绝对路径>
计划示例：{{"calls":[{{"tool":"公开工具名","arguments":{{}},"argument_sources":"题面或此前公开返回"}}],"before_queries":[],"queries":[{{"sql":"SELECT 所需列 FROM 表 WHERE ...","params":[]}}]}}
一次计划按序共享全新隔离副本；后续参数可用 {{"$result":0,"path":["id"]}} 引用此前公开结果。私有 SQL 不能向 Actor 填 ID。
若 sandbox 阻止启动 probe 的 bwrap/本地端口，可通过现有权限机制请求提升执行这个助手，环境本身仍用 bwrap 隔离。
{extra}
完成后只返回 case_id、结论和证据位置；所有结构化证据及必要义务完整落盘。
'''


def claim(db, kinds):
    db.execute('BEGIN IMMEDIATE')
    placeholders = ','.join('?' for _ in kinds)
    row = db.execute(f"SELECT * FROM jobs WHERE status='pending' AND kind IN ({placeholders}) ORDER BY id LIMIT 1", kinds).fetchone()
    if row:
        db.execute("UPDATE jobs SET status='running',attempts=attempts+1,started_at=? WHERE id=?", (now(), row['id']))
        if row['kind'] != 'regression':
            db.execute("UPDATE tasks SET status='running' WHERE case_id=? AND status!='completed'", (row['case_id'],))
        db.execute('INSERT INTO events(at,job_id,event) VALUES(?,?,?)', (now(), row['id'], 'claimed'))
    db.commit()
    return dict(row) if row else None


def ingest(run, db, job, case, d):
    ref = str(case / 'decision.json')
    if job['kind'].startswith('regression'):
        return
    if job['kind'] == 'initial':
        first = case / 'decision.first_pass.json'
        if first.exists():
            if read_json(first) != d:
                raise FileExistsError('immutable first judgment differs from recovered result')
        else:
            exclusive_json(first, d)
        (case / 'decision.first_pass.json').chmod(0o444)
        db.execute("UPDATE tasks SET first_ref=?,status='pending',reason='awaiting_required_quality_checks' WHERE case_id=?",
                   (str(case / 'decision.first_pass.json'), job['case_id']))
        if d['decision'] == 'exclude':
            add_job(db, job['case_id'], 'exclude_review')
        elif d['decision'] == 'review':
            add_job(db, job['case_id'], 'supplement')
        return
    if job['kind'] in {'exclude_review', 'pass_audit'}:
        initial = read_json(db.execute('SELECT first_ref FROM tasks WHERE case_id=?', (job['case_id'],)).fetchone()[0])
        if d['decision'] != initial['decision']:
            db.execute("UPDATE tasks SET status='blocked',reason='requires_master_adjudication' WHERE case_id=?", (job['case_id'],))
            return
    db.execute("UPDATE tasks SET final_ref=?,status='completed',reason=NULL WHERE case_id=?", (ref, job['case_id']))


def freeze_sample(run, db):
    if db.execute("SELECT count(*) FROM jobs WHERE kind='initial' AND status!='completed'").fetchone()[0]:
        return False
    if (run / 'pass_sample.json').exists():
        return False
    groups = defaultdict(list)
    for r in db.execute('SELECT * FROM tasks ORDER BY task_id'):
        if read_json(r['first_ref'])['decision'] == 'pass':
            groups[r['scenario']].append(dict(r))
    rng = random.Random(SEED)
    selected = []
    for scenario in sorted(groups):
        pop = groups[scenario]
        count = math.ceil(len(pop) * .05)
        chosen = sorted(rng.sample(pop, count), key=lambda r: r['task_id'])
        for r in chosen:
            selected.append(dict(case_id=r['case_id'], task_id=r['task_id'], scenario=scenario))
        sampled = {r['case_id'] for r in chosen}
        for r in pop:
            if r['case_id'] in sampled:
                add_job(db, r['case_id'], 'pass_audit')
            else:
                db.execute("UPDATE tasks SET final_ref=first_ref,status='completed',reason=NULL WHERE case_id=?", (r['case_id'],))
    # Commit queue transitions first: an interrupted publication can be regenerated deterministically.
    db.commit()
    write_json(run / 'pass_sample.json', dict(seed=SEED, method='task_id sorted per scenario; scenarios sorted; random.Random(seed).sample; ceil(n*0.05)',
        population=[dict(scenario=s, task_ids=[r['task_id'] for r in groups[s]]) for s in sorted(groups)], selected=selected,
        frozen_at=now()))
    return True


async def recover_claims(run, db):
    old = read_json(run / 'runtime_status.json') if (run / 'runtime_status.json').exists() else {}
    for job in db.execute("SELECT * FROM jobs WHERE status='running'").fetchall():
        owner = old.get('active', {}).get(job['id'], {})
        pid = owner.get('pid')
        if pid and Path(f'/proc/{pid}/cmdline').exists():
            cmdline = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\x00', b' ').decode(errors='replace')
            # Never signal a reused/unrelated PID; ownership includes the assigned case path.
            owned = 'codex' in cmdline and str(run / 'work') in cmdline and job['case_id'] in cmdline
            if owned:
                if os.getpgid(pid) != pid:
                    raise RuntimeError('orphan session has unexpected process group: ' + job['id'])
                try:
                    os.killpg(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic() + 5
                while Path(f'/proc/{pid}').exists() and time.monotonic() < deadline:
                    await asyncio.sleep(.1)
                if Path(f'/proc/{pid}').exists():
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        session = run / 'sessions' / job['id'] / f"attempt-{job['attempts']-1:02d}"
        events = session / 'events.jsonl'
        completed_turn = False
        if events.exists() and not (session / 'usage.json').exists():
            tid = None
            for line in events.read_text().splitlines():
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get('type') == 'thread.started':
                    tid = e.get('thread_id')
            write_json(session / 'usage.json', usage(tid))
        if events.exists():
            for line in events.read_text().splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                completed_turn = completed_turn or event.get('type') == 'turn.completed'
        public_kind = 'quality' if is_quality(job['kind']) else job['kind']
        case = run / 'work' / (job['case_id'] + '-' + public_kind) / f"attempt-{job['attempts']-1:02d}"
        if completed_turn and (session / 'usage.json').exists() and (case / 'decision.json').exists():
            try:
                actual = read_json(session / 'usage.json')
                if not actual.get('actual_config_verified') or (case / 'infrastructure.json').exists():
                    raise ValueError('interrupted result configuration/infrastructure check failed')
                decision, refs = validate_decision(case)
                if is_quality(job['kind']) or job['kind'] == 'supplement':
                    seal = read_json(case / 'independent.seal.json')
                    if seal['sha256'] != file_hash(case / 'independent.frozen.json') or not (case / 'review_note.json').exists():
                        raise ValueError('interrupted independent review seal incomplete')
                ingest(run, db, job, case, decision)
                db.execute("UPDATE jobs SET status='completed',result_path=?,error=NULL,finished_at=? WHERE id=?", (str(case / 'decision.json'), now(), job['id']))
                write_json(session / 'validation.recovered.json', dict(valid=True, evidence=refs,
                    reason='Completed model turn and original artifacts verified after controller interruption; no business replay.'))
            except Exception as exc:
                db.execute("UPDATE jobs SET status='blocked',error=? WHERE id=?", ('completed artifact requires reconciliation: ' + repr(exc), job['id']))
                db.execute("UPDATE tasks SET status='blocked',reason=? WHERE case_id=?", (repr(exc), job['case_id']))
        db.execute('INSERT INTO events(at,job_id,event,detail) VALUES(?,?,?,?)',
            (now(), job['id'], 'interrupted_claim_recovered', 'orphan ownership checked; old evidence retained; next attempt uses a new copy'))
    db.execute("UPDATE jobs SET status='pending',error='controller interrupted; fresh case required' WHERE status='running'")
    db.execute("UPDATE jobs SET status='blocked',error='maximum infrastructure attempts reached' WHERE status='pending' AND attempts>=3")
    db.execute("UPDATE tasks SET status='pending' WHERE status='running'")
    db.execute("UPDATE tasks SET status='blocked',reason='maximum infrastructure attempts reached' WHERE case_id IN (SELECT case_id FROM jobs WHERE status='blocked' AND attempts>=3)")
    db.commit()


async def run_jobs(run, phase):
    verify_run(run)
    if phase != 'regression':
        gate = read_json(run / 'regression_gate.json')
        if not gate.get('approved') or gate['prompt_sha256'] != file_hash(run / 'prompt_zh.txt') or gate['runner_sha256'] != file_hash(__file__):
            raise ValueError('regression and prompt freeze gate is not approved')
    lock = (run / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    db = db_open(run)
    await recover_claims(run, db)
    kinds = [r[0] for r in db.execute("SELECT DISTINCT kind FROM jobs WHERE kind LIKE 'regression%'")] if phase == 'regression' else ['initial', 'exclude_review', 'pass_audit', 'supplement']
    if any(r['kind'] in kinds for r in db.execute("SELECT kind FROM jobs WHERE status='blocked'")):
        raise RuntimeError('unresolved blocked jobs: reconcile their evidence before restarting dispatch')
    stop, peak, model_peak, failures = False, 0, 0, 0
    active = {}
    controller = dict(started_at=now(), pid=os.getpid(), phase=phase, model=MODEL, effort=EFFORT, concurrency=WORKERS)
    write_json(run / 'controller.json', controller)

    def status():
        old = read_json(run / 'runtime_status.json') if (run / 'runtime_status.json').exists() else {}
        write_json(run / 'runtime_status.json', dict(updated_at=now(), active=active, stop_dispatch=stop,
            peak_concurrency=max(peak, old.get('peak_concurrency', 0)), consecutive_infrastructure_failures=failures,
            model_peak_concurrency=max(model_peak, old.get('model_peak_concurrency', 0)),
            phase=phase, model=MODEL, reasoning_effort=EFFORT, concurrency_limit=WORKERS))

    async def worker():
        nonlocal stop, peak, model_peak, failures
        while not stop:
            job = claim(db, kinds)
            if job is None:
                if phase != 'regression' and freeze_sample(run, db):
                    continue
                return
            attempt = job['attempts']
            folder = run / 'sessions' / job['id'] / f'attempt-{attempt:02d}'
            folder.mkdir(parents=True, exist_ok=False)
            case, proc, error, data, refs = None, None, None, None, []
            active[job['id']] = dict(started_at=now(), attempt=attempt, stage='baseline')
            peak = max(peak, len(active))
            status()
            try:
                case = await provision(run, job, attempt)
                prompt = message(run, job, case)
                (folder / 'prompt.txt').write_text(prompt)
                argv = ['codex', 'exec', '--skip-git-repo-check', '--ignore-user-config', '-m', MODEL,
                    '-c', 'model_reasoning_effort="high"', '--approve-for-me', '--json', '-C', str(case),
                    '-o', str(folder / 'response.txt'), '-']
                active[job['id']]['stage'] = 'model'
                with (folder / 'events.jsonl').open('w') as out, (folder / 'stderr.txt').open('w') as err:
                    proc = await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.PIPE,
                        stdout=out, stderr=err, cwd=case, start_new_session=True)
                    active[job['id']]['pid'] = proc.pid
                    model_peak = max(model_peak, sum(v.get('stage') == 'model' and 'pid' in v for v in active.values()))
                    status()
                    try:
                        await asyncio.wait_for(proc.communicate(prompt.encode()), 1500)
                    except asyncio.TimeoutError:
                        os.killpg(proc.pid, signal.SIGTERM)
                        await proc.wait()
                        raise RuntimeError('session_timeout')
                active[job['id']]['stage'] = 'validating'
                events = rows(folder / 'events.jsonl')
                tid = next((e.get('thread_id') for e in events if e.get('type') == 'thread.started'), None)
                u = usage(tid)
                write_json(folder / 'usage.json', u)
                if proc.returncode or not any(e.get('type') == 'turn.completed' for e in events):
                    raise RuntimeError('model_service_failure:' + str([e for e in events if e.get('type') in {'error', 'turn.failed'}]))
                if not u['actual_config_verified']:
                    raise RuntimeError('requested_model_configuration_not_verified')
                if (case / 'infrastructure.json').exists():
                    raise RuntimeError('infrastructure:' + (case / 'infrastructure.json').read_text())
                data, refs = validate_decision(case)
                if is_quality(job['kind']) or job['kind'] == 'supplement':
                    seal = read_json(case / 'independent.seal.json')
                    if seal['sha256'] != file_hash(case / 'independent.frozen.json') or not (case / 'review_note.json').exists():
                        raise ValueError('independent review missing or changed after disclosure')
                    if file_hash(case / 'independent.json') != seal['sha256']:
                        revision = case / ('independent.draft-revision-' + str(time.time_ns()) + '.json')
                        shutil.copyfile(case / 'independent.json', revision)
                        write_json(case / 'independent.draft_revision.json', dict(
                            original_ref='independent.frozen.json', original_sha256=seal['sha256'],
                            revised_draft_ref=revision.name, revised_sha256=file_hash(revision),
                            original_judgment_preserved=True,
                            reason='Model amended its mutable draft after disclosure; immutable original remains authoritative for independent comparison.'))
                write_json(folder / 'validation.json', dict(valid=True, evidence=refs, decision_sha256=file_hash(case / 'decision.json')))
            except Exception as exc:
                error = repr(exc)
                if proc and proc.returncode is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    await proc.wait()
            if error is None:
                failures = 0
                with db:
                    db.execute("UPDATE jobs SET status='completed',result_path=?,finished_at=?,error=NULL WHERE id=?", (str(case / 'decision.json'), now(), job['id']))
                    ingest(run, db, job, case, data)
            else:
                failures += 1
                fatal = bool(re.search(r'auth|unauthoriz|quota|usage.limit|rate.limit|model.{0,30}(unavailable|not.found|not.support)|configuration_not_verified', error, re.I))
                # Invalid result/script errors require fixing, never relabeling the task.
                validation_error = error.startswith(('ValueError', 'FileNotFoundError', 'PermissionError'))
                stop = stop or fatal or failures >= 3 or validation_error or attempt >= 2
                with db:
                    db.execute('UPDATE jobs SET status=?,error=?,finished_at=? WHERE id=?', ('blocked' if stop else 'pending', error, now(), job['id']))
                    if job['kind'] != 'regression':
                        db.execute("UPDATE tasks SET status='blocked',reason=? WHERE case_id=?", (error, job['case_id']))
                write_json(folder / 'failure.json', dict(at=now(), error=error, is_task_defect=False, stop_dispatch=stop))
            active.pop(job['id'])
            status()
            report(run)
            print(json.dumps(dict(job=job['id'], error=error, active=len(active), stop_dispatch=stop), ensure_ascii=False), flush=True)
    try:
        await asyncio.gather(*(worker() for _ in range(WORKERS)))
    finally:
        status()
        db.close()
        report(run)
        lock.close()


def report(run):
    m = read_json(run / 'manifest.json')
    with db_open(run) as db:
        tasks = [dict(r) for r in db.execute('SELECT * FROM tasks ORDER BY task_id')]
        jobs = [dict(r) for r in db.execute('SELECT * FROM jobs ORDER BY id')]
    results, incomplete = [], []
    for task in tasks:
        if task['status'] != 'completed':
            incomplete.append(task)
            continue
        d = read_json(task['final_ref'])
        results.append(dict(**task, first_decision=read_json(task['first_ref'])['decision'],
            final_decision=d['decision'], rubric_accepted=True, verification_level=d['verification_level'],
            category=d.get('category'), summary=d['summary']))
    for label in ['pass', 'exclude', 'review']:
        jsonl(run / (label + '.jsonl'), [r for r in results if r['final_decision'] == label])
    jsonl(run / 'unfinished.jsonl', incomplete)
    jsonl(run / 'queue.jsonl', jobs)
    columns = ['case_id', 'task_id', 'scenario', 'status', 'first_decision', 'final_decision', 'rubric_accepted',
               'verification_level', 'category', 'reason', 'first_ref', 'final_ref', 'summary']
    with (run / 'results.csv').open('w') as out:
        writer = csv.DictWriter(out, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(sorted(results + incomplete, key=lambda r: r['task_id']))
    usage_records, unknown, phase_usage, compactions, sessions = {}, [], defaultdict(Counter), 0, []
    for p in sorted((run / 'sessions').glob('*/*/usage.json')):
        u = read_json(p)
        kind = p.parent.parent.name.rsplit('-', 1)[-1]
        sessions.append(dict(ref=str(p), thread_id=u.get('thread_id'), actual_config_verified=u.get('actual_config_verified'),
                             request_count=u.get('request_count'), sum_matches_cumulative=u.get('sum_matches_cumulative')))
        compactions += u.get('compaction_count', 0)
        if u.get('missing'):
            unknown.append(dict(ref=str(p), reason=u['missing']))
        for r in u.get('requests', []):
            if r['response_id'] not in usage_records:
                usage_records[r['response_id']] = r
                phase_usage[kind].update({k: r['usage'].get(k, 0) for k in FIELDS})
    totals = {k: sum(r['usage'].get(k, 0) for r in usage_records.values()) for k in FIELDS} if usage_records else None
    if totals is not None:
        totals.update(uncached_input_tokens=totals['input_tokens'] - totals['cached_input_tokens'],
                      total_tokens=totals['input_tokens'] + totals['output_tokens'])
    write_json(run / 'token_usage_audit.json', dict(updated_at=now(), totals=totals,
        request_count=len(usage_records), compaction_count=compactions, phases=dict(phase_usage),
        missing=unknown, sessions=sessions, master_usage_ref=str(run / 'master_usage.json'),
        master_usage_status='separate; ongoing turn cannot have a final total',
        accounting='deduplicate response_id; cache subset of input; reasoning subset of output; includes compaction records',
        savings_comparison=None, savings_reason='no matched same-task same-model same-standard baseline'))
    runtime = read_json(run / 'runtime_status.json') if (run / 'runtime_status.json').exists() else {}
    completed = len(results)
    summary = dict(updated_at=now(), frozen_count=m['frozen_count'], completed=completed,
        unfinished=len(incomplete), complete=completed == m['frozen_count'],
        decisions=dict(Counter(r['final_decision'] for r in results)),
        task_states=dict(Counter(r['status'] for r in tasks)),
        job_states=dict(Counter(r['status'] for r in jobs)),
        initial_completed=sum(j['kind'] == 'initial' and j['status'] == 'completed' for j in jobs),
        regression_completed=sum(j['kind'] == 'regression' and j['status'] == 'completed' for j in jobs),
        regression_corrections_completed=sum(j['kind'].startswith('regression_correction') and j['status'] == 'completed' for j in jobs),
        retry_count=sum(max(0, j['attempts'] - 1) for j in jobs), peak_concurrency=runtime.get('peak_concurrency', 0),
        model_peak_concurrency=runtime.get('model_peak_concurrency'),
        model=MODEL, reasoning_effort=EFFORT, categories=dict(Counter(r['category'] for r in results)),
        reason_counts=dict(Counter(r.get('reason') or 'not_started_or_pending_checks' for r in incomplete)))
    write_json(run / 'summary.json', summary)
    (run / 'REPORT.md').write_text(f'''# 逐题环境审查运行报告\n\n冻结 {m['frozen_count']} 题（rubric 通过 {m['accepted_count']}，排除已执行 {m['executed_count']}）。\n\n已完成全部规定检查：{completed}；未完成：{len(incomplete)}。全量完成：{summary['complete']}。\n\n回归完成 {summary['regression_completed']}/{m['regression_count']}，初审完成 {summary['initial_completed']}。\n\n模型 {MODEL}，reasoning {EFFORT}，并发上限 {WORKERS}，实际峰值 {summary['peak_concurrency']}。\n\n最终分类：{json.dumps(summary['decisions'], ensure_ascii=False)}。未开始、基础设施阻塞和待复核题不进入三类最终结果。\n\n队列：queue.sqlite / queue.jsonl；未完成及原因：unfinished.jsonl；用量：token_usage_audit.json（主 agent 单列，缺失不记零）。\n\n每题首次和最终结论引用在 results.csv，完整原始证据位于 work/，会话日志位于 sessions/。\n\n本轮仅提供有限的任务相关环境验证，不证明整个环境没有问题，不设置分类比例，不宣称未经可比基线核实的 token 节省率。\n''')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'run', 'report', 'probe', 'reveal', 'verify', 'usage'])
    p.add_argument('--run', type=Path, default=DEFAULT)
    p.add_argument('--instruction', type=Path)
    p.add_argument('--phase', choices=['regression', 'full'], default='regression')
    p.add_argument('--case', type=Path)
    p.add_argument('--plan', type=Path)
    args = p.parse_args()
    if args.mode == 'probe' and Path(sys.prefix).resolve() != PYTHON.parent.parent.resolve():
        os.execv(str(PYTHON), [str(PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
    run = args.run.resolve()
    if args.mode == 'prepare':
        prepare(run, args.instruction)
    elif args.mode == 'run':
        asyncio.run(run_jobs(run, args.phase))
    elif args.mode == 'report':
        report(run)
    elif args.mode == 'verify':
        print(json.dumps(dict(verified=bool(verify_run(run)))))
    elif args.mode == 'probe':
        asyncio.run(probe(args.case.resolve(), args.plan.resolve()))
    elif args.mode == 'reveal':
        reveal(run, args.case.resolve())
    elif args.mode == 'usage':
        m = read_json(run / 'manifest.json')
        write_json(run / 'master_usage.json', usage(m['master_thread_id']))


if __name__ == '__main__':
    main()
