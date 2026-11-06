#!/usr/bin/env python3
"""Offline full-bank screening. No actor, judge or paid API calls.

The admission result is a finite screening result, never a solvability proof.
Each HTTP probe executes in a separate process and a fresh database copy.
"""
import argparse
import ast
import asyncio
from collections import Counter
import csv
import json
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness_awm.io import db_fingerprint, digest, file_hash, now, read_json, write_json

DEFAULT = ROOT / 'runs/environment_screen_20261104_v2'
RELEASE = ROOT / 'data/awm-dde80a0-repaired-v1'
BANK = ROOT / 'runs/rubric_luna_registry/bank_manifest.json'
EXECUTION = ROOT / 'runs/bank3000_execution_20261104_v1'
WORKER = ROOT / 'scripts/screen_environment_worker.py'
LIMITS = [
    'All bank IDs are accounted for; source binding and database/runtime checks are not a proof that every task is solvable.',
    'HTTP probes use contract examples or seed values, one finite input per operation; uncovered and rejected probes are recorded.',
    'Write probes are isolated ASGI HTTP requests on fresh copies; they do not certify task-specific write semantics or every MCP body encoding.',
    'MCP startup, discovery and a bounded read probe are checked separately through the production adapter.',
    'Shared-environment defects quarantine associated tasks conservatively, without claiming each task is impossible.',
    '4xx, timeouts and arbitrary probe failures are not automatically attributed to an environment defect.',
    'Rubric eligibility is separate. No reward, model success label or training certification is produced.',
]


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def dump_rows(path, records):
    with Path(path).open('w') as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def orm_checks(tree, db):
    from sqlalchemy.engine import processors
    parsers = {'Date': processors.str_to_date, 'DateTime': processors.str_to_datetime,
               'TIMESTAMP': processors.str_to_datetime, 'Time': processors.str_to_time}
    errors, checked = [], 0
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        assigns = {n.targets[0].id: n.value for n in cls.body if isinstance(n, ast.Assign)
                   and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
        tab = assigns.get('__tablename__')
        if not isinstance(tab, ast.Constant) or not isinstance(tab.value, str):
            continue
        table = tab.value
        columns = {r[1] for r in db.execute(f'PRAGMA table_xinfo({quote(table)})')}
        for attr, node in assigns.items():
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != 'Column':
                continue
            args = node.args
            column = args[0].value if args and isinstance(args[0], ast.Constant) and isinstance(args[0].value, str) else attr
            if column not in columns:
                if column.lower() in ('rowid', '_rowid_', 'oid') and columns:
                    continue
                errors.append(dict(kind='orm_schema_mismatch', table=table, column=column, line=node.lineno))
                continue
            types = [a.func.id if isinstance(a, ast.Call) and isinstance(a.func, ast.Name)
                     else a.id if isinstance(a, ast.Name) else '' for a in args]
            typ = next((t for t in types if t in parsers), None)
            if typ:
                for value, count in db.execute(f'SELECT {quote(column)},count(*) FROM {quote(table)} WHERE {quote(column)} IS NOT NULL GROUP BY {quote(column)}'):
                    checked += count
                    try:
                        parsers[typ](value)
                    except Exception as exc:
                        errors.append(dict(kind='orm_value_error', table=table, column=column,
                                           value=value, count=count, line=node.lineno, error=str(exc)))
    return dict(checked_date_values=checked, defects=errors)


def static_routes(tree):
    from starlette.routing import compile_path
    routes, defects = [], []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if not (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                    and dec.func.attr in {'get', 'post', 'put', 'patch', 'delete'}
                    and dec.args and isinstance(dec.args[0], ast.Constant) and isinstance(dec.args[0].value, str)):
                continue
            path = dec.args[0].value
            operation = next((k.value.value for k in dec.keywords if k.arg == 'operation_id' and isinstance(k.value, ast.Constant)), node.name)
            route = dict(path=path, method=dec.func.attr.upper(), operation=operation, line=node.lineno)
            # Concrete paths only: no invented parameter values masquerading as full coverage.
            if '{' not in path:
                earlier = next((r for r in routes if r['method'] == route['method'] and compile_path(r['path'])[0].fullmatch(path)), None)
                if earlier:
                    defects.append(dict(kind='route_shadow', intended=route, actual=earlier))
            routes.append(route)
    counts = Counter(r['operation'] for r in routes)
    for operation, count in counts.items():
        if count > 1:
            defects.append(dict(kind='duplicate_operation_id', operation=operation, count=count))
    return routes, defects


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise ValueError('manifest already exists; use runtime/report to resume')
    bank = read_json(BANK)
    for ref in bank['exports'].values():
        if file_hash(ref['path']) != ref['sha256']:
            raise ValueError('bank export changed')
    questions = rows(bank['exports']['question_bank']['path'])
    if len({q['task_id'] for q in questions}) != len(questions):
        raise ValueError('duplicate bank IDs')
    selected = {q['task_id'] for q in read_json(EXECUTION / 'manifest.json')['tasks']}
    release = read_json(RELEASE / 'manifest.json')
    for name, sha in release['output_sha256'].items():
        if file_hash(RELEASE / name) != sha:
            raise ValueError('release asset changed: ' + name)
    envs = {r['scenario']: r for r in rows(RELEASE / 'gen_envs.jsonl')}
    specs = {r['scenario']: r for r in rows(RELEASE / 'gen_spec.jsonl')}
    task_assets = {r['scenario']: r['tasks'] for r in rows(RELEASE / 'gen_tasks.jsonl')}
    validation = {r['scenario']: r for r in rows(RELEASE / 'validation.jsonl')}
    manifest = dict(version='awm-environment-screen-v1', started_at=now(), release=str(RELEASE),
                    bank_manifest_sha256=file_hash(BANK), execution_manifest_sha256=file_hash(EXECUTION / 'manifest.json'),
                    source_hashes={str(RELEASE / n): h for n, h in release['output_sha256'].items()},
                    implementation_hashes={str(p): file_hash(p) for p in [Path(__file__), WORKER,
                        ROOT / 'harness_awm/environment.py', ROOT / 'harness_awm/environment_screen.py',
                        ROOT / 'agent-world-model/awm/core/server.py']},
                    total_questions=len(questions), selected_prior=len(selected),
                    remaining=len(questions)-len(selected), scenarios=sorted(envs), limits=LIMITS,
                    api_requests=0, scope='Full bank; residual excludes the frozen 3000 planned IDs, including those still running.')
    write_json(out / 'manifest.json', manifest)
    task_rows = []
    for q in questions:
        reasons = []
        original = Path(q['original_ref'])
        if not original.exists() or file_hash(original) != q['original_sha256']:
            reasons.append('original_hash_mismatch_or_missing')
        else:
            src = read_json(original)
            if src.get('task') != q['task'] or src.get('task_id') != q['task_id']:
                reasons.append('original_binding_mismatch')
        tasks = task_assets.get(q['scenario'], [])
        if not isinstance(q['index'], int) or not 0 <= q['index'] < len(tasks) or tasks[q['index']] != q['task']:
            reasons.append('release_task_binding_mismatch')
        task_rows.append(dict(task_id=q['task_id'], scenario=q['scenario'], index=q['index'], task=q['task'],
            remaining=q['task_id'] not in selected, rubric_accepted=q['passed'], rubric_status=q['status'],
            original_ref=q['original_ref'], original_sha256=q['original_sha256'],
            binding_errors=reasons, semantic_solvability='not_proven'))
    dump_rows(out / 'task_inventory.jsonl', task_rows)
    for i, scenario in enumerate(sorted(envs)):
        folder = out / 'scenarios' / scenario
        folder.mkdir(parents=True, exist_ok=True)
        env = envs[scenario]
        write_json(folder / 'environment.json', env)
        write_json(folder / 'contract.json', specs[scenario])
        initial = RELEASE / 'databases' / (scenario + '.db')
        result = dict(scenario=scenario, initial=str(initial), initial_sha256=file_hash(initial),
                      environment_sha256=digest(env), defects=[], warnings=[])
        try:
            if result['initial_sha256'] != validation[scenario]['database_sha256']:
                result['defects'].append(dict(kind='database_hash_mismatch'))
            with sqlite3.connect(initial.resolve().as_uri() + '?mode=ro', uri=True) as db:
                integrity = db.execute('PRAGMA integrity_check').fetchall()
                fk = db.execute('PRAGMA foreign_key_check').fetchall()
                result.update(integrity=integrity, foreign_keys=fk)
                if integrity != [('ok',)] or fk:
                    result['defects'].append(dict(kind='database_integrity'))
                tree = ast.parse(env['full_code'])
                compile(tree, scenario, 'exec')
                orm = orm_checks(tree, db)
                result['checked_date_values'] = orm['checked_date_values']
                result['defects'].extend(orm['defects'])
                routes, defects = static_routes(tree)
                result['routes'] = routes
                result['defects'].extend(defects)
                code_names = {r['operation'] for r in routes}
                for group in specs[scenario]['api_spec']['api_groups']:
                    for op in group['endpoints']:
                        if op['operation_id'] not in code_names:
                            result['warnings'].append(dict(kind='contract_operation_not_found_in_static_routes', operation=op['operation_id']))
        except Exception as exc:
            result['defects'].append(dict(kind='static_check_exception', error=f'{type(exc).__name__}: {exc}'))
        write_json(folder / 'static.json', result)
        if (i+1) % 100 == 0:
            print('static', i+1, '/', len(envs), flush=True)
    print(json.dumps({k: manifest[k] for k in ['total_questions','remaining','scenarios']} | {'scenarios':len(envs)}), flush=True)


def check_manifest(out):
    manifest = read_json(out / 'manifest.json')
    for path, expected in manifest['source_hashes'].items():
        if file_hash(path) != expected:
            raise ValueError('frozen source changed: '+path)
    for path, expected in manifest['implementation_hashes'].items():
        if file_hash(path) != expected:
            raise ValueError('implementation changed: '+path)
    return manifest


async def runtime(out, jobs, limit):
    from harness_awm.contracts import RunConfig
    from harness_awm.environment import AWMEnvAdapter
    from harness_awm.isolation import command, clean_env, stop_process, wait_process
    manifest = check_manifest(out)
    sem = asyncio.Semaphore(jobs)
    done = 0
    async def one(scenario):
        nonlocal done
        async with sem:
            folder = out / 'scenarios' / scenario
            if (folder / 'runtime_result.json').exists():
                done += 1
                return
            static = read_json(folder / 'static.json')
            initial = Path(static['initial'])
            if file_hash(initial) != static['initial_sha256']:
                raise ValueError('initial snapshot changed')
            config = RunConfig(startup_timeout=45, tool_timeout=10, evaluator_timeout=60, stop_timeout=3)
            task = SimpleNamespace(scenario=scenario, initial_state_ref=str(initial),
                initial_state_hash=static['initial_sha256'], initial_state_fingerprint=db_fingerprint(initial),
                environment_ref=str(folder / 'environment.json'), environment_code_hash=static['environment_sha256'],
                tool_schema_hash=None, readiness_profile={'runtime_ready':True})
            # Interrupted setup is retained; each attempt uses a new directory.
            attempt = len(list(folder.glob('mcp_attempt_*')))
            mcp_dir = folder / f'mcp_attempt_{attempt:02d}'
            mcp_dir.mkdir()
            adapter = AWMEnvAdapter(task, config, mcp_dir)
            result = dict(scenario=scenario, mcp_ready=False, mcp_probes=[], started_at=now())
            try:
                await adapter.prepare()
                result.update(mcp_ready=True, tool_count=len(adapter.tools))
                # ASGI worker covers every HTTP operation separately. Check the MCP transport too.
                route_map = {r['operation']: r for r in static.get('routes',[])}
                candidates = [t for t in adapter.tools if route_map.get(t['name'],{}).get('method')=='GET'
                              and not t['inputSchema'].get('required')]
                for tool in candidates[:1]:
                    response = await adapter.call(tool['name'], {})
                    result['mcp_probes'].append(dict(tool=tool['name'], arguments={}, response=response))
            except Exception as exc:
                result['mcp_error'] = f'{type(exc).__name__}: {exc}'
            finally:
                result['mcp_stopped'] = await adapter.close()
            generated = mcp_dir / 'server_build/server.py'
            if generated.exists():
                work = folder / f'http_attempt_{attempt:02d}'
                work.mkdir()
                shutil.copyfile(initial, work / 'initial.db')
                shutil.copyfile(generated, work / 'server.py')
                shutil.copyfile(folder / 'contract.json', work / 'contract.json')
                job = dict(work=str(work), original_db=str(adapter.runtime / 'working.db'),
                           timeout=12, scenario=scenario)
                write_json(work / 'job.json', job)
                argv = command(config, [config.runtime_python, str(WORKER), str(work / 'job.json')], [work], [WORKER])
                # Supplemental HTTP calls need no sockets: block external networking as well.
                argv.insert(1, '--unshare-net')
                process = None
                try:
                    with (work / 'worker.log').open('w') as log:
                        process = subprocess.Popen(argv, env=clean_env(config.timezone), stdout=log, stderr=log, start_new_session=True)
                        await wait_process(process, 600)
                    if process.returncode != 0 or not (work/'result.json').exists():
                        raise RuntimeError('http_worker_failed')
                    result['http'] = read_json(work / 'result.json')
                    result['http_result_ref'] = str(work / 'result.json')
                except Exception as exc:
                    result['http_error'] = f'{type(exc).__name__}: {exc}'
                    result['http_result_ref'] = str(work / 'worker.log')
                finally:
                    result['http_stopped'] = await stop_process(process, 3)
            else:
                result['http_error'] = 'no_generated_server'
            result['initial_unchanged'] = file_hash(initial) == static['initial_sha256']
            result['finished_at'] = now()
            write_json(folder / 'runtime_result.json', result)
            done += 1
            with (out / 'progress.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(scenario=scenario, done=done, mcp_ready=result['mcp_ready'],
                                             http=result.get('http',{}).get('counts',{})))+'\n')
            if done % 20 == 0 or not result['mcp_ready']:
                print('runtime',done,scenario,'mcp',result['mcp_ready'],result.get('http',{}).get('counts',{}),flush=True)
    scenarios = manifest['scenarios'][:limit] if limit else manifest['scenarios']
    await asyncio.gather(*(one(s) for s in scenarios))


def summarize_scenario(static, runtime):
    defects = list(static['defects'])
    warnings = list(static['warnings'])
    coverage = {}
    if runtime is None:
        return dict(status='review', defects=defects, warnings=warnings+[dict(kind='runtime_not_checked')], coverage={})
    if not runtime.get('initial_unchanged'):
        defects.append(dict(kind='source_snapshot_changed'))
    if not runtime.get('mcp_ready'):
        # Infrastructure failure is not necessarily a broken asset.
        warnings.append(dict(kind='mcp_setup_failed', error=runtime.get('mcp_error')))
    if not runtime.get('mcp_stopped') or not runtime.get('http_stopped',True):
        warnings.append(dict(kind='cleanup_incomplete'))
    for probe in runtime.get('mcp_probes',[]):
        if probe['response'].get('isError'):
            warnings.append(dict(kind='mcp_probe_error',tool=probe['tool']))
    http = runtime.get('http',{})
    if runtime.get('http_error') or http.get('setup_error'):
        warnings.append(dict(kind='http_setup_error', error=runtime.get('http_error') or http.get('setup_error')))
    coverage = http.get('counts',{})
    for probe in http.get('probes',[]):
        if probe.get('route_mismatch'):
            defects.append(dict(kind='http_route_mismatch', tool=probe['tool'], evidence=probe['result_ref']))
        if probe.get('new_integrity_errors'):
            defects.append(dict(kind='probe_corrupts_database', tool=probe['tool'], evidence=probe['result_ref']))
        if probe.get('exception') or probe.get('timeout') or probe.get('status',0)>=500:
            warnings.append(dict(kind='probe_server_error_requires_attribution',tool=probe['tool'],evidence=probe['result_ref']))
        elif probe.get('status',0)>=400:
            warnings.append(dict(kind='business_or_probe_rejection',tool=probe['tool'],evidence=probe['result_ref']))
    # Explicit probe rejections remain warnings but do not by themselves label an environment broken.
    material = [w for w in warnings if w['kind']!='business_or_probe_rejection']
    if http.get('skipped'):
        material.append(dict(kind='operations_without_constructible_probe',count=len(http['skipped'])))
        warnings.append(material[-1])
    status = 'exclude' if defects else 'review' if material else 'screen_pass'
    return dict(status=status, defects=defects, warnings=warnings, coverage=coverage)


def report(out):
    manifest = check_manifest(out)
    scenario_rows = []
    for scenario in manifest['scenarios']:
        folder=out/'scenarios'/scenario
        static=read_json(folder/'static.json')
        runtime=read_json(folder/'runtime_result.json') if (folder/'runtime_result.json').exists() else None
        row=dict(scenario=scenario, **summarize_scenario(static,runtime),
                 initial_sha256=static['initial_sha256'], environment_sha256=static['environment_sha256'],
                 static_ref=str(folder/'static.json'), runtime_ref=str(folder/'runtime_result.json'))
        scenario_rows.append(row)
    dump_rows(out/'environments.jsonl',scenario_rows)
    mapping={r['scenario']:r for r in scenario_rows}
    task_rows=[]
    for task in rows(out/'task_inventory.jsonl'):
        env=mapping[task['scenario']]
        status='exclude' if task['binding_errors'] else env['status']
        task_rows.append(dict(**task, environment_status=status,
            sampling_eligible=status=='screen_pass' and task['rubric_accepted'],
            environment_evidence=env['static_ref'], runtime_evidence=env['runtime_ref'],
            initial_sha256=env['initial_sha256'],environment_sha256=env['environment_sha256'],
            exclusion_scope='conservative_shared_environment_quarantine' if status=='exclude' else None))
    dump_rows(out/'tasks.jsonl',task_rows)
    for scope, group in [('all',task_rows),('remaining',[r for r in task_rows if r['remaining']])]:
        for status in ['screen_pass','exclude','review']:
            dump_rows(out/f'{scope}.{status}.jsonl',[r for r in group if r['environment_status']==status])
        dump_rows(out/f'{scope}.sampling_eligible.jsonl',[r for r in group if r['sampling_eligible']])
    with (out/'tasks.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        columns=['task_id','remaining','environment_status','rubric_accepted','sampling_eligible','environment_evidence','runtime_evidence']
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(task_rows)
    remaining=[r for r in task_rows if r['remaining']]
    summary=dict(finished_at=now(),complete=all((out/'scenarios'/s/'runtime_result.json').exists() for s in manifest['scenarios']),
        api_requests=0,total_tasks=len(task_rows),remaining_tasks=len(remaining),
        scenarios=dict(Counter(r['status'] for r in scenario_rows)),
        all_tasks=dict(Counter(r['environment_status'] for r in task_rows)),
        remaining=dict(Counter(r['environment_status'] for r in remaining)),
        remaining_rubric_accepted=sum(r['rubric_accepted'] for r in remaining),
        remaining_sampling_eligible=sum(r['sampling_eligible'] for r in remaining),
        probe_counts=dict(sum((Counter(r['coverage']) for r in scenario_rows),Counter())),
        limits=LIMITS)
    write_json(out/'summary.json',summary)
    admission=dict(version='awm-environment-admission-v1',complete=summary['complete'],created_at=now(),
        screen_manifest=str(out/'manifest.json'),screen_manifest_sha256=file_hash(out/'manifest.json'),
        tasks_ref=str(out/'tasks.jsonl'),tasks_sha256=file_hash(out/'tasks.jsonl'),
        policy={'confirmed_environment_fault':'discard_sample_keep_evidence_no_reward',
                'suspected_fault_or_unknown_write_outcome':'quarantine_sample_no_reward_no_blind_write_retry',
                'normal_business_rejection_or_model_failure':'retain_if_evaluation_valid',
                'judge_or_rubric_fault':'quarantine_evaluation_not_environment_failure',
                'group_policy':'do_not_silently_resample_until_success; record attempted/valid/discarded denominators; skip insufficient valid groups'},
        limits=LIMITS)
    write_json(out/'admission.json',admission)
    body=['# 全量环境预筛查', '', f"完成：{summary['complete']}；覆盖 {len(task_rows)} 题、{len(scenario_rows)} 个场景；剩余集合 {len(remaining)} 题。",'',
          '| 范围 | 通过有限筛查 | 确认缺陷，保守隔离 | 待复核 |','|---|---:|---:|---:|']
    for label,counts in [('全部题库',summary['all_tasks']),('剩余题目',summary['remaining']),('环境',summary['scenarios'])]:
        body.append(f"| {label} | {counts.get('screen_pass',0)} | {counts.get('exclude',0)} | {counts.get('review',0)} |")
    body += ['',f"剩余题中 rubric 已通过 {summary['remaining_rubric_accepted']}；同时满足环境预筛与 rubric 门槛、可进入下一步开发采样的有 {summary['remaining_sampling_eligible']}。",'',
             '共享环境的缺陷按场景保守隔离，不宣称关联每道题都无解。通过有限筛查不等于逐题语义可完成性已证明。', '',
             '工具逐项 HTTP 探测使用独立进程和数据库副本；数据库与生成代码保持冻结。读写输入来自契约示例、schema 或种子数据，不能替代完整业务链验证。4xx 单列为业务/探测拒绝；500 与超时进入复核，不能直接归为模型失败。', '',
             '采样规则：确认数据库/工具故障则作废该次样本，留存轨迹和故障原因且不赋奖励；疑似故障先隔离；正常业务拒绝、参数错误与纯模型失败可作为有效失败保留。未知写入结果不得盲目重放。GRPO 组有效样本不足时跳过或按明确策略重新采集，不反复补采直到成功。', '',
             '本次仅交付独立预筛与采样准入/故障处置接口，没有改动正在运行的 3000 题任务。后续采样入口必须调用该接口，不能仅依据本报告声称已经自动接入。','',
             f"探测统计：`{summary['probe_counts']}`；付费 API 请求 0。", '', '文件：`tasks.csv`、`remaining.exclude.jsonl`、`remaining.review.jsonl`、`remaining.screen_pass.jsonl`、`remaining.sampling_eligible.jsonl`、`admission.json`。']
    (out/'REPORT.md').write_text('\n'.join(body)+'\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','runtime','report'])
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--jobs',type=int,default=8)
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args();out=args.output.resolve()
    if args.mode=='prepare':prepare(out)
    elif args.mode=='runtime':asyncio.run(runtime(out,args.jobs,args.limit))
    else:report(out)


if __name__=='__main__':main()
