#!/usr/bin/env python3
"""Produce compact JSON and CSV findings from a completed repaired-release retest."""
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness_awm.io import file_hash, write_json

release = ROOT / 'data/awm-dde80a0-repaired-v1'
output = ROOT / 'runs/repaired_retest_20261103'


def read(path):
    return json.loads(path.read_text())


def csv_write(name, rows, columns):
    with (output / name).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


# Require completed sweeps before reporting final totals.
for name in ('runtime_summary.json', 'verifier_summary.json', 'database_summary.json', 'orm_dates.json'):
    if not (output / name).exists():
        raise SystemExit('Not finished: ' + name)

env_locations = {}
env_routes = {}
for line_no, line in enumerate((release / 'gen_envs.jsonl').open(), 1):
    env = json.loads(line)
    env_locations[env['scenario']] = line_no
    routes = set()
    for node in ast.walk(ast.parse(env['full_code'])):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                if (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                    and dec.func.attr in ('get', 'post', 'put', 'patch', 'delete')
                    and dec.args and isinstance(dec.args[0], ast.Constant)):
                    routes.add((dec.args[0].value, dec.func.attr))
    env_routes[env['scenario']] = routes

runtime = {}
folders = {}
attempts = []
for parent in (output, output / 'retries', output / 'retries2'):
    for path in sorted((parent / 'runtime').glob('*/result.json')):
        row = read(path)
        attempts.append(dict(scenario=row['scenario'], result=str(path.relative_to(output)),
                             error=row.get('error'), startup=row['startup'], passed=row['passed']))
        runtime[row['scenario']] = row
        folders[row['scenario']] = path.parent

call_findings = []
startup_findings = []
integrity_findings = []
route_mismatches = []
for scenario, row in runtime.items():
    folder = folders[scenario]
    diagnostics = {d['tool']: d for d in read(folder / 'diagnostics.json')} if (folder / 'diagnostics.json').exists() else {}
    if row.get('error'):
        startup_findings.append(dict(scenario=scenario, error=row['error'], evidence=str(folder.relative_to(output)),
                                     environment_jsonl_line=env_locations[scenario]))
    if row.get('foreign_keys') or (row.get('integrity') and row['integrity'] != [['ok']]):
        integrity_findings.append(dict(scenario=scenario, foreign_keys=row.get('foreign_keys'),
            integrity=row.get('integrity'), evidence=str(folder.relative_to(output))))
    if (folder / 'openapi.json').exists():
        spec = read(folder / 'openapi.json')
        actual = {(path, method) for path, ops in spec['paths'].items() for method in ops}
        missing = env_routes[scenario] - actual
        if missing:
            route_mismatches.append(dict(scenario=scenario, missing=sorted(missing)))
    for call in row['calls']:
        if not call['response'].get('isError'):
            continue
        message = '\n'.join(c.get('text', '') for c in call['response'].get('content', []))
        status = re.search(r'Status code: (\d+)', message)
        diag = diagnostics.get(call['tool'], {})
        call_findings.append(dict(scenario=scenario, tool=call['tool'], path=call['path'], arguments=call['arguments'],
            status=int(status[1]) if status else None, message=message, diagnostic=diag.get('error', diag.get('body')),
            environment_jsonl_line=env_locations[scenario], evidence=str(folder.relative_to(output))))
write_json(output / 'runtime_findings.json', dict(startup_errors=startup_findings, call_errors=call_findings,
    post_read_integrity_errors=integrity_findings, route_correspondence_mismatches=route_mismatches, attempts=attempts))
csv_write('runtime_setup_errors.csv', startup_findings, ['scenario', 'error', 'environment_jsonl_line', 'evidence'])
csv_write('runtime_integrity_errors.csv', integrity_findings, ['scenario', 'foreign_keys', 'integrity', 'evidence'])
csv_write('runtime_errors.csv', call_findings,
    ['scenario', 'tool', 'path', 'arguments', 'status', 'diagnostic', 'message', 'environment_jsonl_line', 'evidence'])

verifiers = defaultdict(list)
for line in (output / 'verifiers.jsonl').open():
    row = json.loads(line)
    verifiers[row['scenario'], row['task_idx']].append(row)
verifier_findings = []
duplicates = []
def hard_error(row):
    return row.get('reward_type') == 'judge_error' or bool(row.get('error'))
for key, rows in verifiers.items():
    if len(rows) > 1:
        duplicates.append(dict(scenario=key[0], task_idx=key[1], source_lines=[r['source_line'] for r in rows],
            first_hard_error=hard_error(rows[0]), last_hard_error=hard_error(rows[-1])))
    for row in rows:
        if hard_error(row) or row['sql_errors']:
            verifier_findings.append(dict(scenario=key[0], task_idx=key[1], source_line=row['source_line'],
                first_record=row is rows[0], last_record=row is rows[-1], hard_error=hard_error(row),
                error=row.get('error') or (row.get('details') or {}).get('error_message') or (row.get('details') or {}).get('error'),
                sql_errors=row['sql_errors']))
write_json(output / 'verifier_findings.json', dict(duplicate_tasks=duplicates, errors_and_warnings=verifier_findings))
csv_write('verifier_errors.csv', verifier_findings,
    ['scenario', 'task_idx', 'source_line', 'first_record', 'last_record', 'hard_error', 'error', 'sql_errors'])
csv_write('duplicate_verifiers.csv', duplicates,
    ['scenario', 'task_idx', 'source_lines', 'first_hard_error', 'last_hard_error'])

dates = read(output / 'orm_dates.json')
date_findings = [dict(scenario=r['scenario'], environment_jsonl_line=env_locations[r['scenario']], **e)
                 for r in dates['results'] for e in r['invalid_dates']]
csv_write('invalid_dates.csv', date_findings,
    ['scenario', 'table', 'column', 'orm_type', 'value', 'affected_rows', 'error', 'environment_jsonl_line', 'code_line'])

manifest = read(release / 'manifest.json')
unchanged_assets = {name: file_hash(release / name) == expected for name, expected in manifest['output_sha256'].items()}
unchanged_dbs = all(file_hash(release / 'databases' / (r['scenario'] + '.db')) == r['database_sha256']
                    for r in map(json.loads, (release / 'validation.jsonl').open()))
first = [rows[0] for rows in verifiers.values()]
last = [rows[-1] for rows in verifiers.values()]
summary = dict(finished_at=datetime.now(timezone.utc).isoformat(), release=str(release),
    database=read(output / 'database_summary.json'),
    runtime=dict(scenarios=len(runtime), startup_passed=sum(r['startup'] for r in runtime.values()),
        infrastructure_or_schema_errors=len(startup_findings), clean_scenarios=sum(r['passed'] for r in runtime.values()),
        scenarios_with_calls=sum(bool(r['calls']) for r in runtime.values()),
        calls=sum(len(r['calls']) for r in runtime.values()), failed_calls=len(call_findings),
        call_error_scenarios=len({r['scenario'] for r in call_findings}),
        post_read_integrity_errors=integrity_findings,
        call_error_http_statuses=dict(Counter(str(r['status']) for r in call_findings)),
        skipped_get_operations=sum(len(r['skipped_get']) for r in runtime.values()),
        route_correspondence_mismatches=route_mismatches,
        all_processes_stopped=all(r['process_stopped'] for r in runtime.values())),
    orm_dates={k: v for k, v in dates.items() if k != 'results'},
    verifiers=dict(records=sum(map(len, verifiers.values())), unique_tasks=len(verifiers), duplicate_task_keys=len(duplicates),
        official_first_record_hard_errors=sum(hard_error(r) for r in first),
        official_first_record_error_scenarios=len({r['scenario'] for r in first if hard_error(r)}),
        last_record_hard_errors=sum(hard_error(r) for r in last),
        caught_sql_warning_tasks=sum(bool(r['sql_errors']) and not hard_error(r) for r in last),
        last_record_hard_error_tasks=[dict(scenario=r['scenario'], task_idx=r['task_idx'], source_line=r['source_line']) for r in last if hard_error(r)]),
    original_release_unchanged=all(unchanged_assets.values()) and unchanged_dbs,
    limits=['GET tools without required arguments only; write regression limited to seven scenarios.',
            'Schema errors can stop adapter setup before tool calls.',
            'Verifiers executed with identical initial/final copies; no claim of task success or complete branch coverage.',
            'Caught SQL errors are warnings for review; some are intentional compatibility fallbacks.',
            'Date affected_rows counts field occurrences, not distinct records across columns.',
            'No paid model calls; original release unchanged.'])
shadowing = read(output / 'route_shadowing.json')
route_confirmations = [dict(scenario=p.parent.name, **r)
    for p in (output / 'runtime').glob('*/route_matches.json') for r in read(p)
    if not r['returned_mcp_error']]
summary['route_shadowing'] = dict(static_get_routes=len(shadowing),
    static_scenarios=len({r['scenario'] for r in shadowing}),
    called_routes=sum(r['called'] for r in shadowing),
    silent_wrong_route_calls_confirmed=len(route_confirmations))
csv_write('silent_route_errors.csv', route_confirmations,
    ['scenario', 'tool', 'intended_path', 'actual_path', 'actual_handler', 'returned_mcp_error'])
summary['seven_case_write_regression'] = {k: v for k, v in read(output / 'mcp_regression_enabled/summary.json').items()
                                        if k in ('passed', 'total')}
summary['unit_tests'] = dict(passed=28, subtests_passed=12, log='logs/awm-retest-unit-venv.log')
write_json(output / 'summary.json', summary)
report = f'''# 修复版 1000 场景复测

结论：数据库初始化修复通过，但仍有数据格式、工具实现与验证器问题。原发布版文件及数据库哈希保持不变；无付费模型调用。

| 检查 | 本次结果 |
|---|---|
| 数据库快照、完整性、外键、记录核对、重新初始化 | 1000 / 1000 通过 |
| 服务启动及 MCP 工具目录/schema 检查 | 998 / 1000 通过；2 个工具 schema 不合法 |
| 实际 GET 工具调用 | 980 个场景，6900 次调用 |
| 显式接口错误 | 136 个场景，202 次；500 错误 116 次，422 错误 86 次 |
| 返回成功但路由匹配错误 | {len(route_confirmations)} 次，已用实际 FastAPI 路由器确认 |
| 读取后数据库外键违规 | 3 个场景，已独立复现 |
| ORM 日期字段解析 | 检查 249320 个值；45 个场景有 127 处非法日期字段值 |
| 验证器记录 | 10476 条对应 10000 个任务；476 个任务存在内容不同的重复版本 |
| 按官方“取第一条”行为 | 478 个任务的验证器直接报错，涉及 387 个场景 |
| 改按最后一条取用的对照结果 | 仍有 3 个直接报错；另 5 个任务捕获 SQL 异常后继续返回，需逐项判断 |
| 既有七类 MCP 读写回归 | 7 / 7 通过 |
| 数据库及 Harness 单元测试 | 28 passed，12 subtests passed |

## 优先处理的问题

1. 三个 GET 接口会返回成功并制造无效外键：`e_commerce_marketplace_1.search_vehicles`、`e_commerce_marketplace_10.list_my_listings`、`job_board_8.search_job_postings`。服务刚启动时外键正常；调用相应工具后才出现违规。固定路径被前面的动态路径匹配，错误处理函数自动插入占位记录。[逐步复现](integrity_reproductions/)
2. 日期字段仍有格式错误，例如 `learning_management_system_2.courses.end_at = 2026-06:01 23:59:59`。SQLite 完整性检查接受这些字符串，但 SQLAlchemy 读取会失败。127 是字段出现次数，跨列可能属于同一记录。[全部日期问题](invalid_dates.csv)
3. 两个工具 schema 的 `required` 数组重复：`hr_payroll_management_1` 的 `company_id`，`social_networking_4` 的 `interest_tag_id`。单独重试仍失败。[工具准备错误](runtime_setup_errors.csv)
4. 工具层还包括路由冲突、代码异常、响应模型类型不匹配、字段映射错误等。[显式接口错误](runtime_errors.csv)；[返回成功的错误路由](silent_route_errors.csv)
5. 476 个重复验证器不能直接视为已生效修复。官方 `find_scenario_entry` 取首条，本仓库 registry 对内容不同的重复记录报 `ambiguous_asset`。即使取末条，`e_commerce_8` task_idx=9、`logistics_dispatch_1` task_idx=6、`workflow_automation_6` task_idx=9 仍报错；已用原始官方执行入口复现。下标从 0 开始。[重复版本](duplicate_verifiers.csv)；[验证器异常及警告](verifier_errors.csv)

## 覆盖范围和限制

全量覆盖 1000 个场景的数据库检查和服务准备尝试。自动读取调用限于不需要必填参数的 GET 工具，980 个场景具备实际调用覆盖；9755 个需要参数或未暴露的 GET 操作未执行。写入仅复跑既有 7 个场景。连接类偶发异常已单独重试，最终工具准备错误仅保留两处可复现的 schema 问题。

验证器使用相同内容的初始/最终数据库副本，检查能否执行；没有让模型解答全部业务任务，也没有证明验证器所有分支或判定逻辑正确。捕获后继续的 SQL 异常单列为警告，部分可能是有意兼容处理。

静态路由扫描另外发现 305 个固定 GET 路由被前置路径覆盖的候选，涉及 225 个场景，其中 108 个路径实际调用过；静态结果不与动态错误数直接相加。服务启动与读取期间共有 29 个场景出现记录变化，其中部分可能是默认对象创建；只有已复现的 3 个外键违规直接列为数据损坏。

全部原始响应、工具目录、服务代码、日志、初始与工作数据库、逐验证器返回结果均保存在此目录。详见 [机器可读汇总](summary.json)、[ORM 日期及字段映射明细](orm_dates.json)、[运行时完整发现](runtime_findings.json)。
'''
(output / 'REPORT.md').write_text(report, encoding='utf-8')
print(json.dumps({k: v for k, v in summary.items() if k not in ('orm_dates', 'limits')}, ensure_ascii=False, indent=2))
