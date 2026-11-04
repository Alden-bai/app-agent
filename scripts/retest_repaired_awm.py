#!/usr/bin/env python3
"""Repeatable, model-free release checks; outputs are outside the release."""
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import signal
import sqlite3
import sys
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness_awm.io import file_hash, write_json


def records(path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def database_checks(release, output):
    from harness_awm.database import initialize, quote
    from harness_awm.database_audit import json_default
    assets = {}
    inventory = {}
    for path in sorted(release.glob('gen_*.jsonl')):
        rows = list(records(path))
        keys = [(r['scenario'], r['task_idx']) if path.name == 'gen_verifier.jsonl'
                else r.get('scenario', r.get('name')) for r in rows]
        inventory[path.name] = dict(rows=len(rows), unique_keys=len(set(keys)),
            duplicate_keys=[str(k) for k, n in Counter(keys).items() if n > 1],
            scenarios=sorted({k[0] if isinstance(k, tuple) else k for k in keys}))
        if path.name in ('gen_db.jsonl', 'gen_sample.jsonl', 'gen_tasks.jsonl'):
            assets[path.name] = {r['scenario']: r for r in rows}
        if path.name == 'gen_verifier.jsonl':
            task_map = assets['gen_tasks.jsonl']
            mismatches = []
            for row in rows:
                tasks = task_map.get(row['scenario'], {}).get('tasks', [])
                idx = row['task_idx']
                if not isinstance(idx, int) or not 0 <= idx < len(tasks) or tasks[idx] != row['task']:
                    mismatches.append(dict(scenario=row['scenario'], task_idx=idx))
            inventory[path.name]['task_mismatches'] = mismatches
        del rows
    write_json(output / 'inventory.json', inventory)
    manifest = json.loads((release / 'manifest.json').read_text())
    validation = {r['scenario']: r for r in records(release / 'validation.jsonl')}
    lineage = {r['scenario']: r for r in records(release / 'record_lineage.jsonl')}
    hashes = {name: file_hash(release / name) == expected for name, expected in manifest['output_sha256'].items()}
    results = []
    with (output / 'database.jsonl').open('w') as report:
        for index, (scenario, schema) in enumerate(assets['gen_db.jsonl'].items()):
            path = release / 'databases' / (scenario + '.db')
            prior = validation[scenario]
            result = dict(scenario=scenario, snapshot_hash_matches=file_hash(path) == prior['database_sha256'])
            with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
                result['integrity'] = db.execute('PRAGMA integrity_check').fetchall()
                result['foreign_keys'] = db.execute('PRAGMA foreign_key_check').fetchall()
                result['counts_match'] = all(db.execute(f'SELECT count(*) FROM {quote(t)}').fetchone()[0] == n
                                             for t, n in prior['record_counts'].items())
                errors = []
                for st in lineage.get(scenario, {}).get('statements', []):
                    if not st.get('inserted_table'):
                        continue
                    cur = db.execute(f'SELECT * FROM {quote(st["inserted_table"])} WHERE rowid=?', (st['inserted_rowid'],))
                    row = cur.fetchone()
                    actual = dict(zip([c[0] for c in cur.description], row)) if row else None
                    actual = json.loads(json.dumps(actual, default=json_default))
                    if actual != st['final_record']:
                        errors.append(dict(table=st['table'], index=st['index']))
                result['lineage_errors'] = errors
                result['business_records'] = sum(prior['record_counts'].values())
            rebuilt = initialize(schema['db_schema'], assets['gen_sample.jsonl'][scenario]['sample_data'])
            result['rebuild_complete'] = rebuilt['complete']
            result['rebuild_errors'] = rebuilt['errors']
            result['rebuild_counts_match'] = rebuilt['record_counts'] == prior['record_counts']
            result['passed'] = (result['snapshot_hash_matches'] and result['integrity'] == [('ok',)]
                and not result['foreign_keys'] and result['counts_match'] and not errors
                and rebuilt['complete'] and result['rebuild_counts_match'])
            report.write(json.dumps(result) + '\n'); report.flush()
            results.append(result)
            if (index + 1) % 100 == 0:
                print('database', index + 1, 'failed', sum(not r['passed'] for r in results), flush=True)
    write_json(output / 'database_summary.json', dict(total=len(results), passed=sum(r['passed'] for r in results),
        release_asset_hashes=hashes, failed=[r['scenario'] for r in results if not r['passed']]))


async def runtime_checks(release, output, jobs, limit, scenarios=None):
    import httpx
    from harness_awm.contracts import RunConfig
    from harness_awm.environment import AWMEnvAdapter
    from harness_awm.io import db_fingerprint, digest
    sem = asyncio.Semaphore(jobs)
    results = []
    async def one(env):
        async with sem:
            scenario = env['scenario']
            folder = output / 'runtime' / scenario
            folder.mkdir(parents=True, exist_ok=False)
            env_path = folder / 'environment.json'
            write_json(env_path, env)
            initial = release / 'databases' / (scenario + '.db')
            before = file_hash(initial)
            task = SimpleNamespace(scenario=scenario, initial_state_ref=str(initial), initial_state_hash=before,
                initial_state_fingerprint=db_fingerprint(initial), environment_ref=str(env_path),
                environment_code_hash=digest(env), tool_schema_hash=None, readiness_profile={'runtime_ready': True})
            adapter = AWMEnvAdapter(task, RunConfig(startup_timeout=60, tool_timeout=15), folder)
            result = dict(scenario=scenario, startup=False, calls=[], skipped_get=[])
            try:
                await adapter.prepare()
                result.update(startup=True, tool_count=len(adapter.tools))
                async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
                    spec = (await client.get(f'http://127.0.0.1:{adapter.port}/openapi.json')).json()
                write_json(folder / 'openapi.json', spec)
                tools = {t['name']: t for t in adapter.tools}
                for path, methods in spec.get('paths', {}).items():
                    op = methods.get('get')
                    if not op:
                        continue
                    name = op.get('operationId')
                    tool = tools.get(name)
                    if not tool or tool['inputSchema'].get('required'):
                        result['skipped_get'].append(dict(path=path, tool=name, reason='required_arguments' if tool else 'not_exposed'))
                        continue
                    response = await adapter.call(name, {})
                    result['calls'].append(dict(tool=name, arguments={}, path=path, response=response))
                with sqlite3.connect(adapter.runtime / 'working.db') as db:
                    result['integrity'] = db.execute('PRAGMA integrity_check').fetchall()
                    result['foreign_keys'] = db.execute('PRAGMA foreign_key_check').fetchall()
            except Exception as exc:
                result['error'] = f'{type(exc).__name__}: {exc}'
                result['traceback'] = traceback.format_exc()
            finally:
                result['process_stopped'] = await adapter.close()
                result['snapshot_unchanged'] = file_hash(initial) == before
            result['call_errors'] = sum(bool(c['response'].get('isError')) for c in result['calls'])
            result['passed'] = (result['startup'] and not result.get('error') and not result['call_errors']
                and result.get('integrity') == [('ok',)] and not result.get('foreign_keys')
                and result['process_stopped'] and result['snapshot_unchanged'])
            write_json(folder / 'result.json', result)
            results.append(result)
            with (output / 'runtime_progress.jsonl').open('a') as f:
                f.write(json.dumps({k: v for k, v in result.items() if k not in ('calls', 'traceback', 'skipped_get')}) + '\n')
            if len(results) % 20 == 0 or not result['passed']:
                print('runtime', len(results), scenario, 'startup', result['startup'], 'call_errors', result['call_errors'], result.get('error', ''), flush=True)
    envs = list(records(release / 'gen_envs.jsonl'))
    if scenarios:
        envs = [env for env in envs if env['scenario'] in scenarios]
    if limit:
        envs = envs[:limit]
    await asyncio.gather(*(one(env) for env in envs))
    write_json(output / 'runtime_summary.json', dict(total=len(results), startup_passed=sum(r['startup'] for r in results),
        passed=sum(r['passed'] for r in results), calls=sum(len(r['calls']) for r in results),
        call_errors=sum(r['call_errors'] for r in results), scenarios_with_calls=sum(bool(r['calls']) for r in results),
        failed=[r['scenario'] for r in results if not r['passed']],
        scope='All environment startup/tool discovery; GET tools without required arguments; no task solving'))


def verifier_checks(release, output):
    """Run inside bwrap, with the release read-only and output writable."""
    sys.path.insert(0, str(ROOT / 'agent-world-model'))
    from awm.core.verify import run_verifier
    work = output / 'verifier_work'
    work.mkdir(exist_ok=True)
    sql_errors = []
    connections = []
    class Cursor(sqlite3.Cursor):
        def execute(self, sql, parameters=()):
            try:
                return super().execute(sql, parameters)
            except sqlite3.Error as exc:
                sql_errors.append(dict(sql=sql, error=str(exc)))
                raise
    class Connection(sqlite3.Connection):
        def cursor(self, *args, **kwargs):
            kwargs.setdefault('factory', Cursor)
            return super().cursor(*args, **kwargs)
        def execute(self, sql, parameters=()):
            return self.cursor().execute(sql, parameters)
    original_connect = sqlite3.connect
    def connect(*args, **kwargs):
        kwargs.setdefault('factory', Connection)
        conn = original_connect(*args, **kwargs)
        connections.append(conn)
        return conn
    sqlite3.connect = connect
    def timeout(*args):
        raise TimeoutError('verifier exceeded 5 seconds')
    signal.signal(signal.SIGALRM, timeout)
    results = []
    with (output / 'verifiers.jsonl').open('w') as report:
        for index, entry in enumerate(records(release / 'gen_verifier.jsonl')):
            scenario = entry['scenario']
            src = release / 'databases' / (scenario + '.db')
            initial, final = work / 'initial.db', work / 'final.db'
            for dest in (initial, final):
                if dest.exists():
                    dest.chmod(0o644)
                shutil.copyfile(src, dest)
            sql_errors.clear()
            result = dict(scenario=scenario, task_idx=entry['task_idx'], source_line=index + 1)
            try:
                signal.setitimer(signal.ITIMER_REAL, 5)
                kind, details = run_verifier(entry, 'sql', str(initial), str(final))
                result.update(reward_type=kind, details=details, sql_errors=list(sql_errors),
                              passed=kind != 'judge_error' and not sql_errors)
            except Exception as exc:
                result.update(passed=False, error=f'{type(exc).__name__}: {exc}', sql_errors=list(sql_errors))
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                for conn in connections:
                    conn.close()
                connections.clear()
            report.write(json.dumps(result, default=str, ensure_ascii=False) + '\n'); report.flush()
            results.append({k: v for k, v in result.items() if k != 'details'})
            if (index + 1) % 500 == 0:
                print('verifiers', index + 1, 'failed', sum(not r['passed'] for r in results), flush=True)
                gc.collect()
    write_json(output / 'verifier_summary.json', dict(total=len(results), passed=sum(r['passed'] for r in results),
        failed=sum(not r['passed'] for r in results), scenarios=len({r['scenario'] for r in results}),
        failed_scenarios=sorted({r['scenario'] for r in results if not r['passed']}),
        scope='Initial and final are identical release copies; tests execution, not task completion or verifier correctness'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['database', 'runtime', 'verifiers'])
    parser.add_argument('--release', type=Path, default=ROOT / 'data/awm-dde80a0-repaired-v1')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/repaired_retest_20261103')
    parser.add_argument('--jobs', type=int, default=12)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--scenarios', nargs='+')
    args = parser.parse_args()
    release, output = args.release.resolve(), args.output.resolve()
    if output == release or release in output.parents:
        parser.error('output must be outside the release')
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'database':
        database_checks(release, output)
    elif args.mode == 'runtime':
        asyncio.run(runtime_checks(release, output, args.jobs, args.limit, args.scenarios))
    else:
        verifier_checks(release, output)


if __name__ == '__main__':
    main()
