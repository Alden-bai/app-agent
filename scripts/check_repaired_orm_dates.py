#!/usr/bin/env python3
"""Check values mapped to ORM date/time fields using SQLAlchemy's real parsers."""
import ast
import json
from pathlib import Path
import sqlite3
import sys
from sqlalchemy.engine import processors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness_awm.database import quote
from harness_awm.io import write_json

release = ROOT / 'data/awm-dde80a0-repaired-v1'
output = ROOT / 'runs/repaired_retest_20261103'
parsers = {'DateTime': processors.str_to_datetime, 'TIMESTAMP': processors.str_to_datetime,
           'Date': processors.str_to_date, 'Time': processors.str_to_time}
results = []
for line in (release / 'gen_envs.jsonl').open():
    env = json.loads(line)
    result = dict(scenario=env['scenario'], checked_values=0, invalid_dates=[], schema_mismatches=[])
    tree = ast.parse(env['full_code'])
    with sqlite3.connect(f"file:{release / 'databases' / (env['scenario'] + '.db')}?mode=ro", uri=True) as db:
        for cls in [node for node in tree.body if isinstance(node, ast.ClassDef)]:
            assigns = {n.targets[0].id: n.value for n in cls.body if isinstance(n, ast.Assign)
                       and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
            tab = assigns.get('__tablename__')
            if not isinstance(tab, ast.Constant) or not isinstance(tab.value, str):
                continue
            table = tab.value
            columns = {r[1] for r in db.execute(f'PRAGMA table_xinfo({quote(table)})')}
            for attr, value in assigns.items():
                if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name) or value.func.id != 'Column':
                    continue
                args = value.args
                column = args[0].value if args and isinstance(args[0], ast.Constant) and isinstance(args[0].value, str) else attr
                if column not in columns:
                    # SQLite rowid aliases are queryable without appearing in table_xinfo.
                    try:
                        db.execute(f'SELECT {quote(column)} FROM {quote(table)} LIMIT 0')
                        if column.lower() in ('rowid', '_rowid_', 'oid'):
                            continue
                    except sqlite3.Error:
                        pass
                    result['schema_mismatches'].append(dict(table=table, column=column, code_line=value.lineno))
                    continue
                types = [a.func.id if isinstance(a, ast.Call) and isinstance(a.func, ast.Name)
                         else a.id if isinstance(a, ast.Name) else '' for a in args]
                dtype = next((t for t in types if t in parsers), None)
                if not dtype:
                    continue
                for raw, count in db.execute(f'SELECT {quote(column)}, count(*) FROM {quote(table)} WHERE {quote(column)} IS NOT NULL GROUP BY {quote(column)}'):
                    result['checked_values'] += count
                    try:
                        parsers[dtype](raw)
                    except Exception as exc:
                        result['invalid_dates'].append(dict(table=table, column=column, orm_type=dtype,
                            value=raw, affected_rows=count, code_line=value.lineno, error=f'{type(exc).__name__}: {exc}'))
    results.append(result)
write_json(output / 'orm_dates.json', dict(scenarios=len(results), checked_values=sum(r['checked_values'] for r in results),
    invalid_values=sum(len(r['invalid_dates']) for r in results),
    affected_rows=sum(e['affected_rows'] for r in results for e in r['invalid_dates']),
    invalid_date_scenarios=[r['scenario'] for r in results if r['invalid_dates']],
    schema_mismatch_scenarios=[r['scenario'] for r in results if r['schema_mismatches']], results=results))
print(json.dumps({k: v for k, v in json.loads((output / 'orm_dates.json').read_text()).items() if k != 'results'}))
