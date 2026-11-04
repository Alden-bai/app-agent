"""Read-only source baselines and same-clock logical regression checks."""
import hashlib
import json
import sqlite3
import time

from .database import initialize, quote


def json_default(value):
    if isinstance(value, bytes):
        return {"sqlite_blob_hex": value.hex()}
    raise TypeError(type(value).__name__)


def logical_state(db):
    schema = db.execute("SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name").fetchall()
    shadow = {r[1] for r in db.execute('PRAGMA table_list') if r[2] == 'shadow'}
    tables = {}
    for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        if name in shadow:
            continue
        rows = db.execute(f"SELECT * FROM {quote(name)}").fetchall()
        tables[name] = sorted(json.dumps(row, ensure_ascii=False, default=json_default) for row in rows)
    # FTS segment blobs differ with transaction boundaries. Compare the actual
    # indexed tokens/positions as well as content, not physical segment layout.
    indexes = {}
    for kind, name, _, sql in schema:
        if kind == 'table' and sql and 'using fts5' in sql.lower():
            db.execute(f"CREATE VIRTUAL TABLE temp.awm_audit_vocab USING fts5vocab(main, {quote(name)}, instance)")
            try:
                indexes[name] = db.execute('SELECT * FROM temp.awm_audit_vocab ORDER BY term,doc,col,offset').fetchall()
            finally:
                db.execute('DROP TABLE temp.awm_audit_vocab')
    raw = json.dumps(dict(schema=schema, tables=tables, full_text_indexes=indexes), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def baseline(schema, sample):
    """Mirror released immediate-FK, per-table schema failure boundaries."""
    db = sqlite3.connect(':memory:')
    errors = []

    def execute(sql, phase, table, index):
        try:
            db.execute(sql)
            return True
        except sqlite3.Error as exc:
            errors.append(dict(phase=phase, table=table, index=index, sql=sql,
                               error=str(exc), code=getattr(exc, 'sqlite_errorname', None)))
            return False

    try:
        for table in schema['tables']:
            statements = [('ddl', 0, table['ddl'])]
            statements += [('index', i, sql) for i, sql in enumerate(table.get('indexes', []))]
            statements += [('example', i, sql) for i, sql in enumerate(table.get('examples', []))]
            for phase, index, sql in statements:
                if not execute(sql, phase, table['name'], index):
                    break
        db.commit()
        db.execute('PRAGMA foreign_keys=ON')
        for table in sample.get('tables', []):
            for i, sql in enumerate(table['insert_statements']):
                execute(sql, 'sample', table['table_name'], i)
        db.commit()
        fk = db.execute('PRAGMA foreign_key_check').fetchall()
        integrity = db.execute('PRAGMA integrity_check').fetchall()
        counts = {name: db.execute(f'SELECT count(*) FROM {quote(name)}').fetchone()[0]
                  for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        ordinary = {r[1] for r in db.execute('PRAGMA table_list')
                    if r[2] == 'table' and not r[1].startswith('sqlite_')}
        return dict(complete=not errors and not fk and integrity == [('ok',)],
                    errors=errors, foreign_key_check=fk, integrity_check=integrity,
                    table_counts=counts, record_counts={n: c for n, c in counts.items() if n in ordinary},
                    logical_sha256=logical_state(db))
    finally:
        db.close()


def compare_unchanged(schema, sample, repaired_schema, repaired_sample):
    # Both builds must occur in one UTC second: preserve all relative expressions,
    # SQLite defaults and full row contents, without hiding any time column.
    for _ in range(5):
        start = int(time.time())
        old = baseline(schema, sample)
        new = initialize(repaired_schema, repaired_sample,
                         inspect=lambda db, *_: logical_state(db))
        if int(time.time()) == start:
            return dict(equal=old['logical_sha256'] == new['details'],
                        original_sha256=old['logical_sha256'], repaired_sha256=new['details'],
                        comparison_epoch=start, method='both builds within same UTC second')
    raise RuntimeError('Could not obtain a same-second regression comparison')
