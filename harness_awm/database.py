"""Transactional SQLite initialization for explicitly repaired AWM releases."""
import os
import hashlib
import re
from pathlib import Path
import sqlite3
import tempfile


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def initialize(schema, sample, target=None, *, inspect=None, audit=False):
    """Build in memory; publish only after all statements and constraints pass.

    Errors retain statement locations. Inspection runs before rollback for repair
    diagnostics; failed data is never exported as a usable database.
    """
    errors = []
    db = sqlite3.connect(":memory:")
    db.set_authorizer(lambda action, *args: sqlite3.SQLITE_DENY
                      if action in (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH)
                      else sqlite3.SQLITE_OK)
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("BEGIN")
    db.execute("PRAGMA defer_foreign_keys=ON")
    executed = 0
    statements = []

    def execute(sql, phase, table, index):
        nonlocal executed
        # Reject top-level transaction controls, while allowing FTS's internal
        # savepoints. The SQLite authorizer cannot distinguish those contexts.
        head = re.sub(r'\A\s*(?:(?:--[^\n]*(?:\n|$)|/\*.*?\*/)\s*)*', '', sql, flags=re.S)
        if re.match(r'(BEGIN|END|COMMIT|ROLLBACK|SAVEPOINT|RELEASE|PRAGMA)\b', head, re.I):
            errors.append(dict(phase=phase, table=table, index=index, sql=sql,
                               error='Fixture transaction control is not allowed', code='SQLITE_AUTH'))
            return
        try:
            before = db.total_changes
            cursor = db.execute(sql)
            executed += 1
            if audit:
                match = re.match(r'\s*INSERT\s+INTO\s+["`\[]?(\w+)', sql, re.I)
                statements.append(dict(phase=phase, table=table, index=index,
                    sql_sha256=hashlib.sha256(sql.encode()).hexdigest(),
                    changes=db.total_changes - before,
                    inserted_table=match.group(1) if match else None,
                    inserted_rowid=cursor.lastrowid if match else None))
        except sqlite3.Error as exc:
            errors.append(dict(phase=phase, table=table, index=index, sql=sql,
                               error=str(exc), code=getattr(exc, "sqlite_errorname", None)))

    try:
        for table in schema["tables"]:
            execute(table["ddl"], "ddl", table["name"], 0)
        for table in schema["tables"]:
            for index, sql in enumerate(table.get("indexes", [])):
                execute(sql, "index", table["name"], index)
        for table in schema["tables"]:
            for index, sql in enumerate(table.get("examples", [])):
                execute(sql, "example", table["name"], index)
        for table in sample.get("tables", []):
            for index, sql in enumerate(table["insert_statements"]):
                execute(sql, "sample", table["table_name"], index)
        fk, integrity = [], []
        for name, values in [('foreign_key_check', fk), ('integrity_check', integrity)]:
            try:
                values.extend(db.execute('PRAGMA ' + name).fetchall())
            except sqlite3.Error as exc:
                errors.append(dict(phase='integrity', table=None, index=0, sql='PRAGMA ' + name,
                                   error=str(exc), code=getattr(exc, 'sqlite_errorname', None)))
        counts = {name: db.execute(f"SELECT count(*) FROM {quote(name)}").fetchone()[0]
                  for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")}
        ordinary = {r[1] for r in db.execute('PRAGMA table_list')
                    if r[2] == 'table' and not r[1].startswith('sqlite_')}
        record_counts = {name: count for name, count in counts.items() if name in ordinary}
        details = inspect(db, errors, fk) if inspect else None
        if audit:
            for statement in statements:
                if statement["inserted_table"]:
                    cursor = db.execute(f'SELECT * FROM {quote(statement["inserted_table"])} WHERE rowid=?',
                                        (statement["inserted_rowid"],))
                    row = cursor.fetchone()
                    statement["final_record"] = dict(zip([c[0] for c in cursor.description], row)) if row else None
        complete = not errors and not fk and integrity == [("ok",)]
        if complete:
            db.commit()
            if target is not None:
                target = Path(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                fd, temp = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
                os.close(fd)
                try:
                    with sqlite3.connect(temp) as dest:
                        db.backup(dest)
                    with open(temp, "rb") as f:
                        os.fsync(f.fileno())
                    os.replace(temp, target)
                finally:
                    if os.path.exists(temp):
                        os.unlink(temp)
        else:
            db.rollback()
        return dict(complete=complete, errors=errors, foreign_key_check=fk,
                    integrity_check=integrity, table_counts=counts,
                    executed_statements=executed, details=details, statements=statements,
                    record_counts=record_counts)
    finally:
        db.close()
