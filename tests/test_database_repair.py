import sqlite3
from pathlib import Path

import pytest

from harness_awm.database import initialize


def assets(statements):
    return ({"tables": [
        {"name": "p", "ddl": "CREATE TABLE p(id INTEGER PRIMARY KEY, child INTEGER REFERENCES c(id))"},
        {"name": "c", "ddl": "CREATE TABLE c(id INTEGER PRIMARY KEY, parent INTEGER NOT NULL REFERENCES p(id), name TEXT UNIQUE CHECK(length(name)>1))"},
    ]}, {"tables": [{"table_name": "fixture", "insert_statements": statements}]})


GOOD = ["INSERT INTO p VALUES(1,1)", "INSERT INTO c VALUES(1,1,'name')"]


def test_cycle_and_atomic_publication(tmp_path):
    path = tmp_path / "db.sqlite"
    result = initialize(*assets(GOOD), path)
    assert result["complete"]
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
        assert db.execute("SELECT parent FROM c").fetchall() == [(1,)]


@pytest.mark.parametrize("sql", [
    "INSERT INTO c VALUES(2,99,'missing parent')",
    "INSERT INTO c VALUES(2,1,'name')",
    "INSERT INTO c VALUES(2,NULL,'null parent')",
    "INSERT INTO c VALUES(2,1,'x')",
    "INSERT INTO c VALUES(2,1)",
    "INSERT INTO missing VALUES(2)",
])
def test_failure_does_not_publish_partial_or_replace_existing(tmp_path, sql):
    target = tmp_path / "existing.db"
    assert initialize(*assets(GOOD), target)["complete"]
    previous = target.read_bytes()
    result = initialize(*assets(GOOD + [sql]), target)
    assert not result["complete"]
    assert result["errors"] or result["foreign_key_check"]
    assert target.read_bytes() == previous
    new = tmp_path / "new.db"
    assert not initialize(*assets(GOOD + [sql]), new)["complete"]
    assert not new.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_schema_examples_are_in_same_deferred_transaction():
    schema, sample = assets([GOOD[1]])
    schema["tables"][0]["examples"] = [GOOD[0]]
    assert initialize(schema, sample)["complete"]


def test_schema_error_blocks_publication(tmp_path):
    schema, sample = assets(GOOD)
    schema["tables"][0]["indexes"] = ["CREATE INDEX bad ON p(missing)"]
    result = initialize(schema, sample, tmp_path / "db")
    assert not result["complete"]
    assert result["errors"][0]["phase"] == "index"
    assert not (tmp_path / "db").exists()


@pytest.mark.parametrize('sql', ['COMMIT', 'PRAGMA defer_foreign_keys=OFF', 'SAVEPOINT fixture_sp'])
def test_fixture_cannot_escape_initialization_transaction(tmp_path, sql):
    target = tmp_path / 'db'
    result = initialize(*assets([GOOD[0], sql, GOOD[1]]), target)
    assert not result['complete']
    assert result['errors'][0]['code'] == 'SQLITE_AUTH'
    assert not target.exists()


def test_failed_atomic_replace_preserves_target_and_removes_temporary_file(tmp_path, monkeypatch):
    import os
    target = tmp_path / 'db'
    assert initialize(*assets(GOOD), target)['complete']
    old = target.read_bytes()

    def fail(*args):
        raise OSError('publication failure')

    monkeypatch.setattr(os, 'replace', fail)
    with pytest.raises(OSError, match='publication failure'):
        initialize(*assets(GOOD), target)
    assert target.read_bytes() == old
    assert list(tmp_path.iterdir()) == [target]


def test_repaired_polymorphic_charge_foreign_keys():
    import json
    patches = json.loads((Path(__file__).resolve().parents[1] / 'scripts/awm_database_repairs_v1.json').read_text())
    ddl = next(p['after'] for p in patches if p['kind'] == 'schema' and p['scenario'] == 'property_management_1')
    db = sqlite3.connect(':memory:')
    try:
        db.execute('PRAGMA foreign_keys=ON')
        for name, key in [('payments', 1), ('rent_charges', 10), ('fees', 20)]:
            db.execute(f'CREATE TABLE {name}(id INTEGER PRIMARY KEY)')
            db.execute(f'INSERT INTO {name} VALUES(?)', (key,))
        db.execute(ddl)
        insert = 'INSERT INTO payment_applications(payment_id,charge_type,charge_id,applied_amount,applied_date) VALUES(1,?,?,5,\'2026-01-01\')'
        db.execute(insert, ('rent', 10))
        db.execute(insert, ('fee', 20))
        for kind, key in [('rent',20), ('fee',10), ('fee',999), ('unknown',10)]:
            with pytest.raises(sqlite3.IntegrityError):
                db.execute(insert, (kind,key))
        assert db.execute('PRAGMA foreign_key_check').fetchall() == []
    finally:
        db.close()


def test_regression_compares_fts_tokens_not_transaction_segment_blobs():
    from harness_awm.database_audit import compare_unchanged
    schema = {'tables':[{'name':'docs','ddl':'CREATE VIRTUAL TABLE docs USING fts5(body)',
                         'examples':["INSERT INTO docs VALUES('alpha beta')"]}]}
    sample = {'tables':[{'table_name':'docs','insert_statements':["INSERT INTO docs VALUES('beta gamma')"]}]}
    assert compare_unchanged(schema, sample, schema, sample)['equal']
    altered = {'tables':[{'table_name':'docs','insert_statements':["INSERT INTO docs VALUES('beta delta')"]}]}
    assert not compare_unchanged(schema, sample, schema, altered)['equal']


def test_cli_failed_rebuild_returns_nonzero_and_preserves_old_database(tmp_path):
    import json
    import subprocess
    import sys
    source, output = tmp_path / 'source', tmp_path / 'release'
    source.mkdir()
    schema, sample = assets(GOOD)
    (source / 'gen_db.jsonl').write_text(json.dumps({'scenario':'cycle','db_schema':schema})+'\n')
    samples = source / 'gen_sample.jsonl'
    samples.write_text(json.dumps({'scenario':'cycle','sample_data':sample})+'\n')
    patches = tmp_path / 'patches.json'
    patches.write_text('[]')
    script = Path(__file__).resolve().parents[1] / 'scripts/repair_awm_databases.py'
    cmd = [sys.executable,str(script),'--source',str(source),'--output',str(output),'--patches',str(patches),'--scenarios','cycle']
    assert subprocess.run(cmd, capture_output=True).returncode == 0
    target = output / 'databases/cycle.db'
    before = target.read_bytes()
    sample['tables'][0]['insert_statements'].append("INSERT INTO c VALUES(2,999,'missing')")
    samples.write_text(json.dumps({'scenario':'cycle','sample_data':sample})+'\n')
    failed = subprocess.run(cmd, capture_output=True)
    assert failed.returncode == 1, failed.stderr.decode()
    assert target.read_bytes() == before
    assert not json.loads((output / 'manifest.json').read_text())['release_complete']
