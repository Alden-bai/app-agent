from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AWM = ROOT / "agent-world-model"
DATASET = "Snowflake/AgentWorldModel-1K"
SEED = 20261021
FILES = ("gen_scenario.jsonl", "gen_tasks.jsonl", "gen_db.jsonl",
         "gen_sample.jsonl", "gen_spec.jsonl", "gen_envs.jsonl",
         "gen_verifier.jsonl", "gen_verifier.pure_code.jsonl")


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    if isinstance(value, Path):
        return hashlib.sha256(value.read_bytes()).hexdigest()
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    return hashlib.sha256(value).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".partial-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            f.write("\n")
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def backup(source, target):
    """Copy committed SQLite state, including WAL, without raw file copying."""
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target:
        raise ValueError("SQLite snapshot source and target must differ")
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as src:
        with sqlite3.connect(target) as dst:
            src.backup(dst)


def query(path, sql, params=()):
    with sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        return [dict(r) for r in db.execute(sql, params)]


def code_hash():
    return digest({str(p.relative_to(ROOT)): digest(p) for p in sorted(
        (ROOT / "awm_audit").rglob("*.py")) if "tests" not in p.parts and "rules" not in p.parts})
