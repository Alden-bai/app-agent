"""Bounded, read-only evidence from two immutable SQLite snapshots."""
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from .io import canonical, file_hash, write_json


VERSION = "awm-evidence-v2"


def quote(name):
    return '"' + name.replace('"', '""') + '"'


class EvidenceStore:
    def __init__(self, initial, final, *, folder=None, seconds=3, max_rows=100, max_bytes=24000):
        self.paths = {"initial": Path(initial).resolve(), "final": Path(final).resolve()}
        self.hashes = {k: file_hash(p) for k, p in self.paths.items()}
        self.folder = Path(folder) if folder else None
        self.seconds, self.max_rows, self.max_bytes = seconds, max_rows, max_bytes
        self.records = {}

    def add(self, kind, content, *, valid=True, **metadata):
        eid = f"E{len(self.records) + 1:05d}"
        row = {"id": eid, "kind": kind, "valid": valid, "content": content, **metadata}
        self.records[eid] = row
        if self.folder:
            write_json(self.folder / f"{eid}.json", row)
        return row

    def verify(self):
        if any(file_hash(p) != self.hashes[k] for k, p in self.paths.items()):
            raise ValueError("snapshot_changed_during_evaluation")

    @contextmanager
    def connect(self, target, *, internal=False):
        if target not in self.paths:
            raise ValueError("snapshot_must_be_initial_or_final")
        with sqlite3.connect(self.paths[target].as_uri() + "?mode=ro", uri=True) as db:
            db.execute("PRAGMA query_only=ON")
            db.enable_load_extension(False)
            db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, max(1_048_576, self.max_bytes * 4))
            db.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 100_000)
            allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}

            def authorize(action, arg1, arg2, *_):
                if internal and action == sqlite3.SQLITE_PRAGMA and arg1 in {"table_info", "foreign_key_list", "index_list", "index_info"}:
                    return sqlite3.SQLITE_OK
                if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in {"load_extension", "writefile", "readfile"}:
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

            db.set_authorizer(authorize)
            deadline = time.monotonic() + self.seconds
            db.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
            yield db

    def limited(self, cursor):
        columns = [c[0] for c in cursor.description]
        rows = cursor.fetchmany(self.max_rows + 1)
        content = {"columns": columns, "rows": [
            [{"blob_hex": v.hex()} if isinstance(v, bytes) else v for v in row]
            for row in rows[:self.max_rows]]}
        truncated = len(rows) > self.max_rows
        while len(canonical(content).encode()) > self.max_bytes and content["rows"]:
            content["rows"].pop()
            truncated = True
        if len(canonical(content).encode()) > self.max_bytes:
            raise ValueError("column_metadata_exceeds_output_limit")
        return content, truncated

    def query_snapshot(self, snapshot, sql, params=None):
        with self.connect(snapshot) as db:
            # execute rejects multiple statements; no ATTACH, PRAGMA or file I/O.
            content, truncated = self.limited(db.execute(sql, params or []))
        return content, truncated

    def inspect_schema(self, snapshot="final", table=None):
        with self.connect(snapshot, internal=True) as db:
            if table is None:
                return self.limited(db.execute("SELECT name, type, sql FROM sqlite_master WHERE type IN ('table','view') ORDER BY name"))
            if not db.execute("SELECT 1 FROM sqlite_master WHERE name=? AND type='table'", (table,)).fetchone():
                raise ValueError("unknown_table")
            columns = db.execute(f"PRAGMA table_info({quote(table)})").fetchall()
            foreign_keys = db.execute(f"PRAGMA foreign_key_list({quote(table)})").fetchall()
            value = {"table": table, "columns": [dict(zip(("cid", "name", "type", "notnull", "default", "pk"), r)) for r in columns],
                     "foreign_keys": [dict(zip(("id", "seq", "table", "from", "to", "on_update", "on_delete", "match"), r)) for r in foreign_keys]}
            if len(canonical(value).encode()) > self.max_bytes:
                raise ValueError("schema_exceeds_output_limit; query_specific_columns")
            return value, False

    def compare_records(self, table, key):
        if not isinstance(key, dict) or not key or any(v is None for v in key.values()):
            raise ValueError("non_null_object_key_required")
        result = {}
        for snapshot in self.paths:
            with self.connect(snapshot, internal=True) as db:
                columns = db.execute(f"PRAGMA table_info({quote(table)})").fetchall()
                primary = [c[1] for c in sorted(columns, key=lambda c: c[5]) if c[5]]
                keys = [set(primary)] if primary else []
                for index in db.execute(f"PRAGMA index_list({quote(table)})").fetchall():
                    if index[2] and not index[4]:  # complete, unique index; no partial index
                        names = [r[2] for r in db.execute(f"PRAGMA index_info({quote(index[1])})").fetchall()]
                        if names and all(names):
                            keys.append(set(names))
                if set(key) not in keys:
                    raise ValueError("reliable_primary_or_unique_key_required; use_query_snapshot")
            sql = f"SELECT * FROM {quote(table)} WHERE " + " AND ".join(f"{quote(k)}=?" for k in key)
            data, truncated = self.query_snapshot(snapshot, sql, list(key.values()))
            if truncated or len(data["rows"]) > 1:
                raise ValueError("object_comparison_incomplete_or_nonunique")
            result[snapshot] = data
        result["changed"] = result["initial"] != result["final"]
        return result, False

    def read_evidence(self, evidence_id, offset=0):
        if evidence_id not in self.records or type(offset) is not int or offset < 0:
            raise ValueError("invalid_evidence_page")
        record = self.records[evidence_id]
        if not record["valid"] or record.get("error"):
            raise ValueError("cannot_promote_failed_evidence")
        text = canonical(record["content"])
        if offset > len(text):
            raise ValueError("evidence_offset_out_of_bounds")
        if self.max_bytes <= 512:
            raise ValueError("evidence_page_output_limit_too_small")
        # Character offsets, with a byte bound. Original evidence remains immutable.
        page = text[offset:offset + 2500]
        while len(page.encode()) > min(self.max_bytes - 256, 2500):
            page = page[:max(0, len(page) - 128)]
        end = offset + len(page)
        return {"source_evidence_id": evidence_id, "source_kind": record["kind"], "offset": offset,
                "next_offset": end if end < len(text) else None, "total_chars": len(text), "text": page}, offset > 0 or end < len(text) or record.get("truncated", False)

    def call(self, tool, arguments, *, objective, affected_items):
        metadata = {"tool": tool, "arguments": arguments, "objective": objective,
                    "affected_items": affected_items, "snapshot_hashes": self.hashes}
        try:
            if tool not in {"inspect_schema", "query_snapshot", "compare_records", "read_evidence"}:
                raise ValueError("unsupported_evidence_tool")
            content, truncated = getattr(self, tool)(**arguments)
            kind = "schema" if tool == "inspect_schema" else "evidence_page" if tool == "read_evidence" else "query"
            return self.add(kind, content, truncated=truncated, **metadata)
        except (sqlite3.Error, ValueError, TypeError) as exc:
            return self.add("query_error", None, valid=False, truncated=False,
                            error=f"{type(exc).__name__}: {exc}", **metadata)


TOOL_HELP = {
    "inspect_schema": {"snapshot": "initial|final", "table": "optional real table name; omit to list tables"},
    "query_snapshot": {"snapshot": "initial|final", "sql": "ONE read-only SQLite SELECT, table names local to the selected snapshot", "params": "optional positional list or named parameter object"},
    "compare_records": {"table": "real table name", "key": "object of all primary-key or unique-key columns to non-null values"},
    "read_evidence": {"evidence_id": "existing program-assigned evidence ID", "offset": "optional character offset, default 0; follow next_offset"},
}


def model_view(records, max_bytes=60000):
    """Bounded previews; metadata never claims the preview is complete evidence."""
    result, used = {}, 0
    for eid, row in reversed(list(records.items())):
        text = canonical(row["content"])
        allowance = max(0, min(3200, max_bytes - used))
        preview = text.encode()[:allowance].decode("utf-8", errors="ignore")
        complete = preview == text
        view = {k: row[k] for k in ("id", "kind", "valid", "error", "truncated", "objective", "affected_items") if k in row}
        view.update(content=row["content"] if complete else {"preview": preview, "total_chars": len(text), "read_with": "read_evidence"},
                    view_truncated=not complete)
        if row["kind"] == "source":
            view["source_id"] = row["content"]["id"]
        result[eid] = view
        used += len(canonical(view).encode())
    return dict(reversed(list(result.items())))
