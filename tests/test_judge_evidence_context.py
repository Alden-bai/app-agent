import copy
import json
import sqlite3

import pytest

from harness_awm.io import canonical
from harness_awm.judge_evidence import EvidenceStore, model_view


@pytest.fixture
def store(tmp_path):
    paths = [tmp_path / f"{name}.db" for name in ("initial", "final")]
    for path in paths:
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE items(id INTEGER PRIMARY KEY)")
    return EvidenceStore(*paths)


def add_query_noise(store, count=80):
    for i in range(count):
        store.add("query", {"rows": [["noise" * 1000]]}, tool="query_snapshot",
                  arguments={"snapshot": "final", "sql": f"SELECT {i}"},
                  snapshot_hashes=store.hashes, truncated=False)


def test_execution_facts_remain_visible_after_many_queries(store):
    runtime = store.add("runtime", {"actual_start_utc": "2026-11-09T16:55:52Z", "timezone": "Asia/Shanghai"})
    trace = store.add("trajectory", [{"iteration": 1, "tool_calls": [{"name": "create_item"}]}])
    answer = store.add("final_answer", "Created item 1.")
    add_query_noise(store)
    before = copy.deepcopy(store.records)
    view = model_view(store.records, max_bytes=40000)
    for record in (runtime, trace, answer):
        shown = view[record["id"]]
        assert not shown["view_truncated"]
        assert shown["content"] == record["content"]
    assert list(view) == list(store.records)
    assert len(view) == 83
    assert any(row["view_truncated"] for row in view.values())
    latest_query = view[next(reversed(store.records))]
    assert latest_query["arguments"]["snapshot"] == "final"
    assert latest_query["snapshot_hashes"] == store.hashes
    assert store.records == before


def test_only_latest_record_of_each_execution_kind_gets_priority(store):
    old = [store.add(kind, {"version": "old", "body": "x" * 4000})
           for kind in ("runtime", "trajectory", "final_answer")]
    current = [store.add(kind, {"version": "current", "body": "x" * 4000})
               for kind in ("runtime", "trajectory", "final_answer")]
    add_query_noise(store)
    view = model_view(store.records, max_bytes=12000)
    for record in current:
        shown = view[record["id"]]
        assert shown["view_truncated"]
        assert len(shown["content"]["preview"].encode()) == 3200
    for record in old:
        assert view[record["id"]]["content"]["preview"] == ""


@pytest.mark.parametrize("max_bytes", [0, 3200, 12000])
def test_pinned_previews_remain_utf8_bounded_and_honestly_truncated(store, max_bytes):
    records = [store.add(kind, "中文" * 4000)
               for kind in ("runtime", "trajectory", "final_answer")]
    before = copy.deepcopy(store.records)
    view = model_view(store.records, max_bytes=max_bytes)
    for record in records:
        shown = view[record["id"]]
        assert shown["view_truncated"]
        preview = shown["content"]["preview"]
        assert len(preview.encode()) <= 3200
        assert canonical(record["content"]).startswith(preview)
        assert shown["content"]["total_chars"] == len(canonical(record["content"]))
        if max_bytes == 0:
            assert preview == ""
    assert store.records == before


def test_pagination_cursors_survive_hidden_previews_and_reconstruct_source(store):
    source = store.add("tool_result", {"body": "中文" * 3000, "sentinel": "end"})
    original = copy.deepcopy(source)
    offset, chunks = 0, []
    while offset is not None:
        page = store.call("read_evidence", {"evidence_id": source["id"], "offset": offset},
                          objective="read_result", affected_items=["R01"])
        shown = model_view(store.records, max_bytes=0)[page["id"]]
        assert shown["view_truncated"]
        assert shown["content"]["preview"] == ""
        assert shown["source_evidence_id"] == source["id"]
        assert shown["source_kind"] == "tool_result"
        assert shown["offset"] == offset
        assert shown["total_chars"] == len(canonical(source["content"]))
        chunks.append(page["content"]["text"])
        offset = shown["next_offset"]
    assert json.loads("".join(chunks)) == source["content"]
    assert source == original


def test_query_errors_keep_provenance_and_validity_after_priority_allocation(store):
    failed = store.call("query_snapshot", {"snapshot": "initial", "sql": "SELECT absent FROM items"},
                        objective="check_initial", affected_items=["R01"])
    store.add("runtime", {"actual_start_utc": "2026-11-09T16:55:52Z"})
    add_query_noise(store)
    shown = model_view(store.records, max_bytes=40000)[failed["id"]]
    assert not shown["valid"]
    assert shown["error"] == failed["error"]
    assert shown["arguments"] == failed["arguments"]
    with pytest.raises(ValueError, match="cannot_promote_failed_evidence"):
        store.read_evidence(failed["id"])
