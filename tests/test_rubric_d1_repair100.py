"""Offline stage/replay tests; no model calls."""
import asyncio
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rubric_d1_repair100 as pilot
from test_rubric_d1 import fixture


def review(task_id, score=2):
    return dict(task_id=task_id, dimensions=[dict(name=d, score=score, reason="tested boundary", findings=[dict(
        item_ids=["R01"], candidate_ids=["C01"], problem="tested", counterexample="a concrete boundary")])
        for d in pilot.d1.base.DIMENSIONS])


def task_input(tmp_path):
    inp, _, selection, rubric, _ = fixture()
    rubric["unresolved"] = [dict(issue="old gap", affected_items=["R01"], needed="definition")]
    paths = {}
    for name, value in [("original", inp), ("old_rubric", rubric)]:
        p = tmp_path / (name + ".json")
        p.write_text(json.dumps(value))
        paths[name + "_ref"] = str(p)
    return dict(task_id=inp["task_id"], old_unresolved=1, **paths), selection


@pytest.mark.parametrize("mode,expected,calls", [("pass", "passed", 2), ("reject_then_pass", "passed", 4),
    ("reject_twice", "rejected", 4), ("bad_generation", "invalid_generation", 2), ("bad_review", "invalid_judge", 2),
    ("review_call_failure", "call_failed", 2)])
def test_bounded_stage_workflow(tmp_path, monkeypatch, mode, expected, calls):
    monkeypatch.setattr(pilot, "RUN", tmp_path)
    t, selection = task_input(tmp_path)
    runner = pilot.Runner()
    seen = []

    async def fake_call(task_id, stage, role, data):
        seen.append(stage)
        if role == "repair":
            assert "old_rubric" in data
            if stage == "repair_1":
                assert "revision_feedback" in data
            return "{" if mode == "bad_generation" else json.dumps(selection)
        assert not {"old_rubric", "previous_output", "revision_feedback", "selection"} & set(data)
        assert "items" not in data  # no duplicate full generated business text
        assert all("reason" not in s for s in data["candidate_mapping"])
        if mode == "review_call_failure":
            raise pilot.CallFailed("simulated")
        if mode == "bad_review":
            return "{}"
        score = 0 if mode == "reject_twice" or mode == "reject_then_pass" and stage == "review_0" else 2
        return json.dumps(review(task_id, score))

    runner.call = fake_call
    asyncio.run(runner.task(t))
    result = pilot.read(tmp_path / "tasks/d1__0/result.json")
    assert result["status"] == expected and len(seen) == calls
    asyncio.run(runner.task(t))
    assert len(seen) == calls  # completed task never calls again
    monkeypatch.setattr(pilot, "SIZE", 1)
    pilot.freeze(tmp_path / "manifest.json", dict(tasks=[t]))
    summary = pilot.report()
    assert summary["completed"] == 1
    assert summary["conservative_full100"]["after_unresolved_entries"] == (0 if expected == "passed" else 1)
    assert summary["actual_runtime_unknown_measured"] is False


def test_review_preserves_required_material_without_history():
    inp, _, selection, rubric, _ = fixture()
    result = pilot.review_payload(inp, selection, rubric)
    assert result["input"]["task"] == inp["task"]
    assert result["rubric"] == rubric
    assert len(result["candidate_mapping"]) == len(selection["selection"])
    assert set(result) == {"input", "candidate_mapping", "rubric"}


def test_sent_unresolved_request_cannot_be_resent(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "RUN", tmp_path)
    pilot.freeze(tmp_path / "prompts/repair.txt", "test protocol", raw=True)
    pilot.freeze(tmp_path / "tasks/t__0/repair_0/started.json", {})
    runner = pilot.Runner()
    with pytest.raises(pilot.Paused):
        asyncio.run(runner.call("t:0", "repair_0", "repair", {}))
    assert runner.stopped
    assert not (tmp_path / "tasks/t__0/repair_0/events.jsonl").exists()


def test_twelve_worker_limit_and_unique_dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "RUN", tmp_path)
    monkeypatch.setattr(pilot, "WORKERS", 12)
    monkeypatch.setattr(pilot, "report", lambda: {})
    runner = pilot.Runner()
    active, peak, seen = 0, 0, []

    async def task(t):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        seen.append(t["task_id"])
        await asyncio.sleep(0.002)
        active -= 1

    runner.task = task
    asyncio.run(runner.run(dict(tasks=[dict(task_id=str(i)) for i in range(37)])))
    assert peak == 12 and active == 0
    assert len(seen) == len(set(seen)) == 37
