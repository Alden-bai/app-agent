import sqlite3
from types import SimpleNamespace

import pytest

from harness_awm.io import write_json, seal
from scripts.parallel_baseline import BusyBoundary, budget_paths, finished_episodes, partition


def prior(tmp_path, status="valid", label="failed"):
    run = tmp_path / "validation"
    folder = run / "episodes/t00000/a0"
    write_json(run / "manifest.json", {"candidates": [{"task_instance_id": "task0", "path": "episodes/t00000/a0"}]})
    write_json(folder / "trajectory.json", {"answer": "real prior model output"})
    write_json(folder / "episode_result.json", {"task_instance_id": "task0", "evaluation_status": status,
                                              "task_status": label})
    seal(folder)
    return run, folder


def test_reuse_does_not_select_by_success_or_judge_validity(tmp_path):
    run, folder = prior(tmp_path, "evaluation_error")
    kept = finished_episodes(tmp_path, verify=True)
    assert list(kept) == ["task0"]
    tasks = [SimpleNamespace(task_instance_id=f"task{i}") for i in range(1177)]
    shards = partition(tasks, kept, 8)
    flattened = [ident for shard in shards for ident in shard]
    assert len(flattened) == len(set(flattened)) == 1176
    assert set(flattened) | set(kept) == {t.task_instance_id for t in tasks}
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


@pytest.mark.parametrize("status", ["not_run", "pending"])
def test_no_handoff_before_judge_finished(tmp_path, status):
    prior(tmp_path, status)
    with pytest.raises(BusyBoundary):
        finished_episodes(tmp_path)


def test_in_flight_request_blocks_handoff_even_with_result(tmp_path):
    run, folder = prior(tmp_path)
    with sqlite3.connect(run / "budget.sqlite") as db:
        db.execute("CREATE TABLE requests(status TEXT)")
        db.execute("CREATE TABLE holds(id INTEGER)")
        db.execute("INSERT INTO requests VALUES ('in_flight')")
    with pytest.raises(BusyBoundary, match="in flight"):
        finished_episodes(tmp_path)


def test_next_task_started_during_boundary_race_blocks_handoff(tmp_path):
    run, folder = prior(tmp_path)
    write_json(run / "manifest.json", {"candidates": [
        {"task_instance_id": "task0", "path": "episodes/t00000/a0"},
        {"task_instance_id": "task1", "path": "episodes/t00001/a0"}]})
    (run / "episodes/t00001/a0").mkdir(parents=True)
    with pytest.raises(BusyBoundary, match="still running"):
        finished_episodes(tmp_path)


def test_sealed_evidence_must_match(tmp_path):
    run, folder = prior(tmp_path)
    write_json(folder / "trajectory.json", {"changed": True})
    with pytest.raises(ValueError, match="evidence_hash_mismatch"):
        finished_episodes(tmp_path, verify=True)


def test_environment_startup_may_be_deferred_only_before_any_model_request(tmp_path):
    run, folder = prior(tmp_path)
    next_folder = run / "episodes/t00001/a0"
    next_folder.mkdir(parents=True)
    write_json(run / "manifest.json", {"candidates": [
        {"task_instance_id": "task0", "path": "episodes/t00000/a0"},
        {"task_instance_id": "task1", "path": "episodes/t00001/a0"}]})
    deferred = {}
    kept = finished_episodes(tmp_path, allow_unstarted=True, deferred=deferred)
    assert list(kept) == ["task0"] and list(deferred) == ["task1"]
    (next_folder / "events.jsonl").write_text('{"event_type":"model_request"}\n')
    with pytest.raises(BusyBoundary):
        finished_episodes(tmp_path, allow_unstarted=True)


def test_model_input_written_before_event_still_blocks_deferral(tmp_path):
    run, folder = prior(tmp_path)
    (folder / "episode_result.json").unlink()
    write_json(folder / "model_turns/0001/input.json", {"messages": []})
    with pytest.raises(BusyBoundary):
        finished_episodes(tmp_path, allow_unstarted=True)


def test_multiple_continuations_keep_all_budget_ledgers_once(tmp_path):
    first, second, third = [tmp_path / name for name in ("serial", "parallel8", "parallel19")]
    a = first / "validation/budget.sqlite"
    b = second / "workers/w00/validation/budget.sqlite"
    c = third / "workers/w00/validation/budget.sqlite"
    for path in (a, b, c):
        path.parent.mkdir(parents=True)
        path.touch()
    write_json(first / "plan.json", {})
    write_json(second / "plan.json", {"source_run": str(first)})
    write_json(third / "plan.json", {"source_run": str(second)})
    assert set(budget_paths(third, "validation")) == {a, b, c}
    write_json(first / "plan.json", {"source_run": str(third)})
    with pytest.raises(ValueError, match="cyclic"):
        budget_paths(third, "validation")


def test_second_scale_partition_reuses_every_old_task_exactly_once():
    tasks = [SimpleNamespace(task_instance_id=f"task{i}") for i in range(589)]
    kept = {f"task{i}": {} for i in range(73)}
    assignment = partition(tasks, kept, 19)
    assigned = [ident for shard in assignment for ident in shard]
    assert len(assigned) == len(set(assigned)) == 516
    assert not set(assigned).intersection(kept)
    assert set(assigned).union(kept) == {t.task_instance_id for t in tasks}
