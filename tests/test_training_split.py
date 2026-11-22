import importlib.util
from pathlib import Path
from collections import Counter, defaultdict

import pytest

spec = importlib.util.spec_from_file_location("training_split", Path(__file__).resolve().parents[1] / "scripts/prepare_training_split.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_global_counts_and_development_holdout_exclusion():
    rows = [{"task_id": f"world_{i}:{j}", "scenario": f"world_{i}"}
            for i in range(980) for j in range(4)]
    rows += [{"task_id": f"world_0:{j}", "scenario": "world_0"} for j in range(4, 7)]
    forced = {"world_0:0", "world_0:1", "world_0:2", "world_1:3"}
    result = m.assign(rows, 20261106, forced)
    assert Counter(result.values()) == {"grpo_train": 2746, "validation": 589, "test": 588}
    assert set(result) == {r["task_id"] for r in rows}
    assert all(result[t] == "grpo_train" for t in forced)
    grouped = defaultdict(set)
    for r in rows:
        grouped[r["scenario"]].add(result[r["task_id"]])
    assert all("grpo_train" in splits for splits in grouped.values())
    assert m.assign(list(reversed(rows)), 20261106, forced) == result
    assert m.assign(rows, 20261107, forced) != result


def test_mask_keeps_turn_position_and_never_supervises_excluded_response():
    records = m.supervision([1, 2, 3, 4], [2, 4])
    assert [r["turn_id"] for r in records] == [1, 2, 3, 4]
    assert [r["response_loss_weight"] for r in records] == [1, 0, 1, 0]
    assert [r["supervise_current_response"] for r in records] == [True, False, True, False]
    with pytest.raises(ValueError):
        m.supervision([1, 2], [3])
    with pytest.raises(ValueError):
        m.supervision([1, 2], [2, 2])


def test_duplicate_tasks_and_impossible_constraints_fail():
    with pytest.raises(ValueError):
        m.assign([{"task_id": "a:1", "scenario": "a"}] * 2, 1, set())
    with pytest.raises(ValueError):
        m.assign([{"task_id": f"a:{i}", "scenario": "a"} for i in range(10)], 1,
                 {f"a:{i}" for i in range(10)})
