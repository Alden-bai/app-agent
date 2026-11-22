"""Candidate reward invariants and read-only replay of saved decisions."""
import copy
import importlib.util
import itertools
import json
from pathlib import Path

import pytest

from harness_awm.composite_reward import reward_from_components
from harness_awm.reward_values import COMPOSITE_VERSION
from harness_awm.training.rewards import population_advantages
spec = importlib.util.spec_from_file_location("cst_preview", Path(__file__).parents[1] / "scripts/preview_cst_reward_v2.py")
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)
floored_advantages, preview_reward, replay = preview.floored_advantages, preview.preview_reward, preview.replay


def source(c=1, soft="pass", grades=(0, 0, 0)):
    label, status = {1: ("pass", "confirmed_pass"), .3: ("unknown", "provisional_pass"),
                     0: ("fail", "failed")}[c]
    s = {"pass": 1, "fail": 0, "unknown": .3}[soft]
    parts = {"policy": COMPOSITE_VERSION, "C": c, "S": s if c else None,
             "T": 1 - sum(grades) / 3 if c else None, "alpha": .3, "lambda_s": .05, "lambda_t": .15,
             "soft_items": [{"id": "s1", "status": soft, "value": s}] if c else [],
             "defects": dict(zip(("information", "feedback", "exploration"), grades)) if c else {}}
    return {"reward_components": parts, "business_label": label, "task_status": status,
            "reward_policy": COMPOSITE_VERSION, "reward_valid": True, "test_only": False,
            "evaluation_valid": True, "evaluation_status": "valid", "execution_integrity": "valid",
            "terminal_state_status": "trusted", "task_instance_id": "task-1",
            "reward": reward_from_components(parts, label, status)}


def test_every_grade_combination_preserves_completion_priority():
    tiers = {c: [preview_reward(source(c, soft, grades)) for soft in ("pass", "fail", "unknown")
                for grades in itertools.product((0, .5, 1), repeat=3)] for c in (0, .3, 1)}
    assert set(tiers[0]) == {0}
    assert min(tiers[1]) == pytest.approx(.6)
    assert min(tiers[.3]) == pytest.approx(.18)
    assert min(tiers[1]) > max(tiers[.3]) > max(tiers[0])


@pytest.mark.parametrize("grades,expected", [((0, 0, 0), 1), ((.5, 0, 0), .95), ((1, 0, 0), .9)])
def test_expected_repricing(grades, expected):
    assert preview_reward(source(grades=grades)) == expected


@pytest.mark.parametrize("change", [{"reward_valid": False}, {"reward": None}, {"reward": .5}])
def test_unsettled_or_changed_reward_is_rejected(change):
    with pytest.raises(ValueError):
        preview_reward({**source(), **change})


def test_no_soft_items_remains_neutral():
    r = source()
    r["reward_components"]["soft_items"] = []
    assert preview_reward(r) == 1
    r["reward_components"]["T"] = .9
    with pytest.raises(ValueError):
        preview_reward(r)


def test_floor_limits_small_difference_and_preserves_large_difference():
    r = [1] * 7 + [.95]
    assert floored_advantages(r)[-1] == pytest.approx(-.4375, abs=1e-5)
    assert population_advantages(r)[-1] < -2.64
    assert floored_advantages([1, 0, 0, 1]) == population_advantages([1, 0, 0, 1])
    assert floored_advantages([1] * 8) == [0] * 8
    assert floored_advantages([0] * 8) == [0] * 8
    assert sum(floored_advantages(r)) == pytest.approx(0)


@pytest.mark.parametrize("floor", [0, -1, float("nan"), float("inf"), True])
def test_invalid_floor(floor):
    with pytest.raises(ValueError):
        floored_advantages([0, 1], floor)


def write_group(root, number, results, status="updated"):
    folder = root / "groups" / f"group-{number:06d}"
    folder.mkdir(parents=True)
    (folder / "update.json").write_text(json.dumps({"group": number, "optimizer_step": number,
        "status": status, "rewards": [r["reward"] for r in results]}))
    for i, r in enumerate(results):
        p = folder / f"candidate-{i:03d}"
        p.mkdir()
        (p / "episode_result.json").write_text(json.dumps(r))


def test_replay_compares_ablations_without_modifying_sources(tmp_path):
    write_group(tmp_path, 1, [source(), source(grades=(.5, 0, 0))])
    invalid = {**source(), "reward": None, "reward_valid": False, "reward_exclusion": "api_context_limit_exceeded"}
    write_group(tmp_path, 2, [source(), invalid], status="held")
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*.json")}
    report = replay(tmp_path)
    assert report["summary"]["updated_groups"] == 1
    assert report["summary"]["excluded_groups"] == 1
    assert report["summary"]["all_confirmed_success_groups"] == 1
    assert report["groups"][0]["v2_with_floor"]["rewards"] == [1, .95]
    assert report["excluded_groups"][0]["candidate_exclusions"][0]["reason"] == "api_context_limit_exceeded"
    assert {str(p): p.read_bytes() for p in tmp_path.rglob("*.json")} == before
    assert report["training_ready"] is False


def test_replay_rejects_missing_candidate_or_changed_update(tmp_path):
    results = [source(), source(grades=(.5, 0, 0))]
    write_group(tmp_path, 1, results)
    p = tmp_path / "groups/group-000001/candidate-000/episode_result.json"
    modified = copy.deepcopy(results[0])
    modified.update(source(grades=(1, 0, 0)))
    p.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="source_update_reward_mismatch"):
        replay(tmp_path)
    p.unlink()
    with pytest.raises(ValueError, match="incomplete candidate group"):
        replay(tmp_path)
