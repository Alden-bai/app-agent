"""Baseline score denominators, fault handling and reproducible pilot protocol."""
import copy
import csv
import hashlib
import asyncio
from types import SimpleNamespace

import pytest

from harness_awm.evaluation_metrics import distribution, evaluation_score
from harness_awm.io import read_json, write_json
from harness_awm.stages import metrics
from harness_awm.training.baseline import pilot_gate, projected_budget, select_pilot, strict_probe
from harness_awm.training.evaluate import judge_budget_exhausted


def result(label="pass", **changes):
    return {"judge_protocol": "rubric_v2", "lifecycle": "SETTLED", "evaluation_valid": True,
            "evaluation_status": "valid", "business_label": label,
            "task_status": {"pass": "confirmed_pass", "unknown": "provisional_pass", "fail": "failed"}[label],
            "execution_integrity": "valid", "terminal_state_status": "trusted", "test_only": False,
            "termination_reason": "finish", "reward": None, "reward_valid": False,
            "turns": 4, "tool_calls": 2, **changes}


@pytest.mark.parametrize("changes", [
    {"lifecycle": "EVAL_PENDING"}, {"evaluation_status": "evaluation_error"},
    {"evaluation_status": "invalid_sample"}, {"evaluation_status": "rubric_error"},
    {"evaluation_valid": False}, {"execution_integrity": "outcome_unknown"},
    {"terminal_state_status": "untrusted"}, {"test_only": True},
    {"test_only": None}, {"business_label": "unknown"}, {"judge_protocol": "official"},
])
def test_faults_and_inconsistent_labels_never_receive_eval_reward(changes):
    assert evaluation_score(result(**changes)) is None


@pytest.mark.parametrize("label,expected", [("pass", 1), ("unknown", .3), ("fail", 0)])
def test_score_is_independent_of_training_admission_and_never_mutates_result(label, expected):
    record = result(label, reward_exclusion="evaluation_only_no_training_reward")
    before = copy.deepcopy(record)
    assert evaluation_score(record) == expected
    assert record == before


def test_report_denominators_steps_missing_zero_and_failure_at_limit(tmp_path):
    records = [result(), result("unknown", turns=7, tool_calls=3),
               result("fail", turns=32, tool_calls=10, termination_reason="max_turns"),
               result(evaluation_valid=False, evaluation_status="evaluation_error", lifecycle="EVAL_PENDING", turns=2, tool_calls=0),
               result(execution_integrity="outcome_unknown", turns=5, tool_calls=1), None,
               result(turns=1, tool_calls=0)]
    candidates, tasks = [], []
    for i, record in enumerate(records):
        candidates.append({"rollout_id": f"t{i}", "task_instance_id": f"task{i}", "scenario": "fixture", "path": f"episodes/{i}"})
        tasks.append({"task_instance_id": f"task{i}", "scenario": "fixture", "native_task_id": i})
        if record is not None:
            write_json(tmp_path / f"episodes/{i}/episode_result.json", record)
            write_json(tmp_path / f"episodes/{i}/trajectory.json", {"final_answer": "done"})
    write_json(tmp_path / "manifest.json", {"candidates": candidates, "tasks": tasks, "config": {"purpose": "evaluation"}})
    report = metrics(tmp_path)
    stats = report["evaluation"]
    assert stats["planned"] == 7 and stats["result_records"] == 6 and stats["missing_results"] == 1
    assert stats["scored"] == 4 and stats["confirmed_successes"] == 2
    assert stats["score_coverage"] == 4 / 7
    assert stats["success_over_planned"] == 2 / 7 and stats["success_over_scored"] == .5
    assert stats["eval_reward"] == {"count": 4, "sum": 2.3, "mean": .575, "null_count": 3}
    assert stats["steps"]["all_recorded"]["turns"] == {"count": 6, "mean": 8.5, "median": 4.5, "p90": 32}
    assert stats["steps"]["confirmed_success"]["turns"]["mean"] == 2.5
    assert stats["steps"]["confirmed_success"]["tool_calls"]["mean"] == 1
    assert stats["steps"]["valid_scored"]["turns"]["count"] == 4
    with (tmp_path / "per_task.csv").open(encoding="utf-8-sig") as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 7
    assert exported[5]["eval_reward"] == exported[5]["turns"] == ""
    assert exported[6]["tool_calls"] == "0"
    assert exported[2]["eval_reward"] == "0"
    assert (tmp_path / "report_zh.md").is_file()
    assert read_json(tmp_path / "metrics.json")["evaluation"] == stats
    for i, record in enumerate(records):
        if record is not None:
            assert read_json(tmp_path / f"episodes/{i}/episode_result.json") == record


def test_empty_distribution_is_null_not_zero():
    assert distribution([None, float("nan"), -1]) == {"count": 0, "mean": None, "median": None, "p90": None}
    assert distribution([0])["mean"] == 0


def task(scenario, index):
    return SimpleNamespace(scenario=scenario, native_task_id=index, task_instance_id=f"id:{scenario}:{index}")


def test_pilot_selection_is_order_independent_and_prefers_different_scenarios():
    tasks = [task(f"scenario{i % 25}", i) for i in range(60)]
    selected = select_pilot(tasks)
    assert selected == select_pilot(list(reversed(tasks)))
    assert len(selected) == len({t.scenario for t in selected}) == 20
    order = [hashlib.sha256(f"20261106:{t.scenario}:{t.native_task_id}".encode()).hexdigest() for t in selected]
    assert order == sorted(order)
    all_tasks = [task("one", i) for i in range(20)]
    assert len(select_pilot(all_tasks)) == 20
    with pytest.raises(ValueError, match="insufficient"):
        select_pilot(all_tasks[:19])


def pilot_report():
    return {"planned_candidates": 20, "settled_candidates": 20, "evaluation": {"scored": 20},
            "stop_reason": "all_tasks_attempted", "wall_seconds": 100,
            "budget": {"physical_requests": 41, "tokens_accounted_or_reserved": 101}}


def test_budget_projection_rounds_up_and_includes_reserved_tokens():
    forecast = projected_budget(pilot_report())
    assert forecast["max_physical_requests"] == 1812
    assert forecast["max_total_tokens"] == 4462
    assert forecast["estimated_wall_seconds"] == 2945
    # No requirement for a positive success rate: valid failures are useful data.
    pilot_gate(pilot_report())


@pytest.mark.parametrize("change", [{"settled_candidates": 19}, {"evaluation": {"scored": 19}},
                                    {"stop_reason": "judge_budget_exhausted"}])
def test_faulty_pilot_never_launches_full_validation(change):
    with pytest.raises(ValueError, match="pilot incomplete"):
        projected_budget({**pilot_report(), **change})


def test_budget_reservation_rejection_is_detected_before_numeric_cap(tmp_path):
    assert not judge_budget_exhausted(tmp_path)
    write_json(tmp_path / "rewards_v2/a.json", {"errors": ["BudgetExceeded: global_budget_exhausted"]})
    assert judge_budget_exhausted(tmp_path)


def test_strict_namespace_failure_is_saved_without_fallback(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr("harness_awm.training.baseline.command", lambda *a: ["bwrap", "--unshare-user"])
    def denied(argv, **kwargs):
        seen.append(argv)
        return SimpleNamespace(returncode=1, stdout="", stderr="Operation not permitted")
    monkeypatch.setattr("harness_awm.training.baseline.subprocess.run", denied)
    config = SimpleNamespace(isolation="strict", runtime_python="python", timezone="Asia/Shanghai")
    assert not strict_probe(config, tmp_path)["passed"]
    assert read_json(tmp_path / "result.json")["stderr"] == "Operation not permitted"
    assert seen == [["bwrap", "--unshare-user"]]


def prepared_plan(tmp_path, monkeypatch):
    from harness_awm.contracts import RunConfig
    from harness_awm.io import file_hash
    from harness_awm.training import baseline
    config = RunConfig(purpose="evaluation", actor="training", judge="api", judge_protocol="rubric_v2",
                       protocol_id="awm-single-json-v2", judge_inputs_manifest="sources", max_physical_requests=1000,
                       max_total_tokens=10000000)
    tasks = [task(f"scenario{i}", i) for i in range(589)]
    selected = select_pilot(tasks)
    plan = {"config": config.model_dump(), "source_hashes": {}, "task_manifest": {"sha256": "fixture"},
            "tasks_path": "synthetic", "checkpoint": {"sha256": "weights", "path": "model"},
            "pilot_task_ids": [baseline.task_id(t) for t in selected],
            "device": "cuda", "dtype": "bfloat16", "sampling": "greedy"}
    write_json(tmp_path / "plan.json", plan)
    write_json(tmp_path / "plan.sha256.json", {"sha256": file_hash(tmp_path / "plan.json")})
    monkeypatch.setattr(baseline, "source_hashes", lambda: {})
    monkeypatch.setattr(baseline, "load_tasks", lambda *a: (tasks, plan["task_manifest"], "sources"))
    monkeypatch.setattr(baseline, "checkpoint_identity", lambda *a: plan["checkpoint"])
    monkeypatch.setattr(baseline, "preflight", lambda *a: {})
    monkeypatch.setenv(config.judge_api.api_key_env, "test-only-not-a-real-key")
    return baseline


def test_blocked_execute_never_loads_model_or_calls_judge(tmp_path, monkeypatch):
    baseline = prepared_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(baseline, "strict_probe", lambda *a: {"passed": False})
    def forbidden(*args, **kwargs):
        raise AssertionError("blocked runs must not load or evaluate")
    monkeypatch.setattr(baseline, "load_checkpoint", forbidden)
    monkeypatch.setattr(baseline, "evaluate", forbidden)
    with pytest.raises(RuntimeError, match="strict isolation unavailable"):
        asyncio.run(baseline.execute(tmp_path))
    assert read_json(tmp_path / "status.json")["evaluation_started"] is False


@pytest.mark.parametrize("fault", [False, True])
def test_orchestration_uses_separate_full_run_only_after_valid_pilot(tmp_path, monkeypatch, fault):
    baseline = prepared_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(baseline, "strict_probe", lambda *a: {"passed": True})
    monkeypatch.setattr(baseline, "load_checkpoint", lambda *a, **kw: (object(), object()))
    calls = []
    async def simulated(model, tokenizer, tasks, config, output, **kwargs):
        calls.append((len(tasks), config.max_physical_requests, config.max_total_tokens, output.name, kwargs["sampling"]))
        if output.name == "pilot":
            report = pilot_report()
            if fault:
                report["evaluation"]["scored"] = 19
            return report
        return {"settled_candidates": len(tasks), "evaluation": {"scored": len(tasks)}}
    monkeypatch.setattr(baseline, "evaluate", simulated)
    if fault:
        with pytest.raises(ValueError, match="pilot incomplete"):
            asyncio.run(baseline.execute(tmp_path))
        assert len(calls) == 1
        assert read_json(tmp_path / "status.json")["stage"] == "pilot_needs_review"
    else:
        status = asyncio.run(baseline.execute(tmp_path))
        assert status["stage"] == "validation_finished"
        assert calls == [(20, 1000, 10000000, "pilot", "greedy"), (589, 1812, 4462, "validation", "greedy")]


def test_unlimited_judge_accounting_and_config_are_explicit(tmp_path):
    from harness_awm.backend import Budget
    from harness_awm.contracts import RunConfig
    config = RunConfig(purpose="evaluation", actor="training", judge="api", judge_protocol="rubric_v2",
                       max_physical_requests=None, max_total_tokens=None)
    budget = Budget(tmp_path / "budget.sqlite", config.max_physical_requests, config.max_total_tokens)
    budget.hold("final", 10**9)
    for i in range(3):
        rid = budget.reserve(str(i), "judge", [{"role": "user", "content": "test"}], 8192)
        budget.finish(rid, {"total_tokens": 100})
    budget.release("final")
    assert budget.summary() == {"physical_requests": 3, "tokens_accounted_or_reserved": 300,
                                "unknown_usage_requests": 0, "held_requests": 0, "held_tokens": 0}
    with pytest.raises(ValueError, match="unlimited budgets require"):
        RunConfig(max_physical_requests=None, max_total_tokens=None)
    with pytest.raises(ValueError, match="both limits=null"):
        RunConfig(**{**config.model_dump(), "max_total_tokens": 100})


def test_direct_validation_and_test_use_unlimited_budget_without_pilot(tmp_path, monkeypatch):
    from harness_awm.io import file_hash
    baseline = prepared_plan(tmp_path, monkeypatch)
    plan = read_json(tmp_path / "plan.json")
    plan.update(skip_pilot=True, unlimited_judge_budget=True, splits=["validation", "test"], pilot_task_ids=[])
    plan["config"].update(max_physical_requests=None, max_total_tokens=None)
    write_json(tmp_path / "plan.json", plan)
    write_json(tmp_path / "plan.sha256.json", {"sha256": file_hash(tmp_path / "plan.json")})
    def load(path, split):
        return ([task(f"{split}{i}", i) for i in range(589 if split == "validation" else 588)], plan["task_manifest"], "sources")
    monkeypatch.setattr(baseline, "load_tasks", load)
    monkeypatch.setattr(baseline, "strict_probe", lambda *a: {"passed": True})
    monkeypatch.setattr(baseline, "load_checkpoint", lambda *a, **kw: (object(), object()))
    calls = []
    async def simulated(model, tokenizer, tasks, config, output, **kwargs):
        calls.append((kwargs["split"], len(tasks), config.max_physical_requests, config.max_total_tokens))
        return {"settled_candidates": len(tasks), "evaluation": {"scored": len(tasks)}}
    monkeypatch.setattr(baseline, "evaluate", simulated)
    status = asyncio.run(baseline.execute(tmp_path))
    assert calls == [("validation", 589, None, None), ("test", 588, None, None)]
    assert status["stage"] == "evaluation_finished"
    assert len(status["splits"]) == 2
