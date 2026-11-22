"""Offline CLI/checkpoint integration; real tiny weights, synthetic environment/Judge."""
import asyncio
import json
import shutil
import sys
from pathlib import Path

import pytest

from harness_awm.contracts import RunConfig
from harness_awm.io import canonical, digest, file_hash, read_json, write_json
from harness_awm.runner import EpisodeRunner
from harness_awm.training import evaluate as local_eval, grpo
from harness_awm.training.local import checkpoint_identity, load_checkpoint
from harness_awm.training.tasks import load_tasks, prepare_tasks, select_tasks
from test_harness_v2 import MemoryEnv, TOOLS
from test_judge_v2 import Responder, assessment, rubric_fixture
from test_training_grpo import setup_sources, tiny_policy
from harness_support import fixture


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("SWANLAB_MODE", "disabled")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")


def save_model(path):
    model, tokenizer = tiny_policy("qwen3.5")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return model


def memory_runner(config, budget, semaphore):
    return EpisodeRunner(config, budget, semaphore, env_factory=MemoryEnv)


def test_grpo_cli_preflight_and_explicit_training(tmp_path, monkeypatch, capsys):
    task, config, _, _ = setup_sources(tmp_path / "sources")
    write_json(tmp_path / "config.json", config.model_dump())
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(task.model_dump_json() + "\n")
    command = ["grpo", "--config", str(tmp_path / "config.json"), "--tasks", str(tasks), "--group-size", "2"]
    monkeypatch.setattr(sys, "argv", command)
    grpo.main()
    assert json.loads(capsys.readouterr().out)["training_started"] is False
    model_path = tmp_path / "model"
    save_model(model_path)
    original = checkpoint_identity(model_path)
    responder = Responder([assessment((s, "pass"), refs=["E00002"]) for s in ("pass", "fail")])
    monkeypatch.setattr("harness_awm.judge_runtime.api_request", responder)
    monkeypatch.setattr(grpo, "EpisodeRunner", memory_runner)
    output = tmp_path / "trained"
    monkeypatch.setattr(sys, "argv", command + ["--execute", "--model", str(model_path),
        "--output", str(output), "--groups", "1", "--device", "cpu", "--learning-rate", "0.001"])
    grpo.main()
    assert read_json(output / "summary.json")["groups"][0]["status"] == "updated"
    assert read_json(output / "launch.json")["checkpoint"] == original
    assert (output / "optimizer.pt").is_file()
    model, _ = load_checkpoint(output / "final", device="cpu")
    assert not any(p.requires_grad for p in model.parameters())
    assert checkpoint_identity(model_path) == original
    assert checkpoint_identity(output / "final")["sha256"] != original["sha256"]
    assert responder.calls == 2


def evaluation_fixture(tmp_path):
    task, config, _, _ = setup_sources(tmp_path / "sources")
    task = task.model_copy(update={"split": "dev", "protocol_id": config.protocol_id})
    config = RunConfig.model_validate({**config.model_dump(), "purpose": "evaluation", "reward_admissions": None})
    return task, config


@pytest.mark.parametrize("status,rate,calls", [("pass", 1, 1), ("fail", 0, 1), ("unknown", 0, 2)])
def test_local_checkpoint_evaluation_uses_judge_without_admission_or_update(tmp_path, monkeypatch, status, rate, calls):
    import torch
    task, config = evaluation_fixture(tmp_path)
    model_path = tmp_path / "checkpoint"
    save_model(model_path)
    model, tokenizer = load_checkpoint(model_path, device="cpu")
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    responder = Responder([assessment((status, "pass"), refs=["E00002"])] * calls)
    monkeypatch.setattr("harness_awm.judge_runtime.api_request", responder)
    report = asyncio.run(local_eval.evaluate(model, tokenizer, [task], config, tmp_path / "evaluation",
        split="validation", checkpoint=checkpoint_identity(model_path), runner_factory=memory_runner))
    assert report["success_over_planned"]["rate"] == rate
    assert report["training_updates"] is False
    assert all(torch.equal(p, before[n]) and p.grad is None for n, p in model.named_parameters())
    episode = read_json(tmp_path / "evaluation/episodes/t00000/a0/episode_result.json")
    assert episode["reward"] is None and not episode["reward_valid"]
    assert episode["reward_exclusion"] == "evaluation_only_no_training_reward"
    assert report["evaluation"]["eval_reward"]["mean"] == {"pass": 1, "fail": 0, "unknown": .3}[status]
    assert report["evaluation"]["steps"]["valid_scored"]["turns"]["count"] == 1
    assert report["wall_seconds"] > 0
    assert (tmp_path / "evaluation/per_task.csv").is_file()
    assert (tmp_path / "evaluation/report_zh.md").is_file()
    assert not list((tmp_path / "evaluation").rglob("policy_turn.json"))
    assert responder.calls == calls


def test_evaluation_fault_stays_in_planned_denominator(tmp_path, monkeypatch):
    task, config = evaluation_fixture(tmp_path)
    model, tokenizer = tiny_policy()
    async def unavailable(*args, **kwargs):
        raise RuntimeError("synthetic unavailable Judge")
    monkeypatch.setattr("harness_awm.judge_runtime.api_request", unavailable)
    report = asyncio.run(local_eval.evaluate(model, tokenizer, [task], config, tmp_path / "failed_eval",
        split="validation", checkpoint={"sha256": "synthetic"}, runner_factory=memory_runner))
    assert report["success_over_planned"] == {"numerator": 0, "denominator": 1, "rate": 0}
    assert report["success_over_valid"]["denominator"] == 0
    assert report["invalid_count"] == 1


def test_judge_budget_stop_preserves_unattempted_tasks(tmp_path, monkeypatch):
    from harness_awm.contracts import EpisodeResult
    task, config = evaluation_fixture(tmp_path)
    tasks = [task, task.model_copy(update={"task_instance_id": "second-task"})]
    model, tokenizer = tiny_policy()
    # This test isolates scheduler behavior; task source validation has separate coverage.
    monkeypatch.setattr(local_eval, "preflight", lambda *a: {})
    attempted = []
    class ExhaustedRunner:
        async def run(self, task, folder, rollout, actor):
            attempted.append(task.task_instance_id)
            write_json(folder / "episode_result.json", EpisodeResult(
                rollout_id=rollout, task_instance_id=task.task_instance_id,
                judge_protocol="rubric_v2", evaluation_status="evaluation_error",
                termination_reason="finish", test_only=False).model_dump())
            write_json(folder / "rewards_v2/result.json", {"errors": ["BudgetExceeded: global_budget_exhausted"]})
    report = asyncio.run(local_eval.evaluate(model, tokenizer, tasks, config, tmp_path / "exhausted",
        split="validation", checkpoint={"sha256": "synthetic"}, runner_factory=lambda *a: ExhaustedRunner()))
    assert attempted == [task.task_instance_id]
    assert report["attempted_tasks"] == 1 and report["planned_candidates"] == 2
    assert report["stop_reason"] == "judge_budget_exhausted"
    assert report["evaluation"]["missing_results"] == 1
    assert report["evaluation"]["eval_reward"]["count"] == 0


def test_evaluation_cli_preflight_never_loads_weights(tmp_path, monkeypatch, capsys):
    task, config = evaluation_fixture(tmp_path)
    save_model(tmp_path / "checkpoint")
    write_json(tmp_path / "config.json", config.model_dump())
    (tmp_path / "tasks.jsonl").write_text(task.model_dump_json() + "\n")
    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must not load or execute a model")
    monkeypatch.setattr(local_eval, "load_checkpoint", forbidden)
    monkeypatch.setattr(sys, "argv", ["evaluate", "--model", str(tmp_path / "checkpoint"),
        "--config", str(tmp_path / "config.json"), "--tasks", str(tmp_path / "tasks.jsonl"), "--split", "validation"])
    local_eval.main()
    assert json.loads(capsys.readouterr().out)["evaluation_started"] is False


def test_evaluation_rejects_train_split_admissions_and_wrong_sources(tmp_path):
    task, config = evaluation_fixture(tmp_path)
    with pytest.raises(ValueError, match="held-out"):
        local_eval.preflight([task.model_copy(update={"split": "train"})], config, "validation")
    with pytest.raises(ValueError, match="scoring-only"):
        RunConfig.model_validate({**config.model_dump(), "reward_admissions": "never.json"})
    with pytest.raises(ValueError, match="positive Judge"):
        local_eval.preflight([task], config.model_copy(update={"max_total_tokens": 0}), "validation")
    assert select_tasks([task], [f"{task.scenario}:{task.native_task_id}"]) == [task]
    with pytest.raises(ValueError, match="duplicate"):
        select_tasks([task], [task.task_instance_id, task.task_instance_id])


def frozen_split(root):
    asset = fixture(root / "assets")
    split = root / "split"
    split.mkdir()
    inventory_path = root / "runs/environment_screen_remaining5909_20261104_v1/inventory.jsonl"
    inventory = []
    for i, split_name in enumerate(("grpo_train", "validation", "test")):
        ident = f"harness_fixture:{i}"
        case = root / "screen" / str(i)
        case.mkdir(parents=True)
        original, rubric = rubric_fixture()
        original["task_id"] = rubric["task_id"] = ident
        write_json(case / "original.json", original)
        write_json(case / "rubric.json", rubric)
        write_json(case / "task.json", {"task_id": ident, "task": original["task"]})
        for name, source in (("initial.db", asset.initial_state_ref), ("environment.json", asset.environment_ref)):
            shutil.copyfile(source, case / name)
        hashes = {n: file_hash(case / n) for n in ("initial.db", "environment.json", "task.json")}
        write_json(case / "asset_hashes.json", hashes)
        inventory.append({"task_id": ident, "asset_hashes_sha256": file_hash(case / "asset_hashes.json")})
        write_json(case / "decision.json", {"decision": "pass"})
        write_json(case / "baseline/facts.json", {"mcp_ready": i != 0, "initial_unchanged": True,
            "database_sha256": asset.initial_state_hash, "environment_sha256": asset.environment_code_hash,
            "tool_catalog_hash": digest(TOOLS), "database_logical_fingerprint": asset.initial_state_fingerprint})
        if i:
            write_json(case / "baseline/tool_catalog.json", {"tools": TOOLS})
        row = {"task_id": ident, "scenario": "harness_fixture", "instruction": original["task"],
            "split": split_name, "environment_dataset_ref": str(root / "dataset"),
            "original_ref": str(case / "original.json"), "original_sha256": file_hash(case / "original.json"),
            "judge_only_rubric_ref": str(case / "rubric.json"), "judge_only_rubric_sha256": file_hash(case / "rubric.json"),
            "screen_result_ref": str(case / "decision.json"), "screen_result_sha256": file_hash(case / "decision.json")}
        (split / f"{split_name}.jsonl").write_text(canonical(row) + "\n")
    inventory_path.parent.mkdir(parents=True)
    inventory_path.write_text("".join(canonical(r) + "\n" for r in inventory))
    write_json(split / "source_hashes.json", {str(inventory_path): file_hash(inventory_path)})
    (split / "sft_trajectories.jsonl").write_text(canonical({"task_id": "sft:0", "split": "sft_train"}) + "\n")
    write_json(split / "manifest.json", {"schema_version": "awm-training-split-v1", "source_root": str(root),
        "files": {p.name: file_hash(p) for p in split.iterdir()}})
    return split


def test_runtime_bundle_preserves_splits_relocates_and_checks_hashes(tmp_path):
    split = frozen_split(tmp_path)
    output = tmp_path / "runtime"
    result = prepare_tasks(split, output, tmp_path)
    assert result["counts"] == {"grpo_train": 1, "validation": 1, "test": 1}
    assert result["worlds"] == 1 and not result["reward_admissions_created"]
    assert len(read_json(output / "manifest.json")["catalog_fallbacks"]) == 1
    moved = tmp_path / "moved"
    output.rename(moved)
    for split_name, expected in (("grpo_train", "train"), ("validation", "dev"), ("test", "test")):
        tasks, _, sources = load_tasks(moved, split_name)
        assert tasks[0].split == expected
        assert Path(tasks[0].initial_state_ref).is_relative_to(moved)
        assert Path(sources).is_relative_to(moved)
        assert not tasks[0].readiness_profile["reward_ready"]
    (moved / "test.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_tasks(moved, "test")


def test_split_manifest_is_not_mistaken_for_runtime_tasks(tmp_path):
    split = frozen_split(tmp_path)
    with pytest.raises(ValueError, match="Convert the frozen split"):
        load_tasks(split / "grpo_train.jsonl", "grpo_train")


def test_runtime_source_audit_finds_missing_indirect_assets(tmp_path):
    from harness_awm.training.tasks import audit_sources
    split = frozen_split(tmp_path)
    (tmp_path / "screen/1/initial.db").unlink()
    report = audit_sources(split, tmp_path)
    assert report["hash_mismatches"] == []
    assert "screen/1/initial.db" in report["missing"]
    assert report["splits"]["validation"]["tasks_with_missing_or_mismatched_files"] == 1
    assert report["runtime_preflight_passed"] is False
