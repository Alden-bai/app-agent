"""Portable delivery contracts: no weights, paid services or GPU needed."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("usable,expected", [
    ({"landlock", "strict"}, ["landlock"]),
    ({"strict"}, ["landlock", "strict"]),
    (set(), ["landlock", "strict"]),
])
def test_init_selects_tested_backend(tmp_path, monkeypatch, usable, expected):
    import sys
    module = script("a100_delivery")
    seen = []
    def probe(cfg, folder):
        seen.append(cfg.isolation)
        return {"passed": cfg.isolation in usable, "isolation": cfg.isolation}
    monkeypatch.setattr("harness_awm.training.baseline.isolation_probe", probe)
    work = tmp_path / "work"
    monkeypatch.setattr(sys, "argv", ["delivery", "init", "--work", str(work)])
    if usable:
        module.main()
        for name in ("config.json", "eval-config.json"):
            assert json.loads((work / name).read_text())["isolation"] == expected[-1]
    else:
        with pytest.raises(SystemExit, match="Both isolation probes failed"):
            module.main()
        assert not work.exists()
    assert seen == expected


def test_sft_only_init_blocks_online_commands(tmp_path, monkeypatch):
    import sys
    module = script("a100_delivery")
    work = tmp_path / "sft"
    monkeypatch.setattr(module, "choose_isolation", lambda *args: pytest.fail("offline SFT probed isolation"))
    monkeypatch.setattr(sys, "argv", ["delivery", "init", "--work", str(work), "--sft-only"])
    module.main()
    assert (work / "sft-only").is_file()
    monkeypatch.setattr(sys, "argv", ["delivery", "calibrate", "--work", str(work)])
    with pytest.raises(SystemExit):
        module.main()


def test_eight_gpu_four_candidate_config_and_upstream_validation():
    pytest.importorskip("verl")
    from harness_awm.training.verl.config import compose_config, validate_turnwise_config
    from harness_awm.training.verl.__main__ import resolve_prepare_defaults
    args = resolve_prepare_defaults(SimpleNamespace(preset="a100x8-27b"))
    assert (args.gpus, args.group_size, args.rollout_tensor_parallel_size) == (8, 4, 4)
    plan = {**vars(args), "run_config": {"seed": 1, "context_limit": 32768, "max_output_tokens": 2048},
            "model": "/tmp/model", "output": "/tmp/run", "gpu_memory_utilization": .35,
            "runtime_profile": "pinned"}
    config = compose_config(plan)
    validate_turnwise_config(config)
    assert config.data.train_batch_size == 1
    assert config.actor_rollout_ref.rollout.n == 4
    assert config.actor_rollout_ref.rollout.tensor_model_parallel_size == 4
    assert config.actor_rollout_ref.actor.fsdp_config.optimizer_offload is True


def test_sft_eight_ranks_use_native_microbatches_and_offload(tmp_path):
    pytest.importorskip("verl")
    from test_training_verl_sft import data_fixture
    from harness_awm.training.verl.sft import prepare, load_plan, compose_sft_config
    _, _, model, data, _ = data_fixture(tmp_path)
    args = SimpleNamespace(model=model, data=data, output=tmp_path / "run", gpus=8, global_batch_size=64,
        epochs=2, max_length=32, learning_rate=1e-5, warmup_ratio=.03, seed=1, max_steps=-1,
        optimizer_offload=True, param_offload=False)
    prepare(args)
    config = compose_sft_config(load_plan(args.output))
    assert config.engine.optimizer_offload is True
    assert config.trainer.n_gpus_per_node == 8
    assert not config.data.use_dynamic_bsz
    assert config.data.micro_batch_size_per_gpu == 1


def test_weight_and_secret_files_excluded_but_environment_dbs_allowed():
    allowed = script("build_training_delivery").allowed
    for name in ["models/model.safetensors", "checkpoint/optimizer.pt", ".env", ".venv/lib/a.py", "x/.git/config", "x/id_rsa"]:
        assert not allowed(name)
    for name in ["data/world/initial.db", "harness_awm/io.py", "assets/qwen35-27b/config.json"]:
        assert allowed(name)


def test_extracted_manifest_detects_tampering_and_escape(tmp_path):
    from hashlib import sha256
    verify = script("verify_training_delivery").verify
    (tmp_path / "data.txt").write_bytes(b"frozen")
    manifest = {"files": [{"path": "data.txt", "bytes": 6, "sha256": sha256(b"frozen").hexdigest()}],
                "total_bytes": 6, "weights_included": False, "gpu_validation": "pending"}
    (tmp_path / "DELIVERY_MANIFEST.json").write_text(json.dumps(manifest))
    assert verify(tmp_path)["passed"]
    (tmp_path / "data.txt").write_bytes(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        verify(tmp_path)
    manifest["files"][0]["path"] = "../outside"
    (tmp_path / "DELIVERY_MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        verify(tmp_path)


def test_probe_pads_single_diagnostic_trajectory_to_eight_ranks(tmp_path, monkeypatch):
    pytest.importorskip("verl")
    from harness_awm.training.verl.runtime import AWMTrainer
    from harness_awm.contracts import RunConfig
    from harness_awm.io import write_json, read_json
    class Actor:
        def __init__(self, client, *args, **kwargs):
            self.client = client
            self.records = []
        async def generate(self, *args, **kwargs):
            self.records = [{"prompt_ids": [2, 3], "completion_ids": [9], "old_logprobs": [-2.]}]
    class Client:
        async def generate(self, **kwargs):
            return SimpleNamespace(token_ids=[9], log_probs=[-2.])
    monkeypatch.setattr("harness_awm.training.verl.rollout.VerlActorBackend", Actor)
    plan = {"output": str(tmp_path), "run_config": RunConfig().model_dump(),
            "gpus": 8, "stress_probe": True, "eos_ids": [9]}
    write_json(tmp_path / "plan.json", plan)
    trainer = object.__new__(AWMTrainer)
    trainer.config = SimpleNamespace(awm=SimpleNamespace(plan=str(tmp_path / "plan.json")))
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.async_rollout_manager = SimpleNamespace(llm_client=Client())
    trainer.checkpoint_manager = SimpleNamespace(update_weights=lambda n: None, sleep_replicas=lambda: None,
                                                 abort_replicas=lambda: None)
    trainer.actor_rollout_wg = SimpleNamespace(parameter_probe=lambda: [])
    trainer._shutdown_dump_executor = lambda: None
    def compute(batch):
        assert len(batch) == 8 and batch.batch["response_mask"].sum().item() == 1
        return SimpleNamespace(batch={"old_log_probs": batch.batch["old_log_probs"]}), 0
    trainer._compute_old_log_prob = compute
    def update(tensors):
        assert len(tensors["input_ids"]) == 8
        return {"passed": True}
    trainer._update_turns = update
    trainer.probe()
    assert read_json(tmp_path / "gpu_probe/report.json")["optimizer_steps"] == 1
