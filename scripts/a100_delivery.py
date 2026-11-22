"""Portable orchestration for the weight-free 27B / 8xA100 delivery."""
import argparse
import heapq
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TASKS = ROOT / "data/training_runtime_20261106_v1"
DATA = ROOT / "data/training_qwen35_27b_v1"
CONTROLS = ("e_commerce_marketplace_10:0", "membership_management_2:0")


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def choose_isolation(cfg, parent):
    from harness_awm.contracts import RunConfig
    from harness_awm.training.baseline import isolation_probe

    attempts = []
    with tempfile.TemporaryDirectory(prefix=".awm-isolation-", dir=parent) as tmp:
        backends = ("landlock", "strict")
        if os.environ.get('AWM_NIOFLOW') == '1':
            backends += ('private-root',)
            requested = os.environ.get('AWM_WORKER_ISOLATION', 'auto')
            if requested != 'auto':
                if requested not in backends:
                    raise ValueError('unsupported AWM_WORKER_ISOLATION')
                backends = (requested,)
        for backend in backends:
            config = RunConfig.model_validate({**cfg, "isolation": backend})
            result = isolation_probe(config, Path(tmp) / backend)
            attempts.append(result)
            if result["passed"]:
                return backend, attempts
    message = 'Both isolation probes failed' if len(backends) == 2 else 'Worker isolation probes failed'
    raise SystemExit(message + ':\n' + json.dumps(attempts, indent=2))


def validate_model_assets(model):
    from harness_awm.io import file_hash
    lock = json.loads((ROOT / "assets/qwen35-27b/model-assets.lock.json").read_text())
    raw = json.loads((model / "config.json").read_text())
    # Base assets are byte-pinned. HF exports may reserialize tokenizer metadata;
    # compare the complete tokenizer state, template and tokenization settings.
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if file_hash(model / name) == lock["files"][name]["sha256"]:
            continue
        expected_path = ROOT / "assets/qwen35-27b" / name
        if raw.get("model_type") != "qwen3_5_text" or name == "chat_template.jinja":
            raise ValueError("27B tokenizer mismatch: " + name)
        expected, actual = json.loads(expected_path.read_text()), json.loads((model / name).read_text())
        if name == "tokenizer_config.json":
            metadata = {"tokenizer_class", "name_or_path", "_name_or_path", "transformers_version", "processor_class", "auto_map"}
            # Retain every original behavior-setting key; extra serialization
            # metadata cannot change the tokenizer.json normalization/token IDs.
            if any(actual.get(k) != v for k, v in expected.items() if k not in metadata):
                raise ValueError("27B tokenizer settings changed")
        elif actual != expected:
            raise ValueError("27B tokenizer state changed")
    text = raw.get("text_config", raw)
    canonical = json.loads((ROOT / "assets/qwen35-27b/config.json").read_text())["text_config"]
    for key in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "vocab_size"):
        if text.get(key) != canonical[key]:
            raise ValueError("not the pinned 27B architecture: " + key)


def longest_smoke_data(output):
    from harness_awm.io import file_hash
    if output.exists():
        raise FileExistsError(output)
    with (DATA / "train.jsonl").open() as stream:
        rows = heapq.nlargest(8, (json.loads(line) for line in stream if line.strip()),
                              key=lambda row: len(row["input_ids"]))
    output.mkdir(parents=True)
    (output / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = json.loads((DATA / "report.json").read_text())
    report.update(train_sha256=file_hash(output / "train.jsonl"), diagnostic_only=True,
                  source_train_sha256=report["train_sha256"], supervised_turns=8)
    write(output / "report.json", report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("init", "doctor", "distributed-check", "prepare-data", "sft-smoke", "sft",
        "calibrate", "admissions", "grpo-probe", "grpo-smoke", "grpo", "evaluate", "skills-prepare", "skills-evaluate"))
    p.add_argument("--work", type=Path, required=True, help="new experiment root, on local storage")
    p.add_argument("--model", type=Path, help="external base weights or selected SFT text checkpoint")
    p.add_argument("--env-python", type=Path, default=ROOT / ".venv-delivery-env/bin/python")
    p.add_argument("--sft-only", action="store_true", help="init/doctor only: offline SFT, no online task workers")
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--output", type=Path)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--global-batch-size", type=int, default=64)
    p.add_argument("--target-updates", type=int)
    p.add_argument("--full-pass", action="store_true",
                   help="Run the selected 2,000 GRPO tasks once and checkpoint every 500 tasks")
    p.add_argument("--groups", type=int)
    p.add_argument("--save-every", type=int)
    p.add_argument("--task", action="append")
    p.add_argument("--skills", type=Path, help="Locally prepared inference_skills.json")
    p.add_argument("--limit", type=int, default=0, help="skills-evaluate only: 0 = full split")
    p.add_argument("--execute", action="store_true", help="skills-evaluate only: run inference and Judge")
    a = p.parse_args()
    from harness_awm.training.verl.coverage import SAVE_EVERY
    if a.save_every is None:
        a.save_every = SAVE_EVERY if a.full_pass else 200
    if a.full_pass and (a.command != "grpo" or a.task or a.save_every != SAVE_EVERY
                        or a.groups is not None or a.target_updates is not None):
        p.error("--full-pass requires the frozen 2,000-task selection and --save-every 500")
    if a.groups is None:
        a.groups = 3200
    if a.target_updates is None:
        a.target_updates = 800
    if a.sft_only and a.command not in {"init", "doctor"}:
        p.error("--sft-only applies to init or doctor")
    if a.skills and a.command not in {"evaluate", "skills-evaluate"}:
        p.error("--skills is only supported for evaluation")
    if a.command == "skills-evaluate" and a.task:
        p.error("skills-evaluate uses --limit, not --task")
    work = a.work.absolute()
    if a.command == "init":
        if work.exists():
            raise FileExistsError(work)
        work.parent.mkdir(parents=True, exist_ok=True)
        cfg = json.loads((ROOT / "delivery/run-config.example.json").read_text())
        cfg.update(runtime_python=str(a.env_python.absolute()), official_root=str(ROOT / "agent-world-model"),
                   judge_inputs_manifest=str(TASKS / "judge_inputs_manifest.json"), reward_admissions=None,
                   judge_rate_db=str(work / "judge-rate.sqlite"))
        if os.environ.get('AWM_NIOFLOW') == '1':
            for field, var in [('base_url', 'AWM_JUDGE_BASE_URL'), ('model', 'AWM_JUDGE_MODEL'),
                               ('api_key_env', 'AWM_JUDGE_KEY_ENV')]:
                if os.environ.get(var):
                    cfg['judge_api'][field] = os.environ[var]
        if not a.sft_only:
            cfg["isolation"], attempts = choose_isolation(cfg, work.parent)
            print("Selected worker isolation:", cfg["isolation"], flush=True)
        work.mkdir(exist_ok=False)
        if a.sft_only:
            (work / "sft-only").touch()
        else:
            write(work / "isolation-selection.json", {"selected": cfg["isolation"], "attempts": attempts})
        write(work / "config.json", cfg)
        cfg.update(purpose="evaluation", reward_admissions=None, candidates_per_task=1)
        write(work / "eval-config.json", cfg)
        print(f"Initialized {work}; set the Judge API key in your own environment or repository .env.")
        return
    if not (work / "config.json").is_file():
        p.error("Run init first")
    offline = (work / "sft-only").exists()
    if offline and a.command not in {"doctor", "distributed-check", "prepare-data", "sft-smoke", "sft", "skills-prepare"}:
        p.error("Online workers require a new work directory initialized without --sft-only")
    os.environ.setdefault("SWANLAB_MODE", "disabled")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    def run(*argv):
        subprocess.run([str(x) for x in argv], cwd=ROOT, check=True)
    def module(name, *argv):
        run(sys.executable, "-m", name, *argv)
    if a.command == "skills-prepare":
        run(sys.executable, ROOT / "scripts/prepare_delivery_skills.py",
            "--output", a.output or work / "inference_skills.json")
        return
    if a.command == "prepare-data":
        module("harness_awm.training.data", "--split", ROOT / "data/training_split_20261106_v1", "--root", ROOT,
               "--tokenizer", ROOT / "assets/qwen35-27b", "--output", a.output or work / "encoded-data")
        return
    if a.command == "doctor":
        options = ["--sft-only"] if offline or a.sft_only else []
        run(sys.executable, ROOT / "scripts/delivery_doctor.py", "--config", work / "config.json",
            "--output", work / ("doctor-sft.json" if options else "doctor.json"), *options)
        return
    if a.command == "distributed-check":
        run(sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
            ROOT / "scripts/delivery_doctor.py", "--distributed")
        return
    if a.command == "calibrate":
        args = [sys.executable, ROOT / "scripts/calibrate_verl_grpo.py", "--config", work / "config.json",
                "--tasks", TASKS, "--controls", ROOT / "configs/training/grpo-calibration-controls.json",
                "--output", work / "calibration"]
        for task in CONTROLS:
            args += ["--task", task]
        run(*args)
        return
    if a.command == "admissions":
        run(sys.executable, ROOT / "scripts/prepare_full_grpo_admissions.py", "--config", work / "config.json",
            "--tasks", TASKS, "--calibrated", work / "calibration/reward_admissions.json",
            "--output", work / "admissions", "--expected-tasks", "2746")
        return
    if not a.model:
        p.error("--model is required; weights are deliberately excluded from the delivery")
    a.model = a.model.resolve()
    validate_model_assets(a.model)
    if a.command in {"sft", "sft-smoke"}:
        smoke = a.command == "sft-smoke"
        output = a.output or work / a.command
        data = DATA
        if smoke:
            data = work / "longest-smoke-data"
            longest_smoke_data(data)
        args = ["--model", a.model, "--data", data, "--output", output, "--gpus", "8", "--optimizer-offload",
                "--global-batch-size", 8 if smoke else a.global_batch_size, "--epochs", 1 if smoke else a.epochs,
                "--learning-rate", a.learning_rate or 1e-5, "--max-length", "32768"]
        if smoke:
            args += ["--max-steps", "1"]
        module("harness_awm.training.verl.sft", "prepare", *args)
        module("harness_awm.training.verl.sft", "check", "--run", output)
        module("harness_awm.training.verl.sft", "train", "--run", output)
    elif a.command.startswith("grpo"):
        probe, smoke = a.command == "grpo-probe", a.command == "grpo-smoke"
        output = a.output or work / a.command
        config = work / "config.json" if probe else work / "admissions/config.json"
        args = ["--preset", "a100x8-27b", "--config", config, "--tasks", TASKS, "--model", a.model,
                "--output", output, "--checkpoint-dir", output / "checkpoints", "--shuffle-tasks",
                "--save-every", 1 if smoke or probe else a.save_every]
        if os.environ.get('AWM_NIOFLOW') == '1':
            args += ['--runtime-profile', 'local-sdpa-cu128']
        if a.full_pass:
            args += ["--full-coverage"]
        else:
            smoke_groups = 32 if os.environ.get('AWM_NIOFLOW') == '1' else 4
            args += ["--groups", smoke_groups if smoke else 1 if probe else a.groups,
                     "--target-updates", 1 if smoke or probe else a.target_updates]
        if a.learning_rate:
            args += ["--learning-rate", a.learning_rate]
        if probe:
            args += ["--stress-probe"]
        for task in a.task or []:
            args += ["--task", task]
        module("harness_awm.training.verl", "prepare", *args)
        if not probe:
            module("harness_awm.training.verl", "check", "--run", output)
        module("harness_awm.training.verl", "probe" if probe else "train", "--run", output)
    elif a.command == "skills-evaluate":
        if not a.output:
            p.error("skills-evaluate requires a new --output")
        args = ["--config", work / "eval-config.json", "--tasks", TASKS, "--model", a.model,
                "--skills", (a.skills or work / "inference_skills.json").resolve(),
                "--split", a.split, "--limit", a.limit, "--output", a.output.resolve()]
        if a.execute:
            args += ["--execute"]
        module("harness_awm.training.skill_eval", *args)
    elif a.command == "evaluate":
        if not a.output:
            p.error("evaluate requires a new --output, named for checkpoint and split")
        args = ["--config", work / "eval-config.json", "--tasks", TASKS, "--model", a.model,
                "--split", a.split, "--dtype", "bfloat16", "--output", a.output]
        if a.skills:
            args += ["--skills", a.skills.resolve()]
        for task in a.task or []:
            args += ["--task", task]
        module("harness_awm.training.evaluate", *args)
        module("harness_awm.training.evaluate", *args, "--execute")


if __name__ == "__main__":
    main()
