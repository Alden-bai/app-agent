"""Run held-out evaluation, full-pass GRPO and the frozen SkillRL comparison."""
import fcntl
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from harness_awm.io import digest
from harness_awm.training.verl.coverage import CANDIDATES, SAVE_EVERY, TASK_COUNT, selected_task_ids


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def signature(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(model):
    paths = sorted(p for p in model.iterdir() if p.is_file() and
                   (p.suffix in {".bin", ".safetensors", ".json", ".jinja"}))
    if not any(p.suffix in {".bin", ".safetensors"} for p in paths):
        raise ValueError(f"模型权重不完整 {model}")
    return [[p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in paths]


def metrics(folder, split, ids, model):
    report = read(folder / "metrics.json")
    manifest = read(folder / "manifest.json")
    stats = report["evaluation"]
    episodes = stats["episodes"]
    observed = {e["task_instance_id"] for e in episodes}
    if (observed != ids or len(episodes) != len(ids) or stats["planned"] != len(ids)
            or report["attempted_tasks"] != len(ids) or report["stop_reason"] != "all_tasks_attempted"
            or report["evaluation_split"] != split or report["sampling"] != "greedy"
            or Path(manifest["checkpoint"]["path"]).resolve() != model
            or manifest["checkpoint"]["sha256"] != report["checkpoint_sha256"]):
        raise ValueError(f"题目覆盖、模型或评测终态不匹配 {folder}")
    if not stats["scored"]:
        raise ValueError(f"无有效评分，请检查 Judge 和任务进程日志 {folder}")
    tokens = [e.get("actor_tokens_accounted_or_estimated") for e in episodes]
    return {"model": str(model), "output": str(folder), "planned": len(ids),
            "success": stats["confirmed_successes"], "success_rate": stats["success_over_planned"],
            "mean_reward": stats["eval_reward"]["mean"], "scored": stats["scored"],
            "unscored": len(ids) - stats["scored"], "score_coverage": stats["score_coverage"],
            "actor_tokens_accounted_or_estimated": sum(t for t in tokens if t is not None),
            "episodes_with_actor_usage": sum(t is not None for t in tokens),
            "judge_tokens_accounted_or_reserved": report["budget"]["tokens_accounted_or_reserved"],
            "wall_seconds": report["wall_seconds"]}


def choose(work, name, rows):
    # All validation denominators are identical. For tied scores prefer the
    # higher mean reward, then the earlier step, then the lower learning rate.
    best = min(rows, key=lambda r: (-r["success"], -r["mean_reward"], r["step"], r.get("lr", 0)))
    write(work / (name + "-selection.json"), {"validation": rows, "selected": best})
    return Path(best["model"])


def completed_checkpoints(train, pool_ids):
    summary, plan, coverage = (read(train / name) for name in ("summary.json", "plan.json", "coverage.json"))
    groups = summary["groups"]
    selected = selected_task_ids(pool_ids, plan["run_config"]["seed"])
    if (summary["failed"] or not summary["target_met"]
            or [row["task_instance_id"] for row in groups] != selected
            or [row["task_instance_id"] for row in plan["tasks"]] != selected
            or coverage["selected_task_ids_sha256"] != digest(selected)
            or any(row["candidate_count"] != CANDIDATES for row in groups)
            or not plan.get("full_coverage") or plan["target_updates"] is not None
            or plan["groups"] != TASK_COUNT or plan["group_size"] != CANDIDATES
            or plan["save_unit"] != "groups" or plan["save_every"] != SAVE_EVERY
            or coverage["complete"] is not True or coverage["missing_task_ids"]
            or coverage["trajectories"] != TASK_COUNT * CANDIDATES or coverage["completed_tasks"] != TASK_COUNT):
        raise ValueError("GRPO 的选题、轨迹或任务覆盖记录不符")
    checkpoints = {}
    updates = 0
    for count, group in enumerate(groups, 1):
        status = group["status"]
        primary = group["uniform_primary"]
        rewards = group["final_rewards"]
        if len(rewards) != CANDIDATES:
            raise ValueError(f"第 {count} 题的最终奖励记录不全")
        if primary == "failed" and (group.get("reason") != "all_failed" or status != "held"
                                    or group["quality_scored"] or rewards != [0] * CANDIDATES):
            raise ValueError(f"第 {count} 题的全失败组处理不符")
        if primary == "confirmed_pass":
            if group["quality_scored"] == CANDIDATES and None not in rewards:
                expected = "zero_advantage" if len(set(rewards)) == 1 else "updated"
                if status != expected or group.get("rewards") != rewards:
                    raise ValueError(f"第 {count} 题的全成功组未按最终奖励处理")
            elif status != "held" or group.get("reason") != "unsettled_success_quality":
                raise ValueError(f"第 {count} 题的缺失质量评分未单列")
        updates += int(status == "updated")
        if group["optimizer_step"] != updates:
            raise ValueError(f"第 {count} 题的实际更新数不符")
        if count % SAVE_EVERY == 0:
            checkpoints[count] = updates
    records = [{"completed_tasks": count, "optimizer_steps": step} for count, step in checkpoints.items()]
    if (coverage["checkpoints"] != records or summary["optimizer_steps"] != updates
            or coverage["final_update"] != updates or coverage["optimizer_steps"] != updates):
        raise ValueError("GRPO checkpoint 的题目数与实际更新数不符")
    return checkpoints


def run_pipeline(source, sft_work, work, base, preflight_only=False):
    work.parent.mkdir(parents=True, exist_ok=True)
    env_py = Path(os.environ["ENV_PYTHON"])
    sft_py = Path(os.environ["SFT_PYTHON"])
    grpo_py = Path(os.environ["GRPO_PYTHON"])
    entry = source / "scripts/a100_delivery.py"
    tasks = source / "data/training_runtime_20261106_v1"
    splits = {}
    for split, size in (("test", 588), ("validation", 589), ("grpo_train", 2746)):
        with (tasks / (split + ".jsonl")).open() as stream:
            rows = [json.loads(line)["task_instance_id"] for line in stream if line.strip()]
        if len(rows) != size or len(set(rows)) != size:
            raise ValueError(f"{split} 题目清单不匹配")
        splits[split] = rows if split == "grpo_train" else set(rows)

    def delivery(py, command, *args):
        argv = [str(x) for x in (py, entry, command, "--work", work, *args)]
        print("RUN", command, *(str(x) for x in args), flush=True)
        env = dict(os.environ)
        env["PATH"] = str(py.parent) + os.pathsep + env["PATH"]
        subprocess.run(argv, check=True, cwd=source, env=env)

    if not work.exists():
        delivery(env_py, "init", "--env-python", env_py)
    if (work / "sft-only").exists() or not (work / "isolation-selection.json").is_file():
        raise ValueError(f"ONLINE_WORK 需要独立的在线初始化目录 {work}")
    # Bind expensive reusable stages to this experiment's inputs. A stage
    # interrupted before successful exit keeps its evidence and no receipt.
    state = work / "online-completed"
    state.mkdir(exist_ok=True)
    binding = {"source": str(source), "base": str(base), "base_files": snapshot(base),
               "sft_work": str(sft_work), "config": signature(work / "config.json"),
               "eval_config": signature(work / "eval-config.json"),
               "runtime_manifest": signature(tasks / "manifest.json"),
               "code": {str(p.relative_to(source)): signature(p) for p in
                        [entry, *sorted((source / "harness_awm").rglob("*.py"))]}}
    bound = state / "experiment.json"
    if bound.exists() and read(bound) != binding:
        raise ValueError("代码、模型、配置或数据已改变，请指定新的 ONLINE_WORK")
    write(bound, binding)
    cfg = read(work / "eval-config.json")
    if (Path(cfg['official_root']).resolve() != (source / 'agent-world-model').resolve()
            or Path(cfg['runtime_python']).absolute() != env_py.absolute()):
        raise ValueError('ONLINE_WORK 已绑定到其他源码或 CPU 解释器，请使用新的在线目录')
    for field, variable in [('model', 'AWM_JUDGE_MODEL'), ('base_url', 'AWM_JUDGE_BASE_URL'),
                            ('api_key_env', 'AWM_JUDGE_KEY_ENV')]:
        if os.environ.get(variable) and os.environ[variable] != cfg['judge_api'][field]:
            raise ValueError('Judge 配置已改变，请指定新的 ONLINE_WORK: ' + variable)
    if cfg.get("skills") or cfg.get("judge") != "api":
        raise ValueError("评测的初始配置需要关闭技能并启用正式 API Judge")
    # The same dotenv lookup as the evaluator, without printing any key.
    subprocess.run([env_py, "-c", "import os,sys; from dotenv import load_dotenv; "
                    "load_dotenv('.env',override=False); "
                    "sys.exit(0 if os.environ.get(sys.argv[1]) else '缺少 Job 的 Judge 环境变量 '+sys.argv[1])",
                    cfg["judge_api"]["api_key_env"]], cwd=source, check=True)

    def stage(label, py, command, artifact, *args, model=None):
        receipt = state / (label + ".json")
        inputs = {"args": list(map(str, args)), "command": command}
        if model is not None:
            inputs.update(model=str(model), files=snapshot(model))
        if receipt.exists():
            if not artifact.is_file() or read(receipt) != inputs:
                raise ValueError(f"已记录阶段的输入或输出发生改变 {label}")
            print("REUSE", label, flush=True)
            return
        delivery(py, command, *args)
        if not artifact.is_file():
            raise ValueError(f"阶段缺少产物 {artifact}")
        write(receipt, inputs)

    subprocess.run([env_py, source / "scripts/nioflow_worker_smoke.py", "--config", work / "config.json",
                    "--output", work / "worker-smoke.json"], check=True, cwd=source)
    subprocess.run([grpo_py, "-c", "import torch, vllm._C; from harness_awm.training.verl.__main__ import verify_runtime; "
                    "verify_runtime('local-sdpa-cu128')"], check=True, cwd=source)
    delivery(grpo_py, "doctor")
    delivery(grpo_py, "distributed-check")
    if preflight_only:
        print("ONLINE_PREFLIGHT_PASSED", work, flush=True)
        return
    stage("calibration", env_py, "calibrate", work / "calibration/summary.json")
    if read(work / "calibration/summary.json")["passed"] is not True:
        raise ValueError("在线任务及 Judge 控制题校准失败")
    stage("admissions", env_py, "admissions", work / "admissions/audit.json")
    if read(work / "admissions/audit.json")["task_count"] != 2746:
        raise ValueError("训练题准入不完整")
    skills = work / "inference_skills.json"
    stage("skills", env_py, "skills-prepare", skills.with_suffix(".verification.json"))
    if read(skills)["top_k"] != 3 or read(skills)["max_context_bytes"] != 5000:
        raise ValueError("SkillRL 配置与交付实验不一致")

    def evaluate(label, model, split):
        folder = work / label
        stage(label, sft_py, "evaluate", folder / "metrics.json",
              "--model", model, "--split", split, "--output", folder, model=model)
        return metrics(folder, split, splits[split], model)

    baseline = evaluate("base-test", base, "test")
    candidates = []
    for lr in ("1e-5", "2e-5"):
        run = sft_work / ("sft-lr" + lr)
        plan, summary = read(run / "plan.json"), read(run / "summary.json")
        if (Path(plan["model"]).resolve() != base or plan["learning_rate"] != float(lr)
                or plan["epochs"] != 2 or plan["steps"] % 2 or summary["failed"]
                or summary["exit_code"] != 0 or read(run / "progress.json")["step"] != plan["steps"]):
            raise ValueError(f"SFT 训练产物不匹配 {run}")
        for epoch in (1, 2):
            step = epoch * (plan["steps"] // 2)
            model = (run / "checkpoints" / f"global_step_{step}" / "huggingface").resolve()
            row = evaluate(f"sft-{lr}-epoch-{epoch}-validation", model, "validation")
            candidates.append(dict(row, lr=float(lr), epoch=epoch, step=step))
    start_model = choose(work, "sft", candidates)
    probe = work / "grpo-probe/gpu_probe/report.json"
    stage("grpo-probe", grpo_py, "grpo-probe", probe, "--model", start_model, model=start_model)
    if read(probe)["passed"] is not True or read(probe)["weight_sync"] is not True:
        raise ValueError("GRPO 权重同步未通过")
    smoke = work / "grpo-smoke/summary.json"
    stage("grpo-smoke", grpo_py, "grpo-smoke", smoke, "--model", start_model, model=start_model)
    if read(smoke)["failed"] or read(smoke)["optimizer_steps"] != 1:
        raise ValueError("GRPO 单步短测未完成有效更新")
    train = work / "grpo"
    stage("grpo", grpo_py, "grpo", train / "summary.json", "--model", start_model, "--full-pass", model=start_model)
    checkpoints = completed_checkpoints(train, splits["grpo_train"])
    ckpt_root = train / "checkpoints"
    if {p.name for p in ckpt_root.iterdir() if p.is_dir()} != {f"group_step_{n}" for n in checkpoints}:
        raise ValueError("按训练题数保存的四份 GRPO checkpoint 不完整")
    candidates = []
    for count, updates in checkpoints.items():
        checkpoint = ckpt_root / f"group_step_{count}"
        state = read(checkpoint / "training_state.json")
        if (state["step_unit"] != "groups" or state["groups"] != count or state["optimizer_steps"] != updates
                or not (checkpoint / "data.pt").is_file()):
            raise ValueError(f"第 {count} 题 checkpoint 的训练状态不匹配")
        model = (checkpoint / "actor/huggingface").resolve()
        candidates.append(dict(evaluate(f"grpo-{count}-tasks-validation", model, "validation"),
                               step=count, completed_tasks=count, optimizer_steps=updates))
    best = choose(work, "grpo", candidates)
    pair = work / "best-grpo-test"
    stage("paired-test", sft_py, "skills-evaluate", pair / "comparison.json",
          "--model", best, "--skills", skills, "--split", "test", "--execute", "--output", pair, model=best)
    off, on = (metrics(pair / arm, "test", splits["test"], best) for arm in ("baseline", "skills"))
    plain, augmented = (read(pair / arm / "manifest.json") for arm in ("baseline", "skills"))
    plain_config, augmented_config = dict(plain["config"]), dict(augmented["config"])
    base_config = dict(read(work / "base-test/manifest.json")["config"])
    if (plain_config.pop("skills", None) is not None or augmented_config.pop("skills") != read(skills)
            or base_config.pop("skills", None) is not None or base_config != plain_config
            or plain_config != augmented_config or plain["checkpoint"] != augmented["checkpoint"]
            or plain["candidates"] != augmented["candidates"] or plain["sampling"] != augmented["sampling"]):
        raise ValueError("SkillRL 两组的题目顺序、模型、推理或 Judge 配置不匹配")
    off_rows, on_rows = ({e["task_instance_id"]: e for e in read(pair / arm / "metrics.json")["evaluation"]["episodes"]}
                         for arm in ("baseline", "skills"))
    scored = [(row, on_rows[key]) for key, row in off_rows.items() if row["score_valid"] and on_rows[key]["score_valid"]]
    report = {"base": baseline, "best_grpo": off, "best_grpo_with_skills": on,
              "grpo_gain_pp": 100 * (off["success_rate"] - baseline["success_rate"]),
              "skills_gain_pp": 100 * (on["success_rate"] - off["success_rate"]),
              "failed_to_confirmed_success": sum(a["task_status"] == "failed" and b["confirmed_success"] for a, b in scored),
              "confirmed_success_to_failed": sum(a["confirmed_success"] and b["task_status"] == "failed" for a, b in scored)}
    write(work / "experiment-results.json", report)
    print("ONLINE_COMPLETE", work / "experiment-results.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true",
                        help="Check the online config, worker isolation, key presence and eight-rank NCCL")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    base = Path(os.environ.get("MODEL_DIR", source / "models/Qwen3.5-27B")).resolve()
    sft_work = Path(os.environ["SFT_WORK"]).resolve()
    work = Path(os.environ.get("ONLINE_WORK", sft_work.parent / "qwen-online")).resolve()
    work.parent.mkdir(parents=True, exist_ok=True)
    if work == sft_work:
        sys.exit("ONLINE_WORK 与 SFT_WORK 需要分别存放")
    with (work.parent / ("." + work.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_pipeline(source, sft_work, work, base, args.preflight_only)


if __name__ == "__main__":
    main()
