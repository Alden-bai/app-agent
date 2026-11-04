#!/usr/bin/env python3
"""Freeze and replay historical100 for Judge V2; no actor execution or reward."""
import argparse
import asyncio
import fcntl
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness_awm.backend import Budget
from harness_awm.contracts import RunConfig
from harness_awm.io import canonical, digest, file_hash, read_json, write_json
from harness_awm.judge_inputs import prepare_input, verify_input
from harness_awm.judge_report import summarize
from harness_awm.judge_runtime import config_hash, evaluate_input
from harness_awm.judge_v2 import RubricError, error_result


def prepare(source, development, output, config):
    source, development, output = (Path(p).resolve() for p in (source, development, output))
    original_manifest = read_json(source / "manifest.json")
    if file_hash(source / "manifest.json") != read_json(source / "manifest.sha256.json")["sha256"]:
        raise ValueError("historical_manifest_modified")
    dev = read_json(development / "selection_manifest.json")
    dev_ids = {t["task_id"] for t in dev["tasks"]}
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for task in original_manifest["tasks"]:
        tid = task["id"]
        folder = source / "inputs" / tid
        provenance = {}
        for name in ("gen_tasks", "gen_spec", "gen_db", "rubric"):
            p = folder / (name + ".json")
            expected = original_manifest["input_hashes"][str(p.relative_to(source))]
            if file_hash(p) != expected:
                raise ValueError("historical_input_changed:" + str(p))
            provenance[name] = {"path": str(p), "sha256": expected}
        instruction = read_json(folder / "gen_tasks.json")["tasks"][int(task["task_id"].rsplit(":", 1)[1])]
        sources = [{"id": "task", "kind": "task", "text": instruction},
                   {"id": "contract:" + task["scenario"], "kind": "public_contract", "text": (folder / "gen_spec.json").read_text()}]
        for table in read_json(folder / "gen_db.json")["db_schema"]["tables"]:
            sources.append({"id": "schema:" + table["name"], "kind": "schema", "text": table["ddl"] + "\n" + "\n".join(table.get("indexes", []))})
        original = {"task_id": task["task_id"], "task": instruction, "sources": sources}
        rubric = read_json(folder / "rubric.json")
        history = read_json(source / "tasks" / tid / "result.json")
        entry = {"id": tid, "task_id": task["task_id"], "development": task["task_id"] in dev_ids,
                 "provenance": provenance, "historical_reference": history.get("reference_label", "unknown"),
                 "historical_official_pass": history.get("official_pass"), "historical_result": {"path": str(source / "tasks" / tid / "result.json"), "sha256": file_hash(source / "tasks" / tid / "result.json")}}
        case = output / "cases" / tid
        write_json(case / "original.json", original)
        write_json(case / "rubric.json", rubric)
        try:
            bundle = prepare_input(source / "tasks" / tid / "episode", original, rubric)
            write_json(case / "input.json", bundle)
            entry["input_sha256"] = file_hash(case / "input.json")
        except Exception as exc:
            status = "rubric_error" if isinstance(exc, RubricError) else "evaluation_error"
            if history.get("error") == "ValueError: incomplete_initialization":
                status = "invalid_sample"
            outcome = error_result(task["task_id"], "accuracy100:" + task["task_id"], str(exc), status=status)
            if status == "invalid_sample":
                outcome["sample_validity"] = {"status": "invalid", "evidence_ids": ["historical_initialization_result"],
                                              "reason": history["error"], "source": entry["historical_result"]}
            outcome["evaluation_id"] = digest({"task": task["task_id"], "source": entry["historical_result"], "error": str(exc)})
            write_json(case / "result.json", outcome)
            entry["preparation_error"] = str(exc)
        entries.append(entry)
    manifest = {"schema_version": "awm-judge-regression-v2", "source_run": str(source),
                "source_manifest_sha256": file_hash(source / "manifest.json"), "development_pack": str(development),
                "development_manifest_sha256": file_hash(development / "selection_manifest.json"),
                "config": config.model_dump(), "judge_configuration_hash": config_hash(config), "tasks": entries,
                "historical_labels_hidden_from_judge": True, "independent_test": False, "actor_reruns": 0,
                "rubric_policy": "frozen_source_bound_logical_adapter; no_substantive_predicate_repairs", "reward_policy": "not_implemented"}
    write_json(output / "manifest.json", manifest)
    write_json(output / "manifest.sha256.json", {"sha256": file_hash(output / "manifest.json")})
    return report(output)


def manifest_for(run):
    run = Path(run)
    if file_hash(run / "manifest.json") != read_json(run / "manifest.sha256.json")["sha256"]:
        raise ValueError("regression_manifest_modified")
    return read_json(run / "manifest.json")


def report(run):
    run = Path(run)
    manifest = manifest_for(run)
    pairs = [(t, read_json(run / "cases" / t["id"] / "result.json")) for t in manifest["tasks"] if (run / "cases" / t["id"] / "result.json").exists()]
    baseline = {t["task_id"]: t["historical_reference"] for t in manifest["tasks"]}
    data = summarize([r for _, r in pairs], baseline)
    data.update(planned=len(manifest["tasks"]), pending=[t["id"] for t in manifest["tasks"] if not (run / "cases" / t["id"] / "result.json").exists()],
                development20=summarize([r for t, r in pairs if t["development"]], baseline), actor_reruns=0)
    data["review_queue"] = [{"id": t["id"], "task_id": t["task_id"], "old_reference": t["historical_reference"],
        "task_status": r["task_status"], "evaluation_status": r["evaluation_status"], "result": str(run / "cases" / t["id"] / "result.json")}
        for t, r in pairs if r["task_status"] == "provisional_pass" or t["historical_reference"] == "unknown" and r["business_label"] != "unknown"]
    # Mechanical source/seal verification is separate from semantic evidence review.
    checks = []
    audit_path = run / "semantic_audit.json"
    audit = {a["id"]: a for a in read_json(audit_path).get("cases", [])} if audit_path.exists() else {}
    outcomes = {t["id"]: r for t, r in pairs}
    for t in manifest["tasks"]:
        case = run / "cases" / t["id"]
        result = outcomes.get(t["id"])
        reviewed = bool(result and t["id"] in audit and audit[t["id"]].get("result_sha256") == file_hash(case / "result.json"))
        recovered = total = 0
        evidence_file = Path(result["attempt"]) / "evidence.json" if result and result["attempt"] else None
        if evidence_file and evidence_file.exists():
            history = {}
            for ev in read_json(evidence_file).values():
                if "objective" in ev and ev["objective"] != "initial_schema":
                    history.setdefault(ev["objective"], []).append(ev["valid"])
            total = sum(False in values for values in history.values())
            recovered = sum(False in values and values[-1] for values in history.values())
        execution_intact = None
        if "input_sha256" in t:
            verify_input(read_json(case / "input.json"))
            execution_intact = True
        checks.append({"id": t["id"], "task_id": t["task_id"], "development": t["development"],
                       "sources_intact": all(file_hash(v["path"]) == v["sha256"] for v in t["provenance"].values()),
                       "input_intact": file_hash(case / "input.json") == t["input_sha256"] if "input_sha256" in t else None,
                       "execution_intact": execution_intact, "model_assessment_completed": bool(result and result["items"]), "semantic_review_completed": reviewed,
                       "failed_query_objectives": total, "recovered_query_objectives": recovered})
    data["query_recovery"] = {"failed_objectives": sum(x["failed_query_objectives"] for x in checks), "recovered_objectives": sum(x["recovered_query_objectives"] for x in checks)}
    data["semantic_audit"] = {"reviewed": sum(x["semantic_review_completed"] for x in checks),
                              "development_reviewed": sum(x["development"] and x["semantic_review_completed"] for x in checks),
                              "method": "Codex source/evidence review, not independent human gold"}
    reviewed_ids = {c["id"] for c in checks if c["semantic_review_completed"]}
    data["semantic_audit"]["audit_statuses"] = dict(Counter(audit[i]["audit_status"] for i in reviewed_ids))
    data["semantic_audit"]["needs_review"] = sorted(i for i in reviewed_ids if audit[i]["audit_status"] == "needs_review")
    data["preflight"] = {"sources_verified": sum(x["sources_intact"] for x in checks),
                         "sealed_executions_verified": sum(x["execution_intact"] is True for x in checks),
                         "development_inputs_verified": sum(x["development"] and x["execution_intact"] is True for x in checks)}
    if (run / "budget.sqlite").exists():
        data["authorized_budget_usage"] = Budget(run / "budget.sqlite", manifest["config"]["max_physical_requests"], manifest["config"]["max_total_tokens"]).summary()
    write_json(run / "source_evidence_checks.json", checks)
    write_json(run / "summary.json", data)
    lines = ["# Judge V2 冻结轨迹回归", "", f"计划 {data['planned']} 题，已记录 {data['tasks']} 题，待处理 {len(data['pending'])} 题。", "",
             "已有开发暴露，历史模型标签不是人工真值；本报告不声称独立准确率。奖励未计算，Agent 重跑次数为 0。", "",
             "| 评测状态 | 数量 |", "|---|---:|", *[f"| {k} | {v} |" for k,v in data["evaluation_statuses"].items()], "",
             "| 任务状态 | 数量 |", "|---|---:|", *[f"| {k} | {v} |" for k,v in data["task_statuses"].items()], "",
             "## 开发20题", "", canonical(data["development20"]), "", "## 逐题结果", "",
             "| 样本 | 原参考 | 新评测 | 新状态 |", "|---|---|---|---|"]
    if audit:
        lines[4:4] = [f"已完成来源与证据审查 {data['semantic_audit']['reviewed']} 题，其中开发集 {data['semantic_audit']['development_reviewed']} 题；仍有 {len(data['semantic_audit']['needs_review'])} 题需要进一步裁定。模型标签不等于审查确认。详见 [逐题审查](audit_zh.md)。", ""]
    lines += [f"| {t['id']} {t['task_id']} | {t['historical_reference']} | {outcomes[t['id']]['evaluation_status'] if t['id'] in outcomes else 'pending'} | {outcomes[t['id']]['task_status'] if t['id'] in outcomes else 'awaiting_model_evaluation'} |" for t in manifest["tasks"]]
    lines += ["", "## 取证与复核", "", f"失败取证目标 {data['query_recovery']['failed_objectives']} 个，随后查询恢复 {data['query_recovery']['recovered_objectives']} 个。",
              "", "source_evidence_checks.json 记录来源/输入完整性；review_queue 保存旧 unknown 转确定结论及暂定通过的复核清单。语义准确性需逐项查看证明与执行证据，不能用来源哈希通过替代。", "", "## 调用用量", "", canonical(data["usage"]), "", "授权预算累计（含继承的开发轮次）：" + canonical(data.get("authorized_budget_usage", {}))]
    if audit:
        lines += ["", "## 来源与证据审查", "", canonical(data["semantic_audit"]), "", "详细记录见 semantic_audit.json；原模型结果保留，审查意见不覆盖模型标签。"]
        lines += ["", "以下结论仍有来源解释、样本前提或证明充分性争议，不能据此直接宣称 Judge 语义准确：", ""]
        lines += [f"- {i}: {audit[i]['notes']}" for i in data["semantic_audit"]["needs_review"]]
    (run / "report_zh.md").write_text("\n".join(lines) + "\n")
    return data


async def run_all(run, phase):
    run = Path(run).resolve()
    manifest = manifest_for(run)
    config = RunConfig.model_validate(manifest["config"])
    if config_hash(config) != manifest["judge_configuration_hash"]:
        raise ValueError("judge_implementation_changed; prepare_a_new_regression_run")
    budget = Budget(run / "budget.sqlite", config.max_physical_requests, config.max_total_tokens)
    sem = asyncio.Semaphore(config.judge_concurrency)
    worker_sem = asyncio.Semaphore(config.judge_concurrency)

    async def one(entry):
        case = run / "cases" / entry["id"]
        if (case / "result.json").exists():
            return
        if file_hash(case / "input.json") != entry["input_sha256"]:
            raise ValueError("frozen_judge_input_changed")
        async with worker_sem:
            result = await evaluate_input(read_json(case / "input.json"), config, budget, sem, case / "evaluations")
            write_json(case / "result.json", result)
            print(canonical({"id": entry["id"], "status": result["evaluation_status"], "task_status": result["task_status"], "calls": result["usage"].get("physical_requests", 0)}), flush=True)

    selected = [t for t in manifest["tasks"] if phase == "all" or t["development"]]
    try:
        await asyncio.gather(*(one(t) for t in selected))
    finally:
        data = report(run)
    return data


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False, interpolate=False)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--source", default=str(ROOT / "runs/openai_gpt6_luna_judge_accuracy100_20261103"))
    p.add_argument("--development", default=str(ROOT / "runs/judge_development20_20261103"))
    p.add_argument("--run", required=True)
    p.add_argument("--config", required=True)
    p = sub.add_parser("run")
    p.add_argument("--run", required=True)
    p.add_argument("--phase", choices=["development", "all"], default="development")
    p = sub.add_parser("report")
    p.add_argument("--run", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.source, args.development, args.run, RunConfig.model_validate(read_json(args.config)))
    elif args.action == "report":
        result = report(args.run)
    else:
        with (Path(args.run) / "runner.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = asyncio.run(run_all(args.run, args.phase))
    print(canonical({k: result[k] for k in ("planned", "tasks", "evaluation_statuses", "task_statuses", "usage")}), flush=True)


if __name__ == "__main__":
    main()
