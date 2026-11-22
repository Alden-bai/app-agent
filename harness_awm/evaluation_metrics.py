"""Scoring-only baseline statistics; never issue or alter training rewards."""
import csv
import math
from datetime import datetime
from pathlib import Path
from statistics import mean, median

from .io import read_json, rows
from .reward_values import TERMINAL_REWARDS, VERSION


def evaluation_score(result):
    """Judge aggregation already requires valid samples and complete evidence.

    Additionally require a settled real episode with a trusted execution. The
    training admission contract is deliberately neither invoked nor bypassed.
    """
    if not (result.get("judge_protocol") == "rubric_v2"
            and result.get("lifecycle") == "SETTLED"
            and result.get("evaluation_valid") is True
            and result.get("evaluation_status") == "valid"
            and result.get("execution_integrity") == "valid"
            and result.get("terminal_state_status") == "trusted"
            and result.get("test_only") is False):
        return None
    return TERMINAL_REWARDS.get((result.get("business_label"), result.get("task_status")))


def distribution(values):
    values = sorted(v for v in values if type(v) in (int, float) and v >= 0 and math.isfinite(v))
    return {"count": len(values), "mean": mean(values) if values else None,
            "median": median(values) if values else None,
            "p90": values[math.ceil(.9 * len(values)) - 1] if values else None}


def summarize(run, manifest):
    run = Path(run)
    tasks = {t["task_instance_id"]: t for t in manifest.get("tasks", [])}
    records = []
    for candidate in manifest["candidates"]:
        folder = run / candidate["path"]
        path = folder / "episode_result.json"
        result = read_json(path) if path.exists() else {}
        task = tasks.get(candidate["task_instance_id"], {})
        trace_exists = (folder / "trajectory.json").is_file()
        events = rows(folder / "events.jsonl") if (folder / "events.jsonl").is_file() else []
        closed = next((e for e in reversed(events) if e["event_type"] == "execution_closed"), {})
        seconds = None
        if events and all(e.get("time") for e in (events[0], events[-1])):
            seconds = (datetime.fromisoformat(events[-1]["time"]) - datetime.fromisoformat(events[0]["time"])).total_seconds()
        failure = run / "failures" / (candidate["rollout_id"] + ".json")
        error = result.get("error") or (read_json(failure).get("error") if failure.exists() else None)
        score = evaluation_score(result)
        records.append({"task_id": f"{task.get('scenario', candidate['scenario'])}:{task.get('native_task_id', '')}",
            "task_instance_id": candidate["task_instance_id"], "rollout_id": candidate["rollout_id"],
            "scenario": candidate["scenario"], "result_present": bool(result),
            "evaluation_status": result.get("evaluation_status", "not_run"),
            "task_status": result.get("task_status"), "eval_reward": score,
            "score_valid": score is not None, "confirmed_success": score == 1,
            "turns": result.get("turns") if trace_exists else None,
            "tool_calls": result.get("tool_calls") if trace_exists else None,
            "termination_reason": result.get("termination_reason", "not_settled"),
            "execution_integrity": result.get("execution_integrity"),
            "terminal_state_status": result.get("terminal_state_status"), "error": error,
            "execution_seconds": closed.get("elapsed_seconds"), "episode_seconds": seconds,
            "actor_tokens_accounted_or_estimated": closed.get("token_count_or_estimate"),
            "trajectory_path": str((folder / "trajectory.json").resolve()) if trace_exists else None,
            "episode_path": str(folder.resolve())})
    scores = [r["eval_reward"] for r in records if r["score_valid"]]
    subsets = {"all_recorded": records, "valid_scored": [r for r in records if r["score_valid"]],
               "confirmed_success": [r for r in records if r["confirmed_success"]]}
    planned = len(records)
    valid, successful = len(scores), sum(r["confirmed_success"] for r in records)
    summary = {"schema_version": "awm-evaluation-statistics-v1", "reward_mapping_version": VERSION,
        "training_reward_issued": False, "planned": planned,
        "result_records": sum(r["result_present"] for r in records),
        "missing_results": sum(not r["result_present"] for r in records),
        "scored": valid, "confirmed_successes": successful,
        "score_coverage": valid / planned if planned else None,
        "success_over_planned": successful / planned if planned else None,
        "success_over_scored": successful / valid if valid else None,
        "eval_reward": {"count": valid, "sum": sum(scores), "mean": mean(scores) if scores else None,
                        "null_count": planned - valid},
        "steps": {name: {key: distribution(r[key] for r in subset) for key in ("turns", "tool_calls")}
                  for name, subset in subsets.items()},
        "timing": {key: distribution(r[key] for r in records) for key in ("execution_seconds", "episode_seconds")},
        "percentile_method": "nearest_rank", "episodes": records}
    return summary


def write_reports(run, report):
    """Export the same records and denominators as metrics.json (null -> blank CSV)."""
    run = Path(run)
    stats = report["evaluation"]
    records = stats["episodes"]
    with (run / "per_task.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        if records:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    def display(value):
        return "N/A" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)
    budget = report.get("budget", {})
    manifest = read_json(run / "manifest.json")
    isolation = manifest.get("config", {}).get("isolation", "unknown")
    lines = ["# 基模／checkpoint 评测报告", "",
        f"隔离方案：{isolation}。" + ("Landlock 文件访问限制；共享 PID/网络，旧内核不限制 truncate 等操作，与 strict 结果应区分。" if isolation == "landlock" else ""), "",
        f"计划 {stats['planned']} 题；结果记录 {stats['result_records']}；缺失 {stats['missing_results']}；有效得分 {stats['scored']}。",
        "", "| 指标 | 结果 |", "|---|---:|",
        f"| 确认成功 / 计划题数 | {stats['confirmed_successes']} / {stats['planned']} = {display(stats['success_over_planned'])} |",
        f"| 确认成功 / 有效得分题数 | {stats['confirmed_successes']} / {stats['scored']} = {display(stats['success_over_scored'])} |",
        f"| 有效得分覆盖率 | {display(stats['score_coverage'])} |",
        f"| 平均 eval_reward | {display(stats['eval_reward']['mean'])}（分母 {stats['scored']}） |", "",
        "eval_reward：确认通过 1、暂定通过 0.3、明确失败 0；无效评测留空。训练 reward 不变，未发放训练奖励。",
        "", "| 子集 | 指标 | 样本数 | 均值 | 中位数 | P90 |", "|---|---|---:|---:|---:|---:|"]
    names = {"all_recorded": "有执行记录", "valid_scored": "有效判分", "confirmed_success": "确认成功"}
    for name, subset in stats["steps"].items():
        for key, values in subset.items():
            lines.append(f"| {names[name]} | {key} | " + " | ".join(display(values[k]) for k in ("count", "mean", "median", "p90")) + " |")
    lines += ["", "turns 包含发现工具、纠错、业务动作和最终回答；tool_calls 仅计实际发起的业务工具调用。P90 使用最近秩法。",
        "", f"Judge 请求：{budget.get('physical_requests', 0)}；已计账或预留 token：{budget.get('tokens_accounted_or_reserved', 0)}。",
        f"完整运行耗时（秒）：{display(report.get('wall_seconds'))}。",
        "", f"终止原因：{report.get('termination_reasons', {})}",
        f"无效原因：{report.get('invalid_reasons', {})}",
        f"Judge 状态：{report.get('judge_v2', {}).get('evaluation_statuses', {})}",
        "", "逐题结果见 per_task.csv，机器可读汇总见 metrics.json。暂定通过不计成功；缺失或异常不补零。",
        "本结果是当前 harness 与 Judge 下的任务表现，不是 Judge 准确率；同一环境跨训练和评测划分是既定范围。"]
    (run / "report_zh.md").write_text("\n".join(lines) + "\n")
