"""Read-only evidence reconciliation and paired pilot report; no LLM calls."""
import json
from collections import Counter
from pathlib import Path

import rubric_d1_repair100 as pilot


def main():
    root = pilot.RUN
    manifest = pilot.verify_manifest()
    summary = pilot.read(root / "summary.json")
    results = pilot.read(root / "results.json")
    amendment = root / "execution_manifest_v2.json"
    if amendment.exists():
        execution = pilot.read(amendment)
        assert pilot.digest(execution["implementation"]) == execution["implementation_sha256"]
        assert pilot.digest(root / "manifest.json") == execution["parent_manifest_sha256"]
    else:
        execution = dict(max_concurrency=manifest["workers"])
    tasks = {r["task_id"]: r for r in manifest["tasks"]}
    paired = []
    for row in results:
        t = tasks[row["task_id"]]
        old = pilot.read(t["old_rubric_ref"])
        new = pilot.read(row["new_rubric_ref"]) if row["status"] == "passed" else None
        detail = dict(task_id=row["task_id"], status=row["status"], task=pilot.read(t["original_ref"])["task"],
            old_unresolved=old["unresolved"], new_unresolved=new["unresolved"] if new else None,
            old_item_counts=dict(Counter(x["requirement_level"] for x in old["rubric"])),
            new_item_counts=dict(Counter(x["requirement_level"] for x in new["rubric"])) if new else None,
            old_ref=t["old_rubric_ref"], new_ref=row.get("new_rubric_ref"))
        if new:
            selected_round = row["rounds"][-1]["revision"]
            folder = root / "tasks" / row["task_id"].replace(":", "__")
            detail["judgment_ref"] = str(folder / f"judgment_{selected_round}.json")
            detail["scores"] = row["rounds"][-1]["scores"]
            inp = pilot.read(t["original_ref"])
            _, pack = pilot.d1.base.prepare_input(inp)
            compiled = pilot.d1.compile_rubric(pilot.read(folder / f"selection_{selected_round}.json"), inp, pack)
            if compiled != new:
                raise ValueError("accepted_compilation_mismatch")
            if pilot.d1.base.evaluate(pilot.read(detail["judgment_ref"]), row["task_id"],
                    {x["id"] for x in new["rubric"]}, {x["candidate_id"] for x in pack["candidates"]})["status"] != "passed":
                raise ValueError("accepted_review_mismatch")
        paired.append(detail)
    # Verify every stage configuration, immutable response and actual model usage.
    stage_status, config_verified, tool_calls, missing_usage, usage_totals = Counter(), 0, 0, [], Counter()
    intervals = []
    for p in sorted((root / "tasks").glob("*/*/completed.json")):
        stage = pilot.read(p)
        stage_status[stage["status"]] += 1
        usage = pilot.read(p.with_name("usage.json"))["usage"]
        config_verified += bool(usage.get("actual_config_verified"))
        tool_calls += usage.get("tool_call_count", 0)
        if usage.get("usage_unknown"):
            missing_usage.append(str(p.with_name("usage.json")))
        else:
            usage_totals.update(usage["actual_usage"])
        if stage["response_sha256"] and pilot.digest(p.with_name("response.txt")) != stage["response_sha256"]:
            raise ValueError("response_hash_mismatch")
        start = pilot.read(p.with_name("started.json"))["started_at"]
        intervals.extend([(start, 1), (stage["finished_at"], -1)])
    concurrent, peak = 0, 0
    for _, delta in sorted(intervals):
        concurrent += delta
        peak = max(peak, concurrent)
    assert peak <= execution["max_concurrency"]
    for key in pilot.old.USAGE_KEYS:
        if usage_totals[key] != summary["usage"]["totals"][key]:
            raise ValueError("usage_reconciliation_mismatch:" + key)
    integrity = dict(selected_tasks=len(paired), recompiled_accepted=sum(x["status"] == "passed" for x in paired),
        stage_status=dict(stage_status), actual_configuration_verified_calls=config_verified, tool_calls=tool_calls,
        unknown_usage_refs=missing_usage, hashes_verified=True, runtime_judge_calls=0, actor_runs=0,
        actual_stage_concurrency_peak=peak, execution_manifest=execution,
        limitation="Independent model review is not human ground truth; unresolved count is not execution unknown rate.")
    pilot.snapshot(root / "paired_comparison.json", paired)
    pilot.snapshot(root / "final_integrity.json", integrity)
    accepted = [x for x in paired if x["status"] == "passed"]
    cleared = [x for x in accepted if not x["new_unresolved"]]
    cleared_with_ones = [x["task_id"] for x in cleared if any(v == 1 for v in x["scores"].values())]
    all_two = [x for x in accepted if all(v == 2 for v in x["scores"].values())]
    full = summary["conservative_full100"]
    reduction = (full["before_unresolved_entries"] - full["after_unresolved_entries"]) / full["before_unresolved_entries"]
    assessment = dict(
        unresolved_entry_reduction_full100=reduction,
        cleared_tasks=len(cleared), cleared_with_score_one=cleared_with_ones,
        all_two_pair=dict(tasks=len(all_two), before=sum(len(x["old_unresolved"]) for x in all_two), after=sum(len(x["new_unresolved"]) for x in all_two)),
        conclusion="Partial static improvement, not evidence of large runtime unknown reduction; not ready for blind full-bank promotion.",
        next_steps=["Enforce structured JSON output and unique/mapped IDs before semantic review.",
                    "Review why substantive unresolved gaps received score 1 and passed; do not silently change this pilot's frozen scores.",
                    "Evaluate retained D1 rubrics on frozen trajectories with trusted evidence to measure runtime unknown and false accepts/rejects."],
        runtime_unknown_reduction=None,
    )
    pilot.snapshot(root / "assessment.json", assessment)
    decreased = sorted((x for x in accepted if len(x["new_unresolved"]) < len(x["old_unresolved"])), key=lambda x: (len(x["new_unresolved"])-len(x["old_unresolved"]), x["task_id"]))
    unchanged = sorted((x for x in accepted if len(x["new_unresolved"]) == len(x["old_unresolved"])), key=lambda x: x["task_id"])
    increased = sorted((x for x in accepted if len(x["new_unresolved"]) > len(x["old_unresolved"])), key=lambda x: (-len(x["new_unresolved"])+len(x["old_unresolved"]), x["task_id"]))
    examples = decreased[:3] + unchanged[:2] + increased[:2]
    pilot.snapshot(root / "examples.json", examples)
    structure_failures = []
    rejected_dimensions = Counter()
    for p in root.glob("tasks/*/round_*_validation.json"):
        value = pilot.read(p)
        structure_failures.append(dict(task=p.parent.name, revision=value["revision"], error=value["feedback"]["error"]))
    for p in root.glob("tasks/*/evaluation_*.json"):
        value = pilot.read(p)
        if value["status"] == "rejected":
            rejected_dimensions.update(value["zero_dimensions"])
    failures = dict(structure_failure_attempts=structure_failures, rejected_review_dimension_counts=dict(rejected_dimensions))
    pilot.snapshot(root / "failure_analysis.json", failures)
    report = (root / "report_zh.md").read_text()
    lead = (f"\n## 效果判断\n\n本轮有部分静态改善，但尚不能证明实际判分unknown大幅下降，不建议直接全量替换。"
        f"按100题保守口径，未决条数净减少{reduction:.2%}；{len(cleared)}题数组清零，其中{len(cleared_with_ones)}题的独立审查仍含1分，不能把清零一律解释为完全消歧。"
        f"通过稿中五维全2的{len(all_two)}题，其未决条数为{assessment['all_two_pair']['before']}→{assessment['all_two_pair']['after']}。\n\n"
        "例如grants_management_1:4的新unresolved为空，但独立审查明确指出六个月窗口边界未完整定义且未记录到unresolved；该题按既定无0规则仍通过。这说明通过门槛和评分一致性还需要改进。这里保留模型原评分，不由主调度重新打分。\n\n"
        "下一步应先约束结构化输出、处理重复键/条目ID及映射错误，并复核关键语义缺口给1分仍通过的问题；之后对冻结轨迹做真实判分对照。不能把缩短未决列表当作运行准确率。\n")
    report += lead
    if amendment.exists():
        report = report.replace(str(root/'prompts/run_100.txt'), str(root/'run_100_effective_v2.txt'))
        report += f"\n用户要求中途从3改为12并发；原在途正常收妥后从同一100题断点续跑，已完成阶段复用。实测阶段并发峰值{peak}。\n"
    report += f"\n结构失败阶段共{len(structure_failures)}次（包括随后修复成功的尝试）；独立审查零分维度计数：{dict(rejected_dimensions)}。详见[失败分类]({root/'failure_analysis.json'})。\n"
    report += "\n## 配对实例与审查材料\n\n以下按未决项减少幅度、持平、增加幅度固定排序选例；不代表总体准确率。\n"
    for x in examples:
        report += f"\n### {x['task_id']}：{len(x['old_unresolved'])}→{len(x['new_unresolved'])}\n\n{x['task']}\n\n"
        report += "旧未决：" + "；".join(u["issue"] for u in x["old_unresolved"]) + "\n\n"
        report += "新未决：" + ("；".join(u["issue"] for u in x["new_unresolved"]) or "无（仍须检查规则内的运行时取证条件）") + "\n\n"
        report += f"[新版规则]({x['new_ref']}) · [独立审查]({x['judgment_ref']})\n"
    report += f"\n## 最终核对\n\n独立模型配置已核验{config_verified}次；工具调用{tool_calls}次；通过稿全部重编译并核对独立审查；响应及输入哈希校验通过。\n\n[逐题配对]({root/'paired_comparison.json'}) · [完整性结果]({root/'final_integrity.json'})\n"
    (root / "final_report_zh.md").write_text(report)
    print(pilot.compact(dict(summary=summary, integrity=integrity, examples=[dict(task_id=x["task_id"], before=len(x["old_unresolved"]), after=len(x["new_unresolved"])) for x in examples])))


if __name__ == "__main__":
    main()
