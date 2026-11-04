from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .common import digest, now, read_json, save_json


def metrics(cases, mode):
    eligible = [c for c in cases if c.get("status") == "valid"]
    valid = [c for c in eligible if c.get("modes", {}).get(mode, {}).get("status") == "valid"]
    positive = [c for c in valid if c["oracle"]["correct"]]
    negative = [c for c in valid if not c["oracle"]["correct"]]
    accepted = lambda c: c["modes"][mode]["accepted"]
    ratio = lambda n, d: {"numerator": n, "denominator": d, "rate": n / d if d else None}
    by_task = defaultdict(list)
    for c in eligible:
        by_task[c["task"]].append(c)
    covered, passed = [], []
    for task, group in by_task.items():
        scored = [c for c in group if c.get("modes", {}).get(mode, {}).get("status") == "valid"]
        pos = [c for c in scored if c["oracle"]["correct"]]
        neg = [c for c in scored if not c["oracle"]["correct"]]
        if pos and len(neg) >= 3:
            covered.append(task)
            if all(accepted(c) == c["oracle"]["correct"] for c in scored) and len(scored) == len(group):
                passed.append(task)
    return {"positive_pass_rate": ratio(sum(accepted(c) for c in positive), len(positive)),
            "negative_false_accept_rate": ratio(sum(accepted(c) for c in negative), len(negative)),
            "case_accuracy": ratio(sum(accepted(c) == c["oracle"]["correct"] for c in valid), len(valid)),
            "task_all_cases_pass_rate": ratio(len(passed), len(covered)),
            "covered_tasks": covered, "passed_tasks": passed,
            "valid_cases": len(valid), "eligible_cases": len(eligible),
            "errors": dict(Counter(c.get("modes", {}).get(mode, {}).get("status", "not_tested") for c in eligible
                                   if c.get("modes", {}).get(mode, {}).get("status") != "valid"))}


def generate(run_dir):
    root = Path(run_dir).resolve()
    manifest = read_json(root / "manifest.json")
    states = read_json(root / "task_states.json") if (root / "task_states.json").exists() else {}
    cases, task_rows = [], []
    for task in manifest["tasks"]:
        key = f"{task['scenario']}:{task['task_idx']}"
        state = states.get(key, {**task, "status": "pending", "rule_status": "missing", "results": []})
        task_rows.append(state)
        seen = set()
        for p in state["results"]:
            case = {**read_json(root / p), "evaluation_path": p}
            trace_file = root / case['path'] / 'trajectory.json'
            if case.get('status') == 'valid' and trace_file.exists():
                payload = read_json(trace_file)
                calls = [tc for event in payload['trajectory'] for tc in event.get('tool_calls', [])
                         if tc.get('name') == 'call_tool']
                calls = [{'name': c['name'], 'arguments': c['arguments']} for c in calls]
                answer_matters = any(c['name'] == 'requested answer' for c in case.get('oracle', {}).get('checks', []))
                signature = digest([calls, payload['final_answer'] if answer_matters else None])
                if signature in seen:
                    case['status'] = 'duplicate_case'
                    case['exclusion_reason'] = 'Same real tool calls and relevant answer as an earlier case'
                seen.add(signature)
            cases.append(case)
    summary = {"generated_at": now(), "dataset": manifest["dataset"], "revision": manifest["revision"],
               "sample_environments": len(manifest["scenarios"]), "fixed_task_denominator": len(manifest["tasks"]),
               "task_states": dict(Counter(t["status"] for t in task_rows)),
               "rule_states": dict(Counter(t["rule_status"] for t in task_rows)),
               "case_states": dict(Counter(c["status"] for c in cases)),
               "modes": {m: metrics(cases, m) for m in ("code", "sql")},
               "by_environment": {}, "asset_problems": manifest.get("asset_problems", []),
               "sql_evidence_review": dict(Counter(c.get("sql_evidence_review", {}).get("status", "not_extracted") for c in cases)),
               "limits": ["Environment-cluster sample; no IID 100-task population confidence interval.",
                          "Constructed-case accuracy depends on positive/negative mix.",
                          "Reference failure does not prove unsolvability.",
                          "LLM primary decisions are single draws; no stability claim."]}
    for name in manifest["scenarios"]:
        selected = [c for c in cases if c["task"].startswith(name + ":")]
        summary["by_environment"][name] = {"task_states": dict(Counter(t["status"] for t in task_rows if t["scenario"] == name)),
                                              "modes": {m: metrics(selected, m) for m in ("code", "sql")}}
    output_failures = Counter()
    for case in cases:
        if case.get('modes', {}).get('sql', {}).get('status') != 'llm_invalid_output':
            continue
        evaluation = root / case['evaluation_path']
        visited = set()
        while evaluation not in visited:
            visited.add(evaluation)
            recorded = read_json(evaluation)
            log_file = evaluation.parent / 'llm_requests.json'
            if log_file.exists():
                logs = read_json(log_file)
                choices = logs[-1].get('response', {}).get('choices', []) if logs else []
                output_failures[choices[0].get('finish_reason','unknown') if choices else 'unknown'] += 1
                break
            previous = recorded.get('source_evaluation') or recorded.get('oracle_recheck', {}).get('previous_evaluation')
            if not previous:
                output_failures['unknown'] += 1
                break
            evaluation = root / previous
    summary['llm_invalid_output_finish_reasons'] = dict(output_failures)
    save_json(root / "summary.json", summary)
    with (root / "cases.jsonl").open("w") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    with (root / "tasks.jsonl").open("w") as f:
        for row in task_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (root / "cases.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "case", "status", "oracle_correct", "mode", "mode_status", "accepted", "path"])
        writer.writeheader()
        for case in cases:
            for mode in ("code", "sql"):
                verdict = case.get("modes", {}).get(mode, {})
                writer.writerow({"task": case["task"], "case": case["case"], "status": case["status"],
                                 "oracle_correct": case.get("oracle", {}).get("correct"), "mode": mode,
                                 "mode_status": verdict.get("status", "not_tested"), "accepted": verdict.get("accepted"), "path": case["path"]})
    fmt = lambda r: f"{r['numerator']}/{r['denominator']}（{r['rate']:.1%}）" if r['rate'] is not None else "未测得（分母为0）"
    text = ["# AWM 随机抽样正确性检验报告", "", f"生成时间：{summary['generated_at']}。官方版本：`{manifest['revision']}`。",
            f"固定抽取 {len(manifest['scenarios'])} 个环境、{len(manifest['tasks'])} 个任务，种子 `{manifest['seed']}`；未替换失败样本。", "",
            "## 检验范围", "", "正确结果由任务、初态数据库和工具语义独立建立。正反例均需通过独立规则核对；模型只执行官方SQL模式Judge，不生成真值。", "",
            "参考脚本失败不证明任务无解。运行受阻、规则待确认、资产缺陷和验证器异常分别记录。SQL证据内容的独立复核状态单列，不把Judge意见当作证据正确性的证明。", "",
            "## 固定100题的处理状态", "", "| 状态 | 数量 |", "|---|---:|"]
    labels = {"completed_by_tools": "已通过官方工具完成", "confirmed_defect": "已确认矛盾/工具缺陷", "needs_review": "待确认", "runtime_blocked": "运行受阻", "pending": "尚未运行"}
    for state, label in labels.items():
        text.append(f"| {label} | {summary['task_states'].get(state, 0)} |")
    text += ["", "## 官方判定与独立标签的比较", "", "| 模式 | 正例通过率 | 负例误放率 | 案例准确率 | 覆盖≥1正例+3负例任务 | 全部判对/覆盖任务 |", "|---|---|---|---|---|---|"]
    for mode, data in summary["modes"].items():
        text.append(f"| {mode} | {fmt(data['positive_pass_rate'])} | {fmt(data['negative_false_accept_rate'])} | {fmt(data['case_accuracy'])} | {len(data['covered_tasks'])}/{len(task_rows)} | {fmt(data['task_all_cases_pass_rate'])} |")
    text += ['', 'SQL模式的分母仅包括成功取得有效分类的案例；格式错误、截断、传输失败及资产错误全部排除并单列。规则修订和接口恢复不会重采样已经有效的官方判定。',
             '', '## 按环境统计', '', '| 环境 | 工具完成/10 | code 正例通过 | code 负例误放 | SQL 正例通过 | SQL 负例误放 |', '|---|---:|---|---|---|---|']
    for env, data in summary['by_environment'].items():
        code, sql = data['modes']['code'], data['modes']['sql']
        text.append(f"| {env} | {data['task_states'].get('completed_by_tools',0)}/10 | {fmt(code['positive_pass_rate'])} | {fmt(code['negative_false_accept_rate'])} | {fmt(sql['positive_pass_rate'])} | {fmt(sql['negative_false_accept_rate'])} |")
    text += ["", "## 发现的误判", "", "| 任务 | 案例 | 模式 | 独立标签 | 官方接受 | 证据 |", "|---|---|---|---|---|---|"]
    for c in cases:
        for mode, v in c.get("modes", {}).items():
            if c.get('status') == 'valid' and v["status"] == "valid" and v["accepted"] != c.get("oracle", {}).get("correct"):
                text.append(f"| {c['task']} | {c['case']} | {mode} | {c['oracle']['correct']} | {v['accepted']} | [记录]({c['evaluation_path']}) |")
    text += ["", "## 逐题状态和证据", "", "| 任务 | 状态 | 说明/规则 |", "|---|---|---|"]
    for t in task_rows:
        note = str(t.get("reason", "")).replace("|", "\\|").replace("\n", " ")
        link = f"[独立规则]({t['rule_path']})" if t.get("rule_path") else "规则未冻结"
        text.append(f"| {t['scenario']}:{t['task_idx']} | {labels.get(t['status'], t['status'])} | {note} {link} [状态及证据](tasks/{t['scenario']}/{t['task_idx']}/state.json) |")
    text += ["", "## 异常和统计限制", "", f"资产问题：`{json.dumps(summary['asset_problems'], ensure_ascii=False)}`", "",
             f"案例状态：`{json.dumps(summary['case_states'], ensure_ascii=False)}`。SQL证据复核状态：`{json.dumps(summary['sql_evidence_review'], ensure_ascii=False)}`。", ""]
    for mode, data in summary["modes"].items():
        text.append(f"- {mode} 未进入有效判定分母的状态：`{json.dumps(data['errors'], ensure_ascii=False)}`。")
    text += ['', f"LLM非法输出的原始finish_reason：`{json.dumps(summary['llm_invalid_output_finish_reasons'], ensure_ascii=False)}`；length表示原始响应被截断，不当作正常语义拒绝，也不重采样。"]
    text += ['', '首轮冻结规则在工具运行后有版本化修订，依据是用户要求、原始数据库和工具行为（例如日期表示、upsert和初态已满足），不以官方判定为真值。修订前后规则、轨迹、首次判定与接口失败均保留，补跑只处理此前没有有效判定的传输失败或待测案例。', '',
             'SQL证据的partially_checked只表示列出的数据库字段已核对，其余聚合、布尔摘要和遗漏仍需复核；possible_literal_error需要进一步确认表绑定，不能直接当成已证实的SQL错误。']
    text += ["", "这是按环境成组抽取的样本，不把100题视为独立同分布随机样本。案例准确率依赖本次构造的正负比例；不能等同于全部10000题的答案准确率，也不能等同于某个解题模型的成功率。", ""]
    (root / "report.md").write_text("\n".join(text))
    return summary
