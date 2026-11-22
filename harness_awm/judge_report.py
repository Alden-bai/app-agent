"""Judge-only reports. Historical labels are comparisons, never gold truth."""
from collections import Counter
from pathlib import Path

from .io import read_json, write_json


def summarize(results, baseline=None):
    baseline = baseline or {}
    unknowns = Counter()
    transitions = Counter()
    for result in results:
        for row in result["items"]:
            if row.get("unknown"):
                unknowns[row["requirement_level"] + ":" + row["unknown"]["kind"]] += 1
        if result["task_id"] in baseline:
            transitions[f'{baseline[result["task_id"]]} -> {result["evaluation_status"]}/{result["task_status"]}'] += 1
    known = [r for r in results if r["evaluation_status"] == "valid"]
    return {"schema_version": "awm-judge-report-v2", "tasks": len(results),
            "evaluation_statuses": dict(Counter(r["evaluation_status"] for r in results)),
            "task_statuses": dict(Counter(r["task_status"] for r in results)),
            "confirmed_passes": sum(r["task_status"] == "confirmed_pass" for r in known),
            "provisional_passes": sum(r["task_status"] == "provisional_pass" for r in known),
            "unknown_reasons": dict(unknowns), "historical_transitions": dict(transitions),
            "usage": {k: sum(r["usage"].get(k, 0) for r in results) for k in (
                "physical_requests", "total_tokens", "tokens_accounted_or_reserved", "unknown_usage_requests", "query_calls", "query_errors")},
            "reward_policy": "separate_versioned_admission_required", "independent_accuracy_measured": False,
            "limitations": ["Historical and model labels are not human gold.", "Provisional passes are not confirmed successes.",
                            "Query recovery and false-positive/negative rates require evidence review; status counts alone cannot establish them."]}


def report_directory(root, output):
    grouped = {}
    paths = [*Path(root).rglob("result.json"), *Path(root).rglob("input_error_*.json")]
    for path in sorted(paths):
        result = read_json(path)
        if result.get("schema_version") != "awm-judge-result-v2":
            continue
        key = result["evaluation_id"]
        if key not in grouped or grouped[key]["evaluation_status"] == "evaluation_error":
            grouped[key] = result
    report = summarize(list(grouped.values()))
    write_json(output, report)
    return report
