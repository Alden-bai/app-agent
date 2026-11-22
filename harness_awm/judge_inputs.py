"""Source-bound JudgeInput and sealed execution loading, including frozen replay."""
import json
from pathlib import Path
from typing import Literal

from .io import digest, file_hash, read_json, verify_seal
from .judge_v2 import StrictModel, normalize_rubric


RUNTIME_KEYS = ("rollout_id", "attempt_id", "task_instance_id", "termination_reason", "termination_origin",
                "execution_integrity", "terminal_state_status", "turns", "tool_calls", "test_only", "error")
SEALED_REQUIRED = {"initial.db", "final.db", "trajectory.json", "events.jsonl", "task.json"}


class JudgeInput(StrictModel):
    schema_version: Literal["awm-judge-input-v2"] = "awm-judge-input-v2"
    task_instance_id: str
    original: dict
    rubric: dict
    normalized_rubric: dict
    episode: str
    hashes: dict[str, str]


def strict_json(text):
    def pairs(values):
        result = {}
        for k, v in values:
            if k in result:
                raise ValueError("duplicate_json_key:" + k)
            result[k] = v
        return result
    def invalid_constant(value):
        raise ValueError("invalid_json_constant:" + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)


def load_json(path):
    return strict_json(Path(path).read_text())


def read_execution(episode):
    root = Path(episode).resolve()
    sealed = verify_seal(root)
    if not SEALED_REQUIRED <= sealed.keys():
        raise ValueError("required_execution_files_not_sealed")
    task = load_json(root / "task.json")
    trace = load_json(root / "trajectory.json")
    result = load_json(root / "episode_result.json")
    runtime = {k: result.get(k) for k in RUNTIME_KEYS}
    events = [strict_json(line) for line in (root / "events.jsonl").read_text().splitlines() if line.strip()]
    if not events:
        raise ValueError("empty_runtime_events")
    if task["task_instance_id"] != runtime["task_instance_id"] or task["instruction"] != trace["task"]:
        raise ValueError("execution_task_binding_mismatch")
    if trace.get("finished") and not any(r.get("is_final") and r.get("content") == trace["final_answer"] for r in trace["trajectory"]):
        raise ValueError("final_answer_not_in_trajectory")
    # Raw tool records must be individually covered by the evidence seal.
    raw = []
    for event in events:
        if event.get("event_type") != "tool_completed":
            continue
        ref = event["raw_result_ref"]
        p = (root / ref["path"]).resolve()
        if not p.is_relative_to(root) or ref["path"] not in sealed or file_hash(p) != ref["sha256"]:
            raise ValueError("unsealed_raw_tool_result")
        raw.append({"turn": event["turn_id"], "event_id": event["event_id"], "result": load_json(p)})
    return task, trace, runtime, events, raw


def prepare_input(episode, original, rubric):
    normalized = normalize_rubric(original, rubric)  # source-only, before execution inspection
    root = Path(episode).resolve()
    task, trace, runtime, _, _ = read_execution(root)
    native_id = f'{task["scenario"]}:{task["native_task_id"]}'
    if original["task"] != task["instruction"] or original["task_id"] != native_id:
        raise ValueError("source_execution_binding_mismatch")
    hashes = {name: file_hash(root / name) for name in sorted(SEALED_REQUIRED | {"evidence_seal.json"})}
    hashes.update(original=digest(original), rubric=digest(rubric), normalized_rubric=digest(normalized), runtime=digest(runtime))
    return JudgeInput(task_instance_id=task["task_instance_id"], original=original, rubric=rubric,
                      normalized_rubric=normalized, episode=str(root), hashes=hashes).model_dump()


def verify_input(value):
    bundle = JudgeInput.model_validate(value).model_dump()
    actual = prepare_input(bundle["episode"], bundle["original"], bundle["rubric"])
    if actual != bundle:
        raise ValueError("judge_input_binding_or_hash_mismatch")
    return bundle


def seed_evidence(bundle, store):
    task, trace, runtime, events, raw = read_execution(bundle["episode"])
    store.add("task", bundle["original"]["task"])
    for source in bundle["original"]["sources"]:
        if source["id"] != "task":
            store.add("source", source)
    tool_ids = {}
    for row in raw:
        ev = store.add("tool_result", row)
        tool_ids.setdefault(row["turn"], []).append(ev["id"])
    # Preserve illegal post-action state as evidence, never repair it for scoring.
    for event in events:
        if event["event_type"] in {"state_validation_feedback", "snapshot_created"}:
            store.add("state_validation", event)
    compact = []
    for step in trace["trajectory"]:
        row = {k: v for k, v in step.items() if k != "tool_response"}
        if "tool_response" in step:
            turn = step.get("iteration", step.get("turn_id"))
            if turn in tool_ids:
                row["tool_result_evidence_ids"] = tool_ids[turn]
            elif step.get("tool_calls") and all(t.get("name") in {"list_tools", "search_tools", "get_tool_schema"} for t in step["tool_calls"]):
                row["tool_catalog_evidence_id"] = store.add("tool_catalog", step["tool_response"])["id"]
            else:
                # Keep the complete action index visible even when discovery pages
                # contain tens of thousands of escaped catalog characters.
                row["observation_evidence_id"] = store.add("observation", step["tool_response"])["id"]
        compact.append(row)
    store.add("trajectory", compact)
    store.add("final_answer", trace["final_answer"])
    if trace.get("protocol_id") == "awm-single-json-v2":
        store.add("harness_termination", trace.get("harness_termination_report", {}))
    store.add("runtime", {**runtime, "actual_start_utc": events[0]["time"], "actual_end_utc": events[-1]["time"],
                          "timezone": task["timezone"], "clock_policy": task["clock_policy"],
                          "environment_events": [e for e in events if e["event_type"] in {"environment_stopped", "snapshot_created", "execution_closed"}]})
    for snapshot in ("initial", "final"):
        store.call("inspect_schema", {"snapshot": snapshot}, objective="initial_schema", affected_items=[])
    return runtime


def sources_from_manifest(task, manifest_path):
    root = Path(manifest_path).resolve()
    manifest = load_json(root)
    if manifest.get("schema_version") != "awm-judge-input-manifest-v2":
        raise ValueError("unsupported_judge_input_manifest")
    entry = manifest["tasks"][task.task_instance_id]
    values = {}
    for name in ("original", "rubric"):
        ref = entry[name]
        p = (root.parent / ref["path"]).resolve()
        if file_hash(p) != ref["sha256"]:
            raise ValueError("judge_manifest_source_hash_mismatch:" + name)
        values[name] = load_json(p)
    return values


def input_from_manifest(episode, task, manifest_path):
    return prepare_input(episode, **sources_from_manifest(task, manifest_path))
