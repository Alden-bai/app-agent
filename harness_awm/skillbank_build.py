"""Mine abstract tool workflows and observable failure lessons from sealed GRPO archives.

This first version is deterministic rule-based abstraction, not teacher-LLM distillation.
Only action names/argument keys survive; argument values, answers and rubrics do not.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import tarfile

from .io import canonical, digest, file_hash, read_json, rows, write_json
from .skillbank import VERSION

LESSONS = {
    "protocol_error_limit": (
        "When producing a structured tool action",
        "Return exactly one valid JSON action with the documented fields. Read the schema and correct "
        "the rejected action before trying again; do not repeat malformed output.",
        "tool action json schema arguments format error"),
    "stalled": (
        "When discovery or actions repeat without new information",
        "Review the last observation. Change search terms, follow pagination, or read the specific "
        "schema needed for the next step. Do not repeat an identical request without a new reason.",
        "search find list query tool information discovery repeated stuck"),
    "token_limit": (
        "When a multi-step task requires several tool calls",
        "Plan the required outcomes before exploring. Use targeted tool searches and small result pages; "
        "retain relevant identifiers and reserve room for remaining actions and a supported final answer.",
        "search list find create update report steps task tools"),
}


def training_allowlist(bundle):
    bundle = Path(bundle)
    manifest = read_json(bundle / "manifest.json")
    if manifest.get("schema_version") != "awm-runtime-tasks-v1":
        raise ValueError("expected_frozen_runtime_bundle")
    tasks, seen = {}, set(manifest["sft_task_ids"])
    for name, split in (("grpo_train", "train"), ("validation", "dev"), ("test", "test")):
        path = bundle / (name + ".jsonl")
        if file_hash(path) != manifest["files"][path.name]:
            raise ValueError("split_hash_mismatch")
        for t in rows(path):
            key = f'{t["scenario"]}:{t["native_task_id"]}'
            if key in seen or t["split"] != split:
                raise ValueError("split_overlap")
            seen.add(key)
            if name == "grpo_train":
                tasks[t["task_instance_id"]] = t
    return tasks


def archived_episodes(archive):
    """Read only required JSON files, without extracting paths or databases."""
    wanted = {"task.json", "trajectory.json", "episode_result.json", "evidence_seal.json"}
    episodes = {}
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            parts = Path(member.name).parts
            if len(parts) != 3 or not parts[1].startswith("candidate-") or parts[2] not in wanted:
                continue
            if not member.isfile() or member.size > 32 * 1024 * 1024:
                raise ValueError("invalid_archive_json_member")
            files = episodes.setdefault("/".join(parts[:2]), {})
            if parts[2] in files:
                raise ValueError("duplicate_archive_member")
            files[parts[2]] = stream.extractfile(member).read()
    for name, raw in sorted(episodes.items()):
        if set(raw) != wanted:
            yield name, None
            continue
        data = {k: json.loads(v) for k, v in raw.items()}
        # Seal authenticates actor-visible task and trajectory, never infer success from the actor.
        for key in ("task.json", "trajectory.json"):
            if hashlib.sha256(raw[key]).hexdigest() != data["evidence_seal.json"].get(key):
                raise ValueError(f"sealed_training_evidence_mismatch: {archive}:{name}:{key}")
        yield name, (data, {k: hashlib.sha256(v).hexdigest() for k, v in raw.items()})


def abstract_workflow(trajectory):
    steps = []
    for turn in trajectory.get("trajectory", []):
        for call in turn.get("tool_calls", []):
            if call.get("name") != "call_tool":
                continue
            args = call.get("arguments", {})
            name, values = args.get("tool_name"), args.get("arguments")
            if not isinstance(name, str) or not isinstance(values, dict):
                return []
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,119}", name):
                return []
            keys = sorted(values)
            if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", k) for k in keys):
                return []
            # Explicit tool failures make this episode unsuitable as a positive procedure.
            try:
                observed = json.loads(turn.get("tool_response", {}).get("content", "{}"))
            except (ValueError, TypeError):
                return []
            def failed(value):
                if isinstance(value, dict):
                    return value.get("isError") is True or bool(value.get("transport_error")) or any(failed(v) for v in value.values())
                if isinstance(value, list):
                    return any(failed(v) for v in value)
                return False
            if failed(observed):
                return []
            step = {"tool": name, "argument_keys": keys}
            if not steps or step != steps[-1]:
                steps.append(step)
    # Avoid turning long exploratory trajectories into mandatory procedures.
    return steps if 1 <= len(steps) <= 8 else []


def build(bundle, archives, output):
    output = Path(output)
    if output.exists():
        raise ValueError("output_exists")
    allow = training_allowlist(bundle)
    counts, workflows, lessons, sources = Counter(), {}, {}, []
    for archive in sorted(Path(archives).glob("group-*.tar.gz")):
        source_hash = file_hash(archive)
        sources.append({"path": str(archive.resolve()), "sha256": source_hash})
        for member, episode in archived_episodes(archive):
            counts["episodes_seen"] += 1
            if episode is None:
                counts["incomplete_archives"] += 1
                continue
            data, hashes = episode
            task, result = data["task.json"], data["episode_result.json"]
            original = allow.get(task["task_instance_id"])
            if original is None or task["split"] != "train":
                counts["excluded_nontraining"] += 1
                continue
            for field in ("instruction", "scenario", "native_task_id", "initial_state_hash", "environment_code_hash", "tool_schema_hash"):
                if task[field] != original[field]:
                    raise ValueError("training_task_binding_mismatch:" + field)
            if result["task_instance_id"] != task["task_instance_id"]:
                raise ValueError("episode_result_binding_mismatch")
            if result.get("test_only") is not False or result.get("evaluation_valid") is not True or result.get("execution_integrity") != "valid":
                counts["excluded_invalid_or_test_only"] += 1
                continue
            evidence = {"archive_sha256": source_hash, "member": member,
                        "task_instance_id": task["task_instance_id"], "split": "train", "files": hashes}
            if result.get("task_status") == "confirmed_pass" and result.get("business_label") == "pass":
                steps = abstract_workflow(data["trajectory.json"])
                if not steps:
                    counts["excluded_complex_or_error_workflow"] += 1
                    continue
                key = digest([task["scenario"], task["tool_schema_hash"], steps])
                if key not in workflows:
                    names = [s["tool"] for s in steps]
                    procedure = ["Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes."]
                    procedure += [f'{i}. {s["tool"]}: supply current values for {", ".join(s["argument_keys"]) or "no arguments"}. Inspect the result before proceeding.' for i, s in enumerate(steps, 1)]
                    procedure.append("Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.")
                    workflows[key] = {"skill_id": "workflow_" + key[:16], "kind": "workflow",
                        "scenario": task["scenario"], "tool_schema_hash": task["tool_schema_hash"],
                        "tools": sorted(set(names)), "when_to_apply": "When the request requires operations involving: " + ", ".join(names),
                        "procedure": procedure, "search_text": " ".join(names).replace("_", " "), "evidence": []}
                workflows[key]["evidence"].append(evidence)
                counts["positive_workflows"] += 1
            elif result.get("business_label") == "fail" and result.get("termination_reason") in LESSONS:
                reason = result["termination_reason"]
                when, procedure, search = LESSONS[reason]
                item = lessons.setdefault(reason, {"skill_id": "lesson_" + reason, "kind": "failure_lesson",
                    "scenario": None, "tool_schema_hash": None, "tools": [], "when_to_apply": when,
                    "procedure": [procedure], "search_text": search, "evidence": [],
                    "interpretation": "Rule-based response to an observed termination pattern; not a proven causal diagnosis."})
                item["evidence"].append(evidence)
                counts["failure_lessons"] += 1
            else:
                counts["excluded_unclassified_outcome"] += 1
        if len(sources) % 25 == 0:
            print(canonical({"archives_read": len(sources), "counts": dict(counts)}), flush=True)
    if not sources or not workflows:
        raise ValueError("no_usable_training_workflows")
    skills = sorted([*workflows.values(), *lessons.values()], key=lambda s: s["skill_id"])
    bank = {"schema_version": VERSION, "source_split": "grpo_train",
            "method": "deterministic-tool-workflow-abstraction-v1", "weight_updates": False,
            "runtime_manifest_sha256": file_hash(Path(bundle) / "manifest.json"),
            "skills": skills, "sources": sources, "counts": dict(counts)}
    output.mkdir(parents=True)
    write_json(output / "bank.json", bank)
    report = {"bank_sha256": file_hash(output / "bank.json"), "skills": len(skills),
              "workflows": len(workflows), "failure_lessons": len(lessons),
              "scenarios": len({s["scenario"] for s in workflows.values()}),
              "archives": len(sources), "counts": dict(counts), "evaluation_run": False}
    write_json(output / "build_report.json", report)
    write_json(output / "inference_skills.json", {"bank_path": str((output / "bank.json").resolve()),
               "bank_sha256": report["bank_sha256"], "top_k": 3, "max_context_bytes": 5000})
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--archives", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(canonical(build(a.tasks, a.archives, a.output)))


if __name__ == "__main__":
    main()
