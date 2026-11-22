"""Bind existing task splits to frozen runtime assets; no re-splitting or admission."""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

from ..contracts import TaskSpec
from ..io import canonical, db_fingerprint, digest, file_hash, read_json, write_json
from ..judge_v2 import normalize_rubric
from .data import resolve_ref, rows, verify_split

SPLITS = {"grpo_train": "train", "validation": "dev", "test": "test"}
VERSION = "awm-runtime-tasks-v1"


def audit_sources(split, root):
    """Inventory indirect runtime assets often omitted by training-only copies.

    This is a transfer diagnostic, not runtime admission. Catalog fallbacks and
    full database/content validation remain prepare_tasks' responsibility.
    """
    split, root = Path(split).resolve(), Path(root).resolve()
    source, counts = verify_split(split)
    missing, mismatches, directories, per_split = {}, [], set(), {}
    for name in SPLITS:
        affected = 0
        for row in rows(split / f"{name}.jsonl"):
            case = resolve_ref(row["screen_result_ref"], source["source_root"], root).parent
            directories.add(str(case.relative_to(root)))
            task_missing = False
            for kind in ("original", "judge_only_rubric", "screen_result"):
                path = resolve_ref(row[kind + "_ref"], source["source_root"], root)
                if not path.is_file():
                    missing[str(path.relative_to(root))] = {"kind": "direct_source", "split": name}
                    task_missing = True
                elif file_hash(path) != row[kind + "_sha256"]:
                    mismatches.append(str(path.relative_to(root)))
                    task_missing = True
            for filename in ("asset_hashes.json", "initial.db", "environment.json", "task.json",
                             "baseline/facts.json", "baseline/tool_catalog.json"):
                path = case / filename
                if not path.is_file():
                    missing[str(path.relative_to(root))] = {"kind": "runtime_asset", "split": name}
                    task_missing = True
            affected += task_missing
        per_split[name] = {"tasks": counts[name], "tasks_with_missing_or_mismatched_files": affected}
    return {"schema_version": "awm-runtime-source-audit-v1", "source_root": source["source_root"],
            "local_root": str(root), "source_split_sha256": file_hash(split / "manifest.json"),
            "splits": per_split, "missing_count": len(missing), "missing": missing,
            "hash_mismatches": mismatches, "source_case_directories": sorted(directories),
            "runtime_preflight_passed": False,
            "interpretation": "Transfer inventory only; prepare_tasks must verify hashes, databases and catalog fallbacks."}


def select_tasks(tasks, requested):
    if not requested:
        return tasks
    selected = []
    for key in requested:
        matches = [t for t in tasks if key in {t.task_instance_id, f"{t.scenario}:{t.native_task_id}"}]
        if len(matches) != 1 or matches[0] in selected:
            raise ValueError("unknown or duplicate task selection: " + key)
        selected.extend(matches)
    return selected


def prepare_tasks(split, output, root):
    split, output, root = Path(split).resolve(), Path(output).resolve(), Path(root).resolve()
    source, _ = verify_split(split)
    source_hashes = read_json(split / "source_hashes.json")
    split_hash = file_hash(split / "manifest.json")
    if output.exists():
        raise ValueError("output exists; choose a new runtime bundle")

    def relocate(ref):
        return resolve_ref(ref, source["source_root"], root)

    def verified(path, expected):
        if file_hash(path) != expected:
            raise ValueError("runtime source hash mismatch: " + str(path))
        return path

    # The split freezes the screening inventory, which in turn freezes asset manifests.
    inventory_refs = [p for p in source_hashes if p.endswith("/environment_screen_remaining5909_20261104_v1/inventory.jsonl")]
    if len(inventory_refs) != 1:
        raise ValueError("missing or ambiguous frozen screening inventory")
    inventory_path = verified(relocate(inventory_refs[0]), source_hashes[inventory_refs[0]])
    inventory = {r["task_id"]: r for r in rows(inventory_path)}
    selected = {s: list(rows(split / f"{s}.jsonl")) for s in SPLITS}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".runtime-tasks-", dir=output.parent) as tmp:
        staged = Path(tmp) / "bundle"
        staged.mkdir()
        worlds, sources, runtime_counts, catalog_fallbacks = {}, {}, {}, []
        for split_name, entries in selected.items():
            tasks = []
            for row in entries:
                ident = row["task_id"]
                original_path = verified(relocate(row["original_ref"]), row["original_sha256"])
                rubric_path = verified(relocate(row["judge_only_rubric_ref"]), row["judge_only_rubric_sha256"])
                decision = verified(relocate(row["screen_result_ref"]), row["screen_result_sha256"])
                if read_json(decision)["decision"] != "pass":
                    raise ValueError("runtime task is not screen-pass: " + ident)
                case = decision.parent
                frozen = inventory[ident]
                verified(case / "asset_hashes.json", frozen["asset_hashes_sha256"])
                hashes = read_json(case / "asset_hashes.json")
                for name in ("initial.db", "environment.json", "task.json"):
                    verified(case / name, hashes[name])
                original, rubric, public = read_json(original_path), read_json(rubric_path), read_json(case / "task.json")
                if (original["task_id"] != ident or public["task_id"] != ident or
                        original["task"] != row["instruction"] or public["task"] != row["instruction"]):
                    raise ValueError("runtime instruction binding mismatch: " + ident)
                normalize_rubric(original, rubric)
                env = read_json(case / "environment.json")
                catalog_case = case
                if not (case / "baseline/tool_catalog.json").exists() or not read_json(case / "baseline/facts.json").get("mcp_ready"):
                    # Reuse only the public catalog of the identical frozen world,
                    # never another task's actions, answers or screening conclusions.
                    for other in (r for values in selected.values() for r in values if r["scenario"] == row["scenario"]):
                        other_case = verified(relocate(other["screen_result_ref"]), other["screen_result_sha256"]).parent
                        other_hashes = read_json(verified(other_case / "asset_hashes.json", inventory[other["task_id"]]["asset_hashes_sha256"]))
                        if (all(other_hashes[n] == hashes[n] for n in ("initial.db", "environment.json")) and
                                (other_case / "baseline/tool_catalog.json").exists() and
                                read_json(other_case / "baseline/facts.json").get("mcp_ready")):
                            catalog_case = other_case
                            catalog_fallbacks.append({"task_id": ident, "catalog_source_task": other["task_id"]})
                            break
                facts = read_json(catalog_case / "baseline/facts.json")
                tools = read_json(catalog_case / "baseline/tool_catalog.json")["tools"]
                if (facts.get("mcp_ready") is not True or facts.get("initial_unchanged") is not True or
                        facts["database_sha256"] != hashes["initial.db"] or
                        facts["environment_sha256"] != digest(env) or facts["tool_catalog_hash"] != digest(tools)):
                    raise ValueError("screened environment/catalog binding mismatch: " + ident)
                world_id = digest([hashes["initial.db"], hashes["environment.json"], digest(tools)])
                world = Path("worlds") / world_id
                if world_id not in worlds:
                    fingerprint = db_fingerprint(case / "initial.db")
                    if fingerprint != facts["database_logical_fingerprint"]:
                        raise ValueError("screened database fingerprint mismatch: " + ident)
                    (staged / world).mkdir(parents=True)
                    for name in ("initial.db", "environment.json"):
                        shutil.copyfile(case / name, staged / world / name)
                    write_json(staged / world / "tools.json", tools)
                    write_json(staged / world / "catalog_provenance.json", {
                        "source": str(catalog_case), "facts_sha256": file_hash(catalog_case / "baseline/facts.json"),
                        "catalog_sha256": file_hash(catalog_case / "baseline/tool_catalog.json")})
                    worlds[world_id] = fingerprint
                item = Path("judge_sources") / digest(ident)
                (staged / item).mkdir(parents=True)
                for name, path in (("original", original_path), ("rubric", rubric_path)):
                    shutil.copyfile(path, staged / item / f"{name}.json")
                instance = f"{split_hash[:16]}:{ident}"
                task = TaskSpec(protocol_id="awm-single-json-v2", dataset_version=Path(row["environment_dataset_ref"]).name,
                    split=SPLITS[split_name], lineage_group_id=row["scenario"] + ":" + worlds[world_id],
                    scenario=row["scenario"], native_task_id=ident.rsplit(":", 1)[1], task_instance_id=instance,
                    instruction=row["instruction"], initial_state_ref=str(world / "initial.db"),
                    initial_state_hash=hashes["initial.db"], initial_state_fingerprint=worlds[world_id],
                    environment_ref=str(world / "environment.json"), environment_code_hash=digest(env),
                    tool_schema_hash=digest(tools),
                    source_manifest_hash=split_hash, cleaning_status="environment_screen_pass",
                    readiness_profile={"runtime_ready": True, "reward_ready": False,
                                       "screen_decision_sha256": row["screen_result_sha256"]})
                tasks.append(task.model_dump())
                sources[instance] = {name: {"path": str(item / f"{name}.json"),
                                            "sha256": file_hash(staged / item / f"{name}.json")}
                                     for name in ("original", "rubric")}
            (staged / f"{split_name}.jsonl").write_text("".join(canonical(t) + "\n" for t in tasks))
            runtime_counts[split_name] = len(tasks)
        write_json(staged / "judge_inputs_manifest.json", {"schema_version": "awm-judge-input-manifest-v2", "tasks": sources})
        manifest = {"schema_version": VERSION, "source_split_sha256": split_hash,
            "evaluation_scope": "new_tasks_in_seen_environments", "counts": runtime_counts,
            "sft_task_ids": [r["task_id"] for r in rows(split / "sft_trajectories.jsonl")],
            "worlds": len(worlds), "reward_admissions_created": False,
            "catalog_fallbacks": catalog_fallbacks,
            "files": {str(p.relative_to(staged)): file_hash(p) for p in staged.rglob("*") if p.is_file()}}
        write_json(staged / "manifest.json", manifest)
        staged.rename(output)
    return {"output": str(output), "counts": runtime_counts, "worlds": len(worlds),
            "reward_admissions_created": False}


def load_tasks(path, split_name):
    """Load a portable runtime bundle or an existing absolute-path TaskSpec JSONL."""
    if split_name not in SPLITS:
        raise ValueError("unknown runtime split")
    path = Path(path).resolve()
    if not path.is_dir():
        try:
            tasks = [TaskSpec.model_validate(r) for r in rows(path)]
        except ValueError as exc:
            raise ValueError("Expected runtime TaskSpec rows. Convert the frozen split with harness_awm.training.tasks first.") from exc
        if not tasks or any(t.split != SPLITS[split_name] for t in tasks):
            raise ValueError("wrong or empty runtime split")
        if len({t.task_instance_id for t in tasks}) != len(tasks):
            raise ValueError("duplicate runtime tasks")
        return tasks, {"sha256": file_hash(path)}, None
    manifest = read_json(path / "manifest.json")
    if manifest["schema_version"] != VERSION or manifest["evaluation_scope"] != "new_tasks_in_seen_environments":
        raise ValueError("unsupported runtime task bundle")
    for name, expected in manifest["files"].items():
        asset = (path / name).resolve()
        if not asset.is_relative_to(path) or file_hash(asset) != expected:
            raise ValueError("runtime bundle hash mismatch: " + name)
    seen = set(manifest["sft_task_ids"])
    all_tasks = {}
    for name, runtime_split in SPLITS.items():
        if f"{name}.jsonl" not in manifest["files"]:
            raise ValueError("unfrozen runtime task list")
        tasks = [TaskSpec.model_validate(r) for r in rows(path / f"{name}.jsonl")]
        for task in tasks:
            ident = f"{task.scenario}:{task.native_task_id}"
            if ident in seen or task.split != runtime_split:
                raise ValueError("runtime split overlap or incorrect split")
            seen.add(ident)
        if len(tasks) != manifest["counts"][name]:
            raise ValueError("runtime split count mismatch")
        all_tasks[name] = tasks
    tasks = []
    for task in all_tasks[split_name]:
        refs = {}
        for key in ("initial_state_ref", "environment_ref", "verifier_ref"):
            ref = getattr(task, key)
            if ref is None:
                continue
            asset = (path / ref).resolve()
            if ref not in manifest["files"] or not asset.is_relative_to(path):
                raise ValueError("unfrozen runtime asset")
            refs[key] = str(asset)
        tasks.append(task.model_copy(update=refs))
    if not tasks or "judge_inputs_manifest.json" not in manifest["files"]:
        raise ValueError("empty runtime split or missing frozen Judge sources")
    return tasks, {"sha256": file_hash(path / "manifest.json"), "counts": manifest["counts"],
                   "evaluation_scope": manifest["evaluation_scope"]}, str(path / "judge_inputs_manifest.json")


def main():
    from ..contracts import ROOT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--audit", action="store_true", help="write an indirect-source transfer audit instead of building a bundle")
    a = p.parse_args()
    if a.audit:
        if a.output.exists():
            p.error("audit output already exists")
        report = audit_sources(a.split, a.root)
        write_json(a.output, report)
        print(json.dumps({"output": str(a.output), "splits": report["splits"],
                          "missing_count": report["missing_count"], "hash_mismatches": len(report["hash_mismatches"])}, indent=2))
    else:
        print(json.dumps(prepare_tasks(a.split, a.output, a.root), indent=2))


if __name__ == "__main__":
    main()
