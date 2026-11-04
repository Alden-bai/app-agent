"""Pin release, sample BEFORE validation, and keep the denominator frozen."""
from __future__ import annotations

import json
import random
import re
import urllib.request
from collections import Counter
from pathlib import Path

from .common import DATASET, FILES, SEED, digest, now, read_json, rows, save_json
from .network import open_remote


def normalize(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def sample_names(task_rows, seed=SEED, count=10):
    names = [r["scenario"] for r in task_rows]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate scenario in task sampling frame")
    return random.Random(seed).sample(sorted(names), count)


def url_json(url):
    with open_remote(url, timeout=60) as r:
        return json.load(r)


def stream_selected(url, selected, *, source_file=None):
    import hashlib
    h = hashlib.sha256()
    matched, n = [], 0
    response = Path(source_file).open("rb") if source_file else open_remote(url, timeout=120)
    with response as f:
        for line in f:
            h.update(line)
            if not line.strip():
                continue
            item = json.loads(line)
            n += 1
            if normalize(item.get("scenario", item.get("name", ""))) in selected:
                matched.append(item)
    return matched, {"sha256": h.hexdigest(), "records": n, "selected": len(matched)}


def prepare(run_dir, *, source=None, revision=None, endpoint="https://huggingface.co", seed=SEED):
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["seed"] != seed or (revision and manifest["revision"] != revision):
            raise ValueError("Frozen sample differs; use a new run directory")
        if manifest.get("assets_complete"):
            validate_assets(run_dir, manifest)
            return manifest
    else:
        if source:
            if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError("Offline source requires full official --revision SHA")
        else:
            info = url_json(f"{endpoint}/api/datasets/{DATASET}/revision/{revision or 'main'}")
            revision = info["sha"]
        task_file = Path(source) / "gen_tasks.jsonl" if source else None
        if task_file:
            task_rows = rows(task_file)
        else:
            with open_remote(f"{endpoint}/datasets/{DATASET}/resolve/{revision}/gen_tasks.jsonl", timeout=120) as f:
                task_rows = [json.loads(line) for line in f if line.strip()]
        if len(task_rows) != 1000 or sum(len(r["tasks"]) for r in task_rows) != 10000:
            raise ValueError("Unexpected sampling frame: expected 1000 environments / 10000 tasks")
        names = sample_names(task_rows, seed)
        chosen = {r["scenario"]: r for r in task_rows if r["scenario"] in names}
        tasks = [{"scenario": name, "task_idx": i, "task": task}
                 for name in names for i, task in enumerate(chosen[name]["tasks"])]
        manifest = {"schema_version": 1, "dataset": DATASET, "revision": revision,
                    "created_at": now(), "seed": seed, "scenarios": names, "tasks": tasks,
                    "sampling_frame_environments": 1000, "sampling_frame_tasks": 10000,
                    "task_frame_sha256": digest(task_rows), "files": {}, "assets_complete": False}
        save_json(manifest_path, manifest)
    selected = {normalize(n) for n in manifest["scenarios"]}
    for filename in FILES:
        out = run_dir / "assets" / (filename + ".json")
        if filename in manifest["files"] and out.exists():
            if digest(out) != manifest["files"][filename]["selected_sha256"]:
                raise ValueError(f"Frozen asset changed: {filename}")
            continue
        items, evidence = stream_selected(
            f"{endpoint}/datasets/{DATASET}/resolve/{manifest['revision']}/{filename}", selected,
            source_file=Path(source) / filename if source else None)
        save_json(out, items)
        manifest["files"][filename] = {**evidence, "selected_sha256": digest(out)}
        save_json(manifest_path, manifest)
        print(f"asset {filename}: {len(items)} selected", flush=True)
    problems = validate_assets(run_dir, manifest, check_complete=False)
    manifest["asset_problems"] = problems
    manifest["assets_complete"] = True
    save_json(manifest_path, manifest)
    return manifest


def load_asset(run_dir, filename):
    return read_json(Path(run_dir) / "assets" / (filename + ".json"))


def validate_assets(run_dir, manifest, check_complete=True):
    problems = []
    for filename in FILES:
        p = Path(run_dir) / "assets" / (filename + ".json")
        if not p.exists():
            raise ValueError(f"Incomplete preparation: {filename}")
        if digest(p) != manifest["files"][filename]["selected_sha256"]:
            raise ValueError(f"Frozen asset changed: {filename}")
        entries = read_json(p)
        if "verifier" in filename:
            counts = Counter((normalize(e.get("scenario")), e.get("task_idx")) for e in entries)
            for t in manifest["tasks"]:
                key = (normalize(t["scenario"]), t["task_idx"])
                if counts[key] != 1:
                    problems.append({"scenario": t["scenario"], "task_idx": t["task_idx"],
                                     "file": filename, "error": f"record_count={counts[key]}"})
                    continue
                entry = next(e for e in entries if (normalize(e["scenario"]), e.get("task_idx")) == key)
                if entry.get("task") != t["task"]:
                    problems.append({**t, "file": filename, "error": "task_text_mismatch"})
        else:
            counts = Counter(normalize(e.get("scenario", e.get("name"))) for e in entries)
            for name in manifest["scenarios"]:
                if counts[normalize(name)] != 1:
                    problems.append({"scenario": name, "file": filename,
                                     "error": f"record_count={counts[normalize(name)]}"})
    if check_complete and not manifest.get("assets_complete"):
        raise ValueError("Preparation has not completed")
    return problems


def scenario_asset(run_dir, filename, scenario):
    # Duplicate or missing assets are errors, never silently select the first one.
    matches = [e for e in load_asset(run_dir, filename)
               if normalize(e.get("scenario", e.get("name"))) == normalize(scenario)]
    if len(matches) != 1:
        raise ValueError(f"{filename}: {scenario} has {len(matches)} records")
    return matches[0]
