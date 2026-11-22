"""Freeze task-level SFT/GRPO manifests without changing source evidence or training.

All retained SFT episodes are used. RL tasks are stratified within scenarios;
each scenario has a GRPO training task. These are manifests, not tokenized data.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path
import re
import tempfile
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("grpo_train", "validation", "test")
WEIGHTS = (70, 15, 15)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_key(seed, value):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def normalized(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def quotas(n):
    """Largest remainders; ties follow SPLITS (validation before test)."""
    counts = [n * w // 100 for w in WEIGHTS]
    order = sorted(range(3), key=lambda i: (-(n * WEIGHTS[i] % 100), i))
    for i in order[:n - sum(counts)]:
        counts[i] += 1
    return dict(zip(SPLITS, counts))


def assign(rows, seed, forced_train):
    """Allocate global quotas while keeping scenario allocations near 70/15/15.

Seeded hashes break allocation ties and order tasks. Reserve at least one train
task per scenario and all development exclusions before allocating other slots.
"""
    by_scenario = defaultdict(list)
    for row in rows:
        by_scenario[row["scenario"]].append(row)
    if len({r["task_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate task IDs")
    target = quotas(len(rows))
    allocation, used = {}, Counter()
    for scenario, items in sorted(by_scenario.items()):
        n_train = max(1, sum(r["task_id"] in forced_train for r in items))
        allocation[scenario] = Counter(grpo_train=n_train)
        used["grpo_train"] += n_train
    if used["grpo_train"] > target["grpo_train"]:
        raise ValueError("train quota cannot satisfy scenario/development constraints")
    heap = []

    def push(scenario, split):
        count = allocation[scenario][split]
        deficit = len(by_scenario[scenario]) * WEIGHTS[SPLITS.index(split)] - count * 100
        heapq.heappush(heap, (-deficit, stable_key(seed, scenario + ":" + split),
                              scenario, split, count))

    for scenario in sorted(by_scenario):
        for split in SPLITS:
            push(scenario, split)
    while sum(used.values()) < len(rows):
        if not heap:
            raise ValueError("allocation constraints exhausted")
        _, _, scenario, split, count = heapq.heappop(heap)
        a = allocation[scenario]
        if count != a[split] or sum(a.values()) >= len(by_scenario[scenario]) or used[split] >= target[split]:
            continue
        a[split] += 1
        used[split] += 1
        push(scenario, split)
    result = {}
    for scenario, items in sorted(by_scenario.items()):
        forced = sorted((r for r in items if r["task_id"] in forced_train), key=lambda r: r["task_id"])
        remaining = sorted((r for r in items if r["task_id"] not in forced_train),
                           key=lambda r: (stable_key(seed, r["task_id"]), r["task_id"]))
        for row in forced:
            result[row["task_id"]] = "grpo_train"
        offset = 0
        for split in SPLITS:
            count = allocation[scenario][split] - (len(forced) if split == "grpo_train" else 0)
            for row in remaining[offset:offset + count]:
                result[row["task_id"]] = split
            offset += count
        assert offset == len(remaining)
    assert dict(Counter(result.values())) == target
    return result


def supervision(turn_ids, excluded):
    if len(turn_ids) != len(set(turn_ids)) or len(excluded) != len(set(excluded)):
        raise ValueError("duplicate turn IDs")
    if not set(excluded) <= set(turn_ids):
        raise ValueError("mask points outside the episode")
    return [{"turn_id": i, "supervise_current_response": i not in excluded,
             "response_loss_weight": int(i not in excluded)} for i in sorted(turn_ids)]


def build(root, output, seed, replay_frozen=False):
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise ValueError("output already exists; use a new version")
    source_hashes = {}
    disk_root = root
    frozen_split = disk_root / 'data/training_split_20261106_v1'
    pinned = None
    if replay_frozen:
        manifest = json.loads((frozen_split / 'manifest.json').read_text())
        pin_file = frozen_split / 'source_hashes.json'
        if sha(pin_file) != manifest['files'][pin_file.name] or seed != manifest['seed']:
            raise ValueError('frozen split identity or seed mismatch')
        root = Path(manifest['source_root'])
        pinned = json.loads(pin_file.read_text())

    def local(path):
        return disk_root / Path(path).relative_to(root) if replay_frozen else Path(path)

    def read(path, expected=None, lines=False):
        path = Path(path).resolve()
        data = local(path).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError(f"source hash mismatch: {path}")
        if str(path) in source_hashes and source_hashes[str(path)] != digest:
            raise ValueError(f"source changed during build: {path}")
        source_hashes[str(path)] = digest
        return [json.loads(line) for line in data.splitlines() if line.strip()] if lines else json.loads(data)

    audit = root / "cleaning_runs/sft_review_full966_v1"
    delivery = read(audit / "delivery_manifest.json")
    for path, expected in delivery["files"].items():
        # The transfer's frozen input set excludes ancillary human-readable
        # review reports. Every consumed review/trajectory file is pinned.
        if pinned is not None and path not in pinned and not local(path).is_file():
            continue
        if sha(local(path)) != expected:
            raise ValueError(f"audit deliverable changed: {path}")
    acceptance = read(audit / "acceptance.json")
    assert acceptance["passed"] and acceptance["complete"]
    keep, reject = read(audit / "keep.json"), read(audit / "reject.json")
    assert len(keep) == len(set(keep)) == 758 and not set(keep) & set(reject)
    candidates = read(audit / "candidates.json")
    reviews = read(audit / "reviews.jsonl", lines=True)
    review_map = {r["id"]: r for r in reviews}
    masks = read(audit / "supervision_annotations.json")
    frozen = read(audit / "source_hashes.json")
    assert len(review_map) == len(reviews) == len(candidates) == 966
    sft, turns = [], []
    for row in sorted(candidates, key=lambda r: r["id"]):
        ident = row["id"]
        if ident not in keep:
            continue
        episode = Path(row["episode"])

        def frozen_read(path):
            return read(path, frozen[str(path.resolve())])

        task = frozen_read(episode / "task.json")
        result = frozen_read(episode / "episode_result.json")
        review = review_map[ident]
        assert review["decision"] == "keep"
        assert sorted(review["exclude_loss_turn_ids"]) == sorted(masks[ident])
        turn_folder = episode / 'model_turns'
        paths = sorted(turn_folder / p.name for p in local(turn_folder).iterdir())
        ids = [int(p.name) for p in paths]
        assert ids == list(range(1, row["turn_count"] + 1))
        assert result["turns"] == len(paths) and result["completion_status"] == "completed"
        assert not result["test_only"] and result["execution_integrity"] == "valid"
        flags = {r["turn_id"]: r for r in supervision(ids, masks[ident])}
        assert any(r["supervise_current_response"] for r in flags.values())
        for path in paths:
            inp, out = frozen_read(path / "input.json"), frozen_read(path / "output.json")
            flag = flags[int(path.name)]
            assert isinstance(inp["messages"], list) and inp["messages"]
            if flag["supervise_current_response"]:
                assert out["finish_reason"] == "stop" and not out["test_only"] and out["content"].strip()
            turns.append({"trajectory_id": ident, "task_id": row["task_id"], **flag,
                          "input_ref": str(path / "input.json"), "input_sha256": frozen[str(path / "input.json")],
                          "output_ref": str(path / "output.json"), "output_sha256": frozen[str(path / "output.json")]})
        sft.append({"trajectory_id": ident, "task_id": row["task_id"], "scenario": row["scenario"],
                    "split": "sft_train", "instruction": task["instruction"], "episode_ref": str(episode),
                    "source_split": task["split"], "protocol_id": task["protocol_id"],
                    "turn_count": len(ids), "exclude_loss_turn_ids": sorted(masks[ident]),
                    "supervised_turn_ids": [i for i in ids if i not in masks[ident]],
                    "review_ref": str(audit / "reviews.jsonl"), "review_id": ident})
    assert len(sft) == 758 and len(turns) == 7071
    assert sum(not r["supervise_current_response"] for r in turns) == 357
    sft_scenarios = {r["scenario"] for r in sft}
    inventory_rows = read(root / "runs/environment_screen_remaining5909_20261104_v1/inventory.jsonl", lines=True)
    inventory = {r["task_id"]: r for r in inventory_rows}
    assert len(inventory_rows) == len(inventory) == 5909
    screened, decisions = [], Counter()
    for shard in ("A", "B"):
        folder = root / "runs/environment_screen_dual_20261105_v1" / shard
        assert read(folder / "acceptance.json")["passed"]
        final = read(folder / "final_results.jsonl", lines=True)
        passed = read(folder / "pass.jsonl", lines=True)
        assert {r["task_id"] for r in passed} == {r["task_id"] for r in final if r["final_decision"] == "pass"}
        decisions.update(r["final_decision"] for r in final)
        screened.extend(passed)
    assert decisions == Counter({"pass": 3923, "exclude": 1725, "review": 261})
    assert len(screened) == len({r["task_id"] for r in screened}) == 3923
    exclusions = set(read(root / "configs/evaluation/development_exclusions.json")["task_ids"])
    rl = []
    for row in sorted(screened, key=lambda r: r["task_id"]):
        source = inventory[row["task_id"]]
        assert source["case_id"] == row["case_id"] and source["scenario"] == row["scenario"]
        original = read(source["original_ref"], source["original_sha256"])
        assert original["task_id"] == row["task_id"]
        read(source["rubric_source_ref"], source["rubric_source_sha256"])
        read(row["final_ref"], row["final_sha256"])
        rl.append({"task_id": row["task_id"], "case_id": row["case_id"], "scenario": row["scenario"],
                   "instruction": original["task"], "environment_screen": "pass",
                   "seen_in_sft": row["scenario"] in sft_scenarios,
                   "development_exclusion": row["task_id"] in exclusions,
                   "environment_dataset_ref": str(root / "data/awm-dde80a0-repaired-v1"),
                   "original_ref": source["original_ref"], "original_sha256": source["original_sha256"],
                   "judge_only_rubric_ref": source["rubric_source_ref"],
                   "judge_only_rubric_sha256": source["rubric_source_sha256"],
                   "screen_result_ref": row["final_ref"], "screen_result_sha256": row["final_sha256"]})
    combined = sft + rl
    assert len({r["task_id"] for r in combined}) == len(combined)
    exact = defaultdict(list)
    for row in combined:
        exact[normalized(row["instruction"])].append(row["task_id"])
    duplicate_groups = [v for v in exact.values() if len(v) > 1]
    if duplicate_groups:
        raise ValueError(f"exact duplicate instructions require grouping: {duplicate_groups}")
    assigned = assign(rl, seed, exclusions)
    for row in rl:
        row["split"] = assigned[row["task_id"]]
    by_split = {s: [r for r in rl if r["split"] == s] for s in SPLITS}
    assert all(r["split"] == "grpo_train" for r in rl if r["development_exclusion"])
    scenario_summary = []
    for scenario in sorted({r["scenario"] for r in rl}):
        entry = {"scenario": scenario, "seen_in_sft": scenario in sft_scenarios,
                 **{s: sum(r["scenario"] == scenario for r in by_split[s]) for s in SPLITS}}
        assert entry["grpo_train"] >= 1
        scenario_summary.append(entry)
    summary = {"schema_version": "awm-training-split-v1", "seed": seed,
               "evaluation_scope": "New tasks in training environments; scenario overlap is intentional.",
               "sft": {"trajectories": len(sft), "scenarios": len(sft_scenarios), "turns": len(turns),
                       "masked_turns": 357, "supervised_turns": 6714,
                       "trajectories_with_mask": sum(bool(r["exclude_loss_turn_ids"]) for r in sft),
                       "validation_trajectories": 0, "all_kept_used_as_requested": True},
               "rl_screen": dict(decisions),
               "rl_splits": {s: {"tasks": len(by_split[s]), "scenarios": len({r["scenario"] for r in by_split[s]}),
                                  "seen_in_sft_tasks": sum(r["seen_in_sft"] for r in by_split[s])} for s in SPLITS},
               "development_tasks_forced_to_train": sorted(exclusions & set(assigned)),
               "checks": {"task_id_overlap": 0, "exact_normalized_duplicate_groups": duplicate_groups,
                          "all_pass_tasks_assigned_once": True, "source_mask_matches_reviews": True,
                          "every_eval_scenario_has_grpo_train_task": True},
               "allocation_method": "Scenario-stratified deficit allocation to global largest-remainder 70/15/15 quotas. Reserve >=1 train task per scenario and all development exclusions; SHA256(seed:identifier) breaks ties and orders tasks.",
               "supervision_policy": {"unit": "whole current assistant response",
                                      "masked_response_labels": -100, "all_prompt_labels": -100,
                                      "preserve_original_input_messages": True,
                                      "keep_masked_actions_in_later_history": True,
                                      "token_labels_materialized": False,
                                      "turn_manifest_is_not_a_trainer_loss_mask": True},
               "limitations": ["Source episode development metadata and reward admission remain unchanged.",
                               "Manifest creation does not implement a trainer, tokenizer, reward adapter or runtime admission.",
                               "Semantic equivalence and answer exposure are not ruled out by exact-text checks.",
                               "Screen pass is evidence of no identified blocking environment issue, not a guarantee.",
                               "Human validation in the source SFT review remains pending; no labels were changed.",
                               "Judge-only references and review metadata must never become actor input."]}
    # Recheck every consumed source before publishing the new, independent version.
    for path, expected in source_hashes.items():
        if sha(local(path)) != expected:
            raise ValueError(f"source changed before publication: {path}")
    summary["checks"]["source_files_verified"] = len(source_hashes)
    if pinned is not None and pinned != source_hashes:
        raise ValueError('reconstructed source closure differs from frozen split')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".training-split-", dir=output.parent) as staging:
        folder = Path(staging) / "bundle"
        folder.mkdir()

        def write_json(name, value):
            (folder / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

        def write_rows(name, values):
            (folder / name).write_text("".join(json.dumps(v, ensure_ascii=False, sort_keys=True) + "\n" for v in values))

        write_rows("sft_trajectories.jsonl", sft)
        write_rows("sft_turn_supervision.jsonl", turns)
        for split in SPLITS:
            write_rows(split + ".jsonl", by_split[split])
        write_rows("scenario_counts.jsonl", scenario_summary)
        if pinned is not None:
            for path in folder.glob('*.jsonl'):
                if sha(path) != manifest['files'][path.name]:
                    raise ValueError('replayed rows differ from frozen split: ' + path.name)
        write_json("summary.json", summary)
        write_json("source_hashes.json", source_hashes)
        (folder / "prepare_training_split.py").write_bytes(Path(__file__).read_bytes())
        write_json("manifest.json", {"schema_version": "awm-training-split-v1", "seed": seed,
                                    "source_root": str(root),
                                    "files": {p.name: sha(p) for p in sorted(folder.iterdir())}})
        folder.rename(output)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument('--replay-frozen', action='store_true', help='Relocate the archived input paths and verify regenerated rows')
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.output, args.seed, args.replay_frozen), ensure_ascii=False, indent=2))
