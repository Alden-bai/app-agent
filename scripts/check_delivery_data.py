"""Audit frozen splits, source closure, runtime assets and response masks on CPU."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check(root):
    from harness_awm.training.data import verify_split, resolve_ref, sha, rows
    from harness_awm.training.tasks import load_tasks
    split = root / "data/training_split_20261106_v1"
    manifest, counts = verify_split(split)
    assert counts == {"sft_trajectories": 758, "grpo_train": 2746, "validation": 589, "test": 588}
    source_hashes = json.loads((split / "source_hashes.json").read_text())
    for ref, expected in source_hashes.items():
        if sha(resolve_ref(ref, manifest["source_root"], root)) != expected:
            raise ValueError("changed source: " + ref)
    runtime = root / "data/training_runtime_20261106_v1"
    ids = set()
    experiment = {}
    for name in ("grpo_train", "validation", "test"):
        tasks, _, _ = load_tasks(runtime, name)
        native = {f"{t.scenario}:{t.native_task_id}" for t in tasks}
        frozen = {r["task_id"] for r in rows(split / (name + ".jsonl"))}
        if len(tasks) != counts[name] or native != frozen or ids.intersection(native):
            raise ValueError("runtime tasks differ from frozen split: " + name)
        ids.update(native)
        if name == "grpo_train" and (root / "delivery/nioflow/grpo-2000.json").is_file():
            from harness_awm.io import digest
            from harness_awm.training.verl.coverage import CANDIDATES, SAVE_EVERY, selected_task_ids
            seed = json.loads((root / "delivery/run-config.example.json").read_text())["seed"]
            selected = selected_task_ids((t.task_instance_id for t in tasks), seed)
            experiment["grpo_experiment"] = {"seed": seed, "selected_tasks": len(selected),
                "selected_task_ids_sha256": digest(selected), "candidates_per_task": CANDIDATES,
                "trajectories": len(selected) * CANDIDATES, "checkpoint_after_tasks": list(range(SAVE_EVERY, len(selected) + 1, SAVE_EVERY))}
    data = root / "data/training_qwen35_27b_v1"
    report = json.loads((data / "report.json").read_text())
    if sha(data / "train.jsonl") != report["train_sha256"]:
        raise ValueError("encoded data checksum mismatch")
    for name, expected in report["tokenizer_files"].items():
        if sha(root / "assets/qwen35-27b" / name) != expected:
            raise ValueError("27B tokenizer identity mismatch")
    samples, supervised, longest = 0, 0, 0
    for row in rows(data / "train.jsonl"):
        tokens, labels = row["input_ids"], row["labels"]
        if len(tokens) != len(labels) or labels[0] != -100 or not any(x != -100 for x in labels):
            raise ValueError("invalid response mask")
        if any(label != -100 and label != token for label, token in zip(labels, tokens)):
            raise ValueError("label does not match input token")
        samples += 1
        supervised += sum(x != -100 for x in labels)
        longest = max(longest, len(tokens))
    if samples != 6714 or supervised != report["target_tokens"] or longest > 32768:
        raise ValueError("encoded sample coverage differs")
    return {"passed": True, "split_counts": counts, **experiment, "source_files": len(source_hashes),
            "sft_trajectories": 758, "sft_turn_samples": samples, "masked_turns": report["masked_turns"],
            "max_tokens": longest, "weights_loaded": False, "judge_requests": 0}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    result = check(a.root.resolve())
    if a.output:
        a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
