"""Verify frozen manifests and materialize current-response-only SFT labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def verify_split(split):
    split = Path(split)
    manifest = json.loads((split / "manifest.json").read_text())
    if manifest["schema_version"] != "awm-training-split-v1":
        raise ValueError("unsupported split schema")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or sha(split / name) != expected:
            raise ValueError(f"split file hash mismatch: {name}")
    seen, counts = set(), {}
    for name in ("sft_trajectories", "grpo_train", "validation", "test"):
        items = list(rows(split / f"{name}.jsonl"))
        ids = {r["task_id"] for r in items}
        expected_split = "sft_train" if name == "sft_trajectories" else name
        if len(ids) != len(items) or ids & seen or any(r["split"] != expected_split for r in items):
            raise ValueError(f"duplicate tasks, overlap or wrong split: {name}")
        seen |= ids
        counts[name] = len(items)
    return manifest, counts


def resolve_ref(ref, source_root, root):
    # Absolute paths in frozen manifests remain unchanged when transferring the repo.
    relative = Path(ref).relative_to(source_root)
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("source reference escapes repository")
    return path


def encode_turn(tokenizer, messages, response):
    # No reasoning traces are present in the single-JSON AWM action targets.
    kwargs = dict(tokenize=True, return_dict=False, enable_thinking=False)
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, **kwargs)
    full = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": response}],
        add_generation_prompt=False, **kwargs,
    )
    if full[:len(prompt)] != prompt or len(full) <= len(prompt):
        raise ValueError("chat template does not preserve the generation prefix")
    return {"input_ids": full, "labels": [-100] * len(prompt) + full[len(prompt):]}


class ResponseCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch
        width = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "labels": [], "attention_mask": []}
        for f in features:
            ids, labels = f["input_ids"], f["labels"]
            if len(ids) != len(labels) or not any(x != -100 for x in labels[1:]):
                raise ValueError("invalid labels or empty shifted loss target")
            n = width - len(ids)
            batch["input_ids"].append(ids + [self.pad_token_id] * n)
            batch["labels"].append(labels + [-100] * n)
            batch["attention_mask"].append([1] * len(ids) + [0] * n)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def prepare(split, root, tokenizer_path, output):
    from transformers import AutoTokenizer
    split, root, output = Path(split), Path(root), Path(output)
    if output.exists():
        raise ValueError("output exists; choose a new directory")
    manifest, counts = verify_split(split)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
    tokenizer_files = {p.name: sha(p) for p in Path(tokenizer_path).iterdir()
                       if p.name in {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}}
    trajectories = {r["trajectory_id"]: r for r in rows(split / "sft_trajectories.jsonl")}
    lengths, response_lengths, seen, observed = [], [], set(), Counter()
    skipped = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sft-", dir=output.parent) as temp:
        staged = Path(temp) / "bundle"
        staged.mkdir()
        with (staged / "train.jsonl").open("w") as stream:
            for row in rows(split / "sft_turn_supervision.jsonl"):
                ident, turn = row["trajectory_id"], row["turn_id"]
                trajectory = trajectories[ident]
                supervise = turn in trajectory["supervised_turn_ids"]
                if (ident, turn) in seen or row["task_id"] != trajectory["task_id"]:
                    raise ValueError("duplicate turn or incorrect task binding")
                seen.add((ident, turn))
                observed[ident] += 1
                if row["supervise_current_response"] is not supervise or row["response_loss_weight"] != int(supervise):
                    raise ValueError("inconsistent supervision mask")
                sources = {}
                for kind in ("input", "output"):
                    path = resolve_ref(row[kind + "_ref"], manifest["source_root"], root)
                    if sha(path) != row[kind + "_sha256"]:
                        raise ValueError(f"source hash mismatch: {path}")
                    sources[kind] = json.loads(path.read_text())
                if not supervise:
                    # Later turns retain this action in their original, unmodified input.
                    skipped += 1
                    continue
                out = sources["output"]
                if out["test_only"] or out["finish_reason"] != "stop" or not out["content"].strip():
                    raise ValueError("invalid supervised response")
                encoded = encode_turn(tokenizer, sources["input"]["messages"], out["content"])
                lengths.append(len(encoded["input_ids"]))
                response_lengths.append(sum(x != -100 for x in encoded["labels"]))
                stream.write(json.dumps({**encoded, "trajectory_id": ident, "turn_id": turn}) + "\n")
        for ident, trajectory in trajectories.items():
            expected = set(range(1, trajectory["turn_count"] + 1))
            if {t for i, t in seen if i == ident} != expected:
                raise ValueError(f"missing source turns: {ident}")
        if not lengths:
            raise ValueError("no supervised samples")
        lengths.sort()
        report = {
            "split_counts": counts, "source_turns": len(seen), "masked_turns": skipped,
            "supervised_turns": len(lengths), "total_tokens": sum(lengths),
            "target_tokens": sum(response_lengths),
            "lengths": {"min": lengths[0], "p50": lengths[len(lengths)//2],
                        "p95": lengths[int((len(lengths)-1)*0.95)], "max": lengths[-1]},
            "over_length": {str(n): sum(x > n for x in lengths) for n in (4096,8192,16384,32768)},
            "truncated": 0, "enable_thinking": False,
            "loss_policy": "current_response_only; excluded actions retained in later history",
            "split_manifest_sha256": sha(split / "manifest.json"),
            "tokenizer_files": tokenizer_files,
            "train_sha256": sha(staged / "train.jsonl"),
            "formal_grpo_ready": False,
        }
        if any(sha(Path(tokenizer_path) / name) != digest for name, digest in tokenizer_files.items()):
            raise ValueError("tokenizer changed during preparation")
        (staged / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        staged.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.split, args.root, args.tokenizer, args.output), indent=2))


if __name__ == "__main__":
    main()
