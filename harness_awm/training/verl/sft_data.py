"""Preserve frozen AWM input IDs and labels in verl's SFT loss-mask format."""
import math
from pathlib import Path

from ...io import read_json
from ..data import rows, sha


def verified_rows(data, model, max_length):
    data, model = Path(data), Path(model)
    report = read_json(data / "report.json")
    if sha(data / "train.jsonl") != report["train_sha256"]:
        raise ValueError("training data checksum mismatch")
    for name, expected in report["tokenizer_files"].items():
        if name in {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}:
            if sha(model / name) != expected:
                raise ValueError("tokenizer differs from frozen training data")
    samples = list(rows(data / "train.jsonl"))
    if not samples:
        raise ValueError("empty SFT dataset")
    for sample in samples:
        ids, labels = sample["input_ids"], sample["labels"]
        if (len(ids) < 2 or len(ids) != len(labels) or labels[0] != -100
                or not any(x != -100 for x in labels[1:])
                or any(type(t) is not int or t < 0 for t in ids)
                or any(type(label) is not int or label not in {-100, token} for token, label in zip(ids, labels))):
            raise ValueError("invalid current-response SFT labels")
        if len(ids) > max_length:
            raise ValueError("sequence exceeds max_length; truncation and dropping are prohibited")
    return samples, report


class EncodedSFTDataset:
    def __init__(self, samples, pad_token_id):
        self.samples, self.pad_token_id = samples, pad_token_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import torch
        if index == -1:
            # DP/batch alignment only; no supervised token and no sample duplication.
            ids, labels = [self.pad_token_id, self.pad_token_id], [-100, -100]
        else:
            sample = self.samples[index]
            ids, labels = sample["input_ids"], sample["labels"]
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "position_ids": torch.arange(len(ids)),
                "loss_mask": torch.tensor([int(label != -100) for label in labels], dtype=torch.long)}


class LengthGroupedSampler:
    """Every real sample once per epoch; masked rows complete the last DP batch."""
    def __init__(self, lengths, batch_size, rank=0, world_size=1, seed=20261106):
        if batch_size < 1 or world_size < 1 or batch_size % world_size or not 0 <= rank < world_size:
            raise ValueError("global batch must be divisible by DP world size")
        self.lengths, self.batch_size = lengths, batch_size
        self.rank, self.world_size, self.seed, self.epoch = rank, world_size, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batch_size) * self.batch_size // self.world_size

    def __iter__(self):
        import torch
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.lengths), generator=generator).tolist()
        window = self.batch_size * 50
        ordered = []
        for start in range(0, len(indices), window):
            ordered.extend(sorted(indices[start:start+window], key=lambda i: self.lengths[i], reverse=True))
        ordered.extend([-1] * (len(self) * self.world_size - len(ordered)))
        return iter(ordered[self.rank::self.world_size])
