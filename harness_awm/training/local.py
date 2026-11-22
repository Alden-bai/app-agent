"""Shared offline checkpoint loading for local training and evaluation."""
from pathlib import Path

from ..io import digest, file_hash


def checkpoint_files(path):
    return sorted(p for p in Path(path).resolve().iterdir() if p.is_file() and (
        p.suffix == ".safetensors" or p.name.startswith("pytorch_model") or
        p.name in {"config.json", "generation_config.json", "tokenizer.json",
                   "tokenizer_config.json", "chat_template.jinja"}))


def checkpoint_filesystem(path):
    """Guard a previously hashed checkpoint against writes, replacements and new files."""
    records = {}
    for p in checkpoint_files(path):
        s = p.stat()
        records[p.name] = {"device": s.st_dev, "inode": s.st_ino, "size": s.st_size,
                           "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns}
    return {"path": str(Path(path).resolve()), "files": records}


def checkpoint_identity(path):
    path = Path(path).resolve()
    files = checkpoint_files(path)
    if not any(p.suffix == ".safetensors" or p.name.endswith(".bin") for p in files):
        raise ValueError("checkpoint has no local model weights")
    hashes = {p.name: file_hash(p) for p in sorted(files)}
    return {"path": str(path), "sha256": digest(hashes), "files": hashes}


def load_checkpoint(path, *, device="cuda", dtype="float32", trainable=False):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForCausalLM

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; use --device cpu only for explicit CPU execution")
    if dtype not in {"float32", "bfloat16"}:
        raise ValueError("unsupported checkpoint dtype")
    path = Path(path).resolve()
    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    cls = Qwen3_5ForCausalLM if config.model_type in {"qwen3_5", "qwen3_5_text"} else AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model, loading = cls.from_pretrained(path, local_files_only=True, trust_remote_code=False,
        dtype=getattr(torch, dtype), attn_implementation="sdpa", output_loading_info=True)
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise ValueError("incomplete local checkpoint: " + str(loading))
    model.to(device).requires_grad_(trainable).eval()
    return model, tokenizer
