"""Paired real-model SFT benchmark; no checkpoint writes or external services.

Uses the installed TRL fused LM head and loss, FP32 weights, BF16 compute,
non-reentrant checkpointing, and the existing training FLA dependency path.
AdamW runs with lr=0 so all A/B trials see identical weights. Timing includes
forward, backward, gradient clipping and optimizer step, but excludes loading,
collation, gradient diagnostics, and per-shape/per-mode warmup.
"""
import argparse
from collections import Counter
import gc
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
from statistics import median
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness_awm.io import write_json, now
from harness_awm.training.data import ResponseCollator, rows, sha


def select_batches(data):
    ordered = sorted(enumerate(data), key=lambda x: len(x[1]["input_ids"]))
    cases = {}
    for name, quantile, size in (("median_batch16", .50, 16), ("p95_batch8", .95, 8),
                                  ("longest_batch1", 1., 1)):
        center = int((len(ordered) - 1) * quantile)
        start = max(0, min(len(ordered) - size, center - size // 2))
        cases[name] = ordered[start:start + size]
    return cases


def sample_indices(size, limit=512):
    """Integer arithmetic avoids rounding a large tensor's last index out of bounds."""
    count = min(limit, size)
    if count < 1:
        raise ValueError("cannot sample an empty gradient")
    return [i * (size - 1) // max(count - 1, 1) for i in range(count)]


def gradient_probe(model):
    import torch
    result = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            result[name] = None
            continue
        grad = p.grad.detach().flatten()
        indices = torch.tensor(sample_indices(grad.numel()), device=grad.device, dtype=torch.long)
        # Full tensor norm checks every gradient for NaN/Inf; sampled entries
        # compare direction and scale without storing 7 GiB of gradients.
        result[name] = {"norm": float(grad.norm()), "sample": grad[indices].float().cpu().tolist()}
    return result


def compare_gradients(reference, candidate):
    import torch
    differences = []
    a, b = [], []
    missing = []
    for name, r in reference.items():
        c = candidate[name]
        if r is None or c is None:
            if r != c:
                missing.append(name)
            continue
        x, y = torch.tensor(r["sample"], dtype=torch.float64), torch.tensor(c["sample"], dtype=torch.float64)
        a.append(x); b.append(y)
        differences.append({"parameter": name,
                            "relative_norm_delta": abs(c["norm"] - r["norm"]) / max(r["norm"], 1e-12)})
    x, y = torch.cat(a), torch.cat(b)
    return {"missing_gradient_mismatches": missing,
            "all_full_gradient_norms_finite": all(
                torch.isfinite(torch.tensor(v["norm"])).item()
                for d in (reference, candidate) for v in d.values() if v is not None),
            "sampled_elements": x.numel(),
            "sampled_relative_l2": float((y-x).norm() / x.norm().clamp_min(1e-12)),
            "sampled_cosine": float(torch.nn.functional.cosine_similarity(x, y, dim=0)),
            "sampled_max_abs": float((y-x).abs().max()),
            "largest_parameter_norm_deltas": sorted(differences, key=lambda d: d["relative_norm_delta"], reverse=True)[:8],
            "scope": "512 evenly spaced entries per parameter plus full gradient norms; not all-element equality"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path, default=ROOT / "models/Qwen3.5-2B")
    p.add_argument("--data", type=Path, default=ROOT / "data/training_qwen35_2b_v1")
    p.add_argument("--mode", choices=("both", "rms", "swiglu"), default="both")
    p.add_argument("--repetitions", type=int, default=4)
    p.add_argument("--cases", nargs="+")
    a = p.parse_args()
    if a.repetitions < 2:
        p.error("at least two repetitions required for alternating order")
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "benchmark_source.py").write_bytes(Path(__file__).read_bytes())
    sys.path[:0] = [str(ROOT / "runs/sft_kernel_deps"), str(ROOT / "runs/liger_benchmark_deps")]
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", SWANLAB_MODE="disabled",
                      TOKENIZERS_PARALLELISM="false", PYTORCH_ALLOC_CONF="expandable_segments:True")
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    from trl import SFTConfig, SFTTrainer
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5

    torch.set_num_threads(4)
    torch.manual_seed(20261106)
    free, total = torch.cuda.mem_get_info()
    if free < 80 * 1024**3:
        raise RuntimeError("Need at least 80 GiB free for isolated full-parameter training benchmark")
    torch.cuda.set_per_process_memory_fraction(88 * 1024**3 / total)
    report = json.loads((a.data / "report.json").read_text())
    if sha(a.data / "train.jsonl") != report["train_sha256"]:
        raise ValueError("training data checksum mismatch")
    for name, digest in report["tokenizer_files"].items():
        if sha(a.model / name) != digest:
            raise ValueError("model tokenizer differs from training data")
    data = list(rows(a.data / "train.jsonl"))
    cases = select_batches(data)
    if a.cases:
        cases = {k: cases[k] for k in a.cases}
    metadata = {"at": now(), "mode": a.mode, "train_sha256": report["train_sha256"],
                "versions": {k: importlib.metadata.version(k) for k in
                             ("torch", "transformers", "trl", "triton", "liger-kernel", "flash-linear-attention")},
                "gpu": torch.cuda.get_device_name(), "initial_free_GiB": free / 1024**3,
                "weights": "float32", "compute": "bfloat16", "gradient_checkpointing": True,
                "optimizer": "fused AdamW, lr=0, full FP32 states; gradient clipping max_norm=1",
                "loss": "TRL fused LM head, current-response-only target mean",
                "cross_entropy_patch": False, "fused_linear_cross_entropy_patch": False,
                "repetitions_per_mode_per_case": a.repetitions,
                "source_sha256": sha(Path(__file__)), "model_file_sha256": {
                    f.name: sha(f) for f in a.model.glob("*.safetensors")},
                "cases": {k: {"row_indices": [i for i, _ in batch],
                              "lengths": [len(r["input_ids"]) for _, r in batch],
                              "input_tokens": sum(len(r["input_ids"]) for _, r in batch),
                              "target_tokens": sum(sum(t != -100 for t in r["labels"][1:]) for _, r in batch)}
                          for k, batch in cases.items()}}
    bindings = {}
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        cells = inspect.getclosurevars(getattr(q, name)).nonlocals
        bindings[name] = {"fast": cells.get("is_new_implementation", False),
                          "module": getattr(cells.get("implementation"), "__module__", None)}
    if not all(v["fast"] for v in bindings.values()):
        raise RuntimeError("Existing training FLA kernels not bound")
    metadata["fla_bindings"] = bindings
    write_json(a.output / "metadata.json", metadata)
    write_json(a.output / "status.json", {"stage": "loading", "at": now()})
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        a.model, dtype=torch.float32, attn_implementation="sdpa", local_files_only=True,
        output_loading_info=True)
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise RuntimeError(str(loading))
    model.config.use_cache = False
    config = SFTConfig(output_dir=str(a.output / "unused_trainer_output"), bf16=True,
                       gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                       per_device_train_batch_size=1, gradient_accumulation_steps=1,
                       max_length=32768, packing=False, dataset_kwargs={"skip_prepare_dataset": True},
                       report_to="none", save_strategy="no", eval_strategy="no", seed=20261106)
    collator = ResponseCollator(tokenizer.pad_token_id)
    trainer = SFTTrainer(model=model, args=config, processing_class=tokenizer, data_collator=collator,
                         train_dataset=Dataset.from_list([{k: v for k, v in data[0].items()
                                                           if k in ("input_ids", "labels")}]))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    original_forward = model.forward
    original_classes = (q.Qwen3_5RMSNorm, q.Qwen3_5MLP)
    original_methods = [(m, m.forward) for m in model.modules()]
    parameter_ids = [id(v) for v in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0, fused=True)

    def set_mode(mode):
        for m, forward in original_methods:
            m.forward = forward
        q.Qwen3_5RMSNorm, q.Qwen3_5MLP = original_classes
        if mode == "liger":
            apply_liger_kernel_to_qwen3_5(model=model, rope=False, cross_entropy=False,
                                        fused_linear_cross_entropy=False,
                                        rms_norm=a.mode in ("both", "rms"), swiglu=a.mode in ("both", "swiglu"))
        if model.forward != original_forward or [id(v) for v in model.parameters()] != parameter_ids:
            raise RuntimeError("Layer patch unexpectedly replaced loss forward or parameters")
        return dict(Counter(m.forward.__module__ for m in model.modules()
                            if m.forward.__module__.startswith("liger_kernel")))

    measurements, checks = [], {}
    for case, samples in cases.items():
        inputs = {k: v.to("cuda") for k, v in collator([r for _, r in samples]).items()}
        n_targets = (inputs["labels"][:, 1:] != -100).sum()
        probes, losses = {}, {}
        for mode in ("baseline", "baseline_repeat", "liger"):
            bindings = set_mode("baseline" if mode == "baseline_repeat" else mode)
            if mode == "liger" and not bindings:
                raise RuntimeError("Liger layer kernels not installed")
            write_json(a.output / "status.json", {"stage": "warmup_and_gradient_check", "case": case,
                                                   "mode": mode, "bindings": bindings, "at": now()})
            print(json.dumps({"stage": "warmup", "case": case, "mode": mode, "bindings": bindings}), flush=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = trainer.compute_loss(model, dict(inputs), num_items_in_batch=n_targets)
            loss.backward()
            losses[mode] = float(loss.detach())
            probes[mode] = gradient_probe(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
        check = compare_gradients(probes["baseline"], probes["liger"])
        check.update(losses=losses, loss_abs_delta=abs(losses["liger"] - losses["baseline"]))
        check["baseline_repeat"] = compare_gradients(probes["baseline"], probes["baseline_repeat"])
        check["baseline_repeat"]["loss_abs_delta"] = abs(losses["baseline_repeat"] - losses["baseline"])
        check["screening_thresholds"] = {"sampled_relative_l2_max": .05, "sampled_cosine_min": .999,
                                         "loss_abs_delta_max": .01}
        check["screening_passed"] = (check["all_full_gradient_norms_finite"]
                                     and not check["missing_gradient_mismatches"]
                                     and check["sampled_relative_l2"] <= .05
                                     and check["sampled_cosine"] >= .999 and check["loss_abs_delta"] <= .01)
        checks[case] = check
        write_json(a.output / "numerical_checks.json", checks)
        # Finite numerical differences remain visible even when a mode is timed;
        # speed does not override failed screening or authorize production use.
        if not check["all_full_gradient_norms_finite"] or check["missing_gradient_mismatches"]:
            raise RuntimeError("Nonfinite or missing gradients; see numerical_checks.json")
        optimizer.zero_grad(set_to_none=True)
        del probes
        gc.collect()
        torch.cuda.empty_cache()
        for repetition in range(a.repetitions):
            for mode in (("baseline", "liger") if repetition % 2 == 0 else ("liger", "baseline")):
                set_mode(mode)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = trainer.compute_loss(model, dict(inputs), num_items_in_batch=n_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                item = {"case": case, "mode": mode, "repetition": repetition, "seconds": seconds,
                        "loss": float(loss.detach()), "input_tokens_per_second": metadata["cases"][case]["input_tokens"] / seconds,
                        "target_tokens_per_second": metadata["cases"][case]["target_tokens"] / seconds,
                        "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3,
                        "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 1024**3}
                measurements.append(item)
                write_json(a.output / "measurements.json", measurements)
                write_json(a.output / "status.json", {"stage": "timing", "at": now(), **item})
                print(json.dumps(item), flush=True)
        optimizer.zero_grad(set_to_none=True)
        del inputs
        torch.cuda.empty_cache()
    summaries = {}
    for case in cases:
        by_mode = {mode: [r for r in measurements if r["case"] == case and r["mode"] == mode]
                   for mode in ("baseline", "liger")}
        stats = {mode: {"median_seconds": median(r["seconds"] for r in records),
                        "median_input_tokens_per_second": median(r["input_tokens_per_second"] for r in records),
                        "peak_allocated_GiB": max(r["peak_allocated_GiB"] for r in records),
                        "seconds": [r["seconds"] for r in records]} for mode, records in by_mode.items()}
        stats["speedup"] = stats["baseline"]["median_seconds"] / stats["liger"]["median_seconds"]
        summaries[case] = stats
    write_json(a.output / "summary.json", {"at": now(), "cases": summaries, "numerical_checks": checks,
                "limitations": ["Fixed initial weights, AdamW lr=0; not a convergence test",
                                "Real training microbatches, not the full batch64 accumulated update or an epoch",
                                "Excludes data loading, checkpoints, rollout, Judge, and compilation warmup",
                                "Sampled gradient agreement is a screening test, not a proof of equivalence"],
                "formal_training_changed": False, "judge_requests": 0})
    write_json(a.output / "status.json", {"stage": "finished", "at": now()})


if __name__ == "__main__":
    main()
