"""Bounded TRL GRPO diagnostic on frozen SFT contexts, with inference benchmarks.

The reference-action reward is diagnostic only. This entrypoint does not issue
Judge requests, admit business rewards, or execute generated tools.
"""
import argparse
import inspect
import json
import math
import os
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median
import time

from ..io import read_json, write_json, file_hash
from .data import verify_split, rows, resolve_ref


KERNEL_NAMES = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
                "causal_conv1d_fn", "causal_conv1d_update")


def diagnostic_reward(completion, reference):
    """JSON structure + canonical reference similarity, not task success."""
    expected = json.loads(reference)
    try:
        action = json.loads(completion)
    except (ValueError, TypeError):
        return 0.0
    if not isinstance(action, dict) or not isinstance(action.get("type"), str):
        return 0.0
    canonical = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    similarity = SequenceMatcher(None, canonical(expected), canonical(action), autojunk=False).ratio()
    return 0.25 + 0.25 * (action["type"] == expected["type"]) + 0.5 * similarity


def select_examples(root, split, tokenizer, count, max_prompt_tokens):
    manifest, counts = verify_split(split)
    trajectories = {r["trajectory_id"]: r for r in rows(split / "sft_trajectories.jsonl")}
    selected, scenarios = [], set()
    for row in rows(split / "sft_turn_supervision.jsonl"):
        trajectory = trajectories[row["trajectory_id"]]
        if not row["supervise_current_response"] or trajectory["scenario"] in scenarios:
            continue
        source = {}
        for kind in ("input", "output"):
            path = resolve_ref(row[kind + "_ref"], manifest["source_root"], root)
            if file_hash(path) != row[kind + "_sha256"]:
                raise ValueError("source_hash_mismatch")
            source[kind] = read_json(path)
        target = source["output"]["content"]
        # Query actions provide a meaningful range of reference similarities.
        # Selection depends only on frozen training data, never sampled rewards.
        if json.loads(target).get("type") != "search_tools":
            continue
        messages = source["input"]["messages"]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                            add_generation_prompt=True, enable_thinking=False)
        if not 128 <= len(ids) <= max_prompt_tokens:
            continue
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        # Use standard text prompts so TRL cannot template a second time.
        if tokenizer(prompt, add_special_tokens=False)["input_ids"] != ids:
            raise ValueError("prompt_round_trip_mismatch")
        selected.append({"prompt": prompt, "reference": target, "task_id": row["task_id"],
                         "trajectory_id": row["trajectory_id"], "turn_id": row["turn_id"],
                         "split": "sft_train", "prompt_tokens": len(ids),
                         "input_sha256": row["input_sha256"], "output_sha256": row["output_sha256"]})
        scenarios.add(trajectory["scenario"])
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("not_enough_bounded_training_examples")
    return selected, {"split_manifest_sha256": file_hash(split / "manifest.json"),
                      "split_counts": counts, "selection": "first search_tools action per distinct training scenario",
                      "truncated_prompts": 0, "validation_or_test_used": False}


def kernel_bindings(module):
    bindings = {}
    for name in KERNEL_NAMES:
        cells = inspect.getclosurevars(getattr(module, name)).nonlocals
        bindings[name] = {"fast": cells.get("is_new_implementation", False),
                          "implementation": getattr(cells.get("implementation"), "__module__", None)}
    return bindings


def benchmark(model, tokenizer, examples, output, *, group_size, tokens, repetitions):
    import torch
    from transformers import GenerationConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    fast = {name: getattr(q, name) for name in KERNEL_NAMES}
    reference = {name: inspect.unwrap(fn) for name, fn in fast.items()}
    modes = ("reference_serial", "fast_serial", "fast_batch")
    prompts = [example["prompt"] for example in examples[:2] for _ in range(group_size)]
    model.eval()
    generation = GenerationConfig(max_new_tokens=tokens, min_new_tokens=tokens,
        do_sample=False, eos_token_id=None, pad_token_id=tokenizer.pad_token_id,
        use_cache=True, disable_compile=True)
    records = []

    def run(mode, repetition, warmup=False):
        for name, fn in (reference if mode == "reference_serial" else fast).items():
            setattr(q, name, fn)
        # Match training: one prompt's four candidates in a batch.
        width = group_size if mode == "fast_batch" else 1
        batches = [tokenizer(prompts[i:i + width], padding=True, add_special_tokens=False,
                             return_tensors="pt").to(model.device)
                   for i in range(0, len(prompts), width)]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started, completions = time.perf_counter(), []
        with torch.inference_mode():
            for batch in batches:
                ids = model.generate(**batch, generation_config=generation, logits_to_keep=1)
                completions.extend(ids[:, batch["input_ids"].shape[1]:].tolist())
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        row = {"mode": mode, "repetition": repetition, "warmup": warmup, "seconds": elapsed,
               "tokens": sum(map(len, completions)), "tokens_per_second": sum(map(len, completions)) / elapsed,
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
               "completion_ids": completions}
        records.append(row)
        write_json(output / "benchmark_measurements.json", records)
        print(json.dumps({k: v for k, v in row.items() if k != "completion_ids"}), flush=True)

    try:
        for mode in modes:
            run(mode, -1, True)
        for repetition in range(repetitions):
            for mode in (modes if repetition % 2 == 0 else tuple(reversed(modes))):
                run(mode, repetition)
    finally:
        for name, fn in fast.items():
            setattr(q, name, fn)
    times = {mode: median(r["seconds"] for r in records if r["mode"] == mode and not r["warmup"])
             for mode in modes}
    outputs = {mode: [r["completion_ids"] for r in records if r["mode"] == mode and not r["warmup"]]
               for mode in modes}
    result = {"median_seconds": times, "kernel_speedup": times[modes[0]] / times[modes[1]],
              "batch_speedup": times[modes[1]] / times[modes[2]],
              "combined_speedup": times[modes[0]] / times[modes[2]],
              "identical_greedy_tokens": outputs[modes[0]] == outputs[modes[1]] == outputs[modes[2]],
              "sequences": len(prompts), "tokens_per_sequence": tokens, "repetitions": repetitions,
              "prompt_lengths": [x["prompt_tokens"] for x in examples[:2]],
              "scope": "fixed-token generation only; warmup excluded; EOS disabled equally in all benchmark modes",
              "not_an_end_to_end_training_speedup": True}
    write_json(output / "benchmark_summary.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", type=Path, default=Path("data/training_split_20261106_v1"))
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-prompt-tokens", type=int, default=3072)
    p.add_argument("--max-completion-tokens", type=int, default=192)
    p.add_argument("--benchmark-tokens", type=int, default=48)
    p.add_argument("--benchmark-repetitions", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--skip-benchmark", action="store_true")
    p.add_argument("--save-model", action="store_true")
    a = p.parse_args()
    if not 1 <= a.steps <= 10 or not 2 <= a.group_size <= 8:
        p.error("diagnostic is bounded to 1..10 steps and groups of 2..8")
    if min(a.max_prompt_tokens, a.max_completion_tokens, a.benchmark_tokens, a.benchmark_repetitions) < 1:
        p.error("token budgets and repetitions must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        p.error("this diagnostic supports one process")
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      HF_HUB_DISABLE_TELEMETRY="1")
    import importlib.metadata as metadata
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, TrainerCallback, set_seed
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    from trl import GRPOConfig, GRPOTrainer
    from .local import load_checkpoint, checkpoint_identity
    from .actor import stop_token_ids
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this smoke")
    torch.set_num_threads(4)
    set_seed(20261110)
    bindings = kernel_bindings(q)
    if not all(value["fast"] for value in bindings.values()):
        raise RuntimeError("FLA and causal-conv1d must both be active; see runtime PYTHONPATH")
    identity = checkpoint_identity(a.model)
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    tokenizer.padding_side = "left"
    examples, provenance = select_examples(Path.cwd(), a.split.resolve(), tokenizer,
                                           max(3, a.steps), a.max_prompt_tokens)
    write_json(a.output / "examples.json", examples)
    model, tokenizer = load_checkpoint(a.model, device="cuda", dtype="bfloat16", trainable=True)
    tokenizer.padding_side = "left"
    model.generation_config.eos_token_id = stop_token_ids(model, tokenizer)
    runtime = {"framework": "trl.GRPOTrainer", "diagnostic_only": True, "formal_grpo_ready": False,
               "model": identity, "gpu": torch.cuda.get_device_name(), "bindings": bindings,
               "versions": {name: metadata.version(name) for name in
                   ("torch", "transformers", "trl", "accelerate", "datasets", "flash-linear-attention", "causal-conv1d")},
               "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "parameter_dtype": str(next(model.parameters()).dtype), "attention": "sdpa",
               "data": provenance, "judge_requests": 0, "environment_rollouts": 0,
               "reward": "diagnostic JSON/reference-action similarity; not business success",
               "source_sha256": file_hash(__file__), "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}}
    write_json(a.output / "runtime.json", runtime)
    bench = None if a.skip_benchmark else benchmark(model, tokenizer, examples, a.output,
        group_size=a.group_size, tokens=a.benchmark_tokens, repetitions=a.benchmark_repetitions)

    reward_batches, generation_times = [], []

    def reference_action_reward(completions, reference, task_id, **kwargs):
        values = [diagnostic_reward(c, r) for c, r in zip(completions, reference, strict=True)]
        reward_batches.append({"task_ids": task_id, "completions": completions, "rewards": values})
        write_json(a.output / "sampled_rewards.json", reward_batches)
        return values

    class TimingTrainer(GRPOTrainer):
        def _generate_single_turn(self, *args, **kwargs):
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = super()._generate_single_turn(*args, **kwargs)
            torch.cuda.synchronize()
            generation_times.append({"step": self.state.global_step, "seconds": time.perf_counter() - started,
                                     "completion_tokens": sum(map(len, result[0])), "sequences": len(result[0])})
            write_json(a.output / "generation_times.json", generation_times)
            return result

    class AuditCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            write_json(a.output / "training_logs.json", state.log_history)

    args = GRPOConfig(output_dir=str(a.output / "trainer"), max_steps=a.steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=a.group_size,
        num_generations=a.group_size, generation_batch_size=a.group_size,
        max_completion_length=a.max_completion_tokens, temperature=1.0, top_p=1.0, top_k=0,
        beta=0.0, loss_type="grpo", scale_rewards="group", num_iterations=1,
        learning_rate=a.learning_rate, lr_scheduler_type="constant", weight_decay=0.0,
        bf16=True, tf32=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused", logging_steps=1, save_strategy="no", report_to="none",
        seed=20261110, data_seed=20261110, disable_tqdm=True, use_vllm=False,
        generation_kwargs={"use_cache": True, "logits_to_keep": 1},
        mask_truncated_completions=False, log_completions=False)
    trainer = TimingTrainer(model=model, processing_class=tokenizer, args=args,
        reward_funcs=reference_action_reward, train_dataset=Dataset.from_list(examples), callbacks=[AuditCallback()])
    # A small deterministic sample from every tensor demonstrates actual updates,
    # without retaining a second model in memory. Weight decay is disabled.
    snapshots = {name: param.detach().flatten()[::max(1, param.numel() // 128)].float().cpu().clone()
                 for name, param in model.named_parameters()}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    trained = trainer.train()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    changes = []
    for name, param in model.named_parameters():
        current = param.detach().flatten()[::max(1, param.numel() // 128)].float().cpu()
        delta = (current - snapshots[name]).abs()
        if not torch.isfinite(current).all():
            raise RuntimeError("nonfinite_parameter_sample")
        if delta.max() > 0:
            changes.append({"name": name, "max_sampled_delta": float(delta.max())})
    grads = [r["grad_norm"] for r in trainer.state.log_history if "grad_norm" in r]
    result = {"status": "passed", "diagnostic_only": True, "optimizer_steps": trainer.state.global_step,
              "train_seconds": elapsed, "generation_seconds": sum(r["seconds"] for r in generation_times),
              "generation_tokens": sum(r["completion_tokens"] for r in generation_times),
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
              "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
              "changed_parameter_tensors_sampled": len(changes), "parameter_changes": changes,
              "grad_norms": grads, "training_metrics": trained.metrics, "benchmark": bench,
              "reward_batches": len(reward_batches), "judge_requests": 0,
              "limitations": ["single-turn reference-action diagnostic on SFT training examples",
                              "no multi-turn environment or admitted business rewards",
                              "three-step smoke does not establish quality improvement",
                              "TRL group std uses its default sample estimator, unlike the custom engine's population std"],
              "saved_model": None}
    if (trainer.state.global_step != a.steps or not changes or not grads
            or not all(math.isfinite(g) for g in grads) or not any(g > 0 for g in grads)):
        result["status"] = "failed_no_verified_update"
    if a.save_model and result["status"] == "passed":
        destination = a.output / "diagnostic_model"
        trainer.save_model(str(destination))
        result["saved_model"] = str(destination)
        result["checkpoint_sha256"] = checkpoint_identity(destination)["sha256"]
        result["optimizer_state_saved"] = False
    write_json(a.output / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "parameter_changes"}), flush=True)
    if result["status"] != "passed":
        raise RuntimeError(result["status"])


if __name__ == "__main__":
    main()
