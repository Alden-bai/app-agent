"""Full-parameter text SFT on verified local Qwen3.5 weights and tokenized data."""
import argparse
import json
import os
from pathlib import Path

from .data import ResponseCollator, rows, sha


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    from trl import SFTConfig, SFTTrainer
    from .tracking import SwanLabCallback

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-length", type=int, default=32768)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--microbatch-tokens", type=int, default=0)
    p.add_argument("--save-strategy", choices=("steps", "epoch", "no"), default="steps")
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--skip-final-save", action="store_true")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--resume", type=Path)
    p.add_argument("--lr-stage-start-step", type=int)
    p.add_argument("--lr-stage-end-step", type=int)
    p.add_argument("--lr-stage-warmup-steps", type=int, default=7)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--preflight-longest", action="store_true",
                   help="Use the longest samples first for a short GPU memory check")
    a = p.parse_args()
    callbacks = []
    if a.lr_stage_start_step is not None or a.lr_stage_end_step is not None:
        if a.resume is None or a.lr_stage_start_step is None or a.lr_stage_end_step is None:
            p.error("learning-rate continuation requires --resume and both stage boundaries")
        from .continuation import continuation_callback
        callbacks.append(continuation_callback(a.lr_stage_start_step, a.lr_stage_end_step,
                                               a.lr_stage_warmup_steps))
    if a.batch_size < 1 or a.gradient_accumulation < 1:
        p.error("batch size and gradient accumulation must be positive")
    if a.preflight_longest and a.max_steps < 1:
        p.error("--preflight-longest requires a positive --max-steps")
    if a.output.exists() and any(a.output.iterdir()) and a.resume is None:
        p.error("output is not empty; explicitly resume or choose a new directory")
    if not a.cpu and not torch.cuda.is_available():
        p.error("CUDA unavailable; --cpu is only for explicit local debugging")
    report = json.loads((a.data / "report.json").read_text())
    if sha(a.data / "train.jsonl") != report["train_sha256"]:
        p.error("training data checksum mismatch")
    # A template/tokenizer change invalidates already-materialized labels.
    for name, expected in report["tokenizer_files"].items():
        if name in {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}:
            if sha(a.model / name) != expected:
                p.error("tokenizer differs from prepared training data")
    for row in rows(a.data / "train.jsonl"):
        if len(row["input_ids"]) > a.max_length:
            p.error("sequence exceeds --max-length; no silent truncation or dropping is allowed")
        if len(row["input_ids"]) != len(row["labels"]) or not any(x != -100 for x in row["labels"][1:]):
            p.error("invalid labels")
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=False)
    # This AWM dataset is text-only: train all language-model parameters.
    # The unused vision encoder is omitted, not adapted or used as frozen text weights.
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        a.model, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="sdpa", output_loading_info=True,
    )
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise ValueError(f"incomplete pretrained text checkpoint: {loading}")
    if not all(param.requires_grad for param in model.parameters()):
        raise ValueError("full-parameter training requires every text parameter trainable")
    # SFTTrainer aligns EOS with the response tokenizer; do not restore a stale
    # pretrained generation EOS that ignores the supervised end-of-turn token.
    inference_tokens = {key: getattr(model.config, key, None)
                        for key in ("eos_token_id", "bos_token_id", "pad_token_id")}
    model.config.use_cache = False
    dataset = load_dataset("json", data_files=str(a.data / "train.jsonl"), split="train")
    dataset = dataset.select_columns(["input_ids", "labels"])
    if a.preflight_longest:
        order = sorted(range(len(dataset)), key=lambda i: len(dataset[i]["input_ids"]), reverse=True)
        dataset = dataset.select(order)
    fsdp2 = os.environ.get("ACCELERATE_USE_FSDP", "").lower() == "true" and os.environ.get("FSDP_VERSION") == "2"
    args = SFTConfig(
        output_dir=str(a.output), max_steps=a.max_steps, num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size, gradient_accumulation_steps=a.gradient_accumulation,
        learning_rate=a.learning_rate, optim="adamw_torch_fused" if not a.cpu else "adamw_torch", bf16=not a.cpu, use_cpu=a.cpu,
        gradient_checkpointing=not fsdp2 and not a.no_gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=a.max_length, packing=False, dataset_kwargs={"skip_prepare_dataset": True},
        dataloader_pin_memory=not a.cpu, report_to="none", logging_steps=1,
        logging_nan_inf_filter=False,
        save_strategy=a.save_strategy, save_steps=100, save_total_limit=a.save_total_limit,
        warmup_steps=a.warmup_ratio, lr_scheduler_type="cosine",
        seed=20261106, eval_strategy="no",
        train_sampling_strategy="sequential" if a.preflight_longest else "group_by_length",
    )
    tracking = SwanLabCallback()
    args.microbatch_tokens = a.microbatch_tokens
    args.strict_resume_weights = bool(callbacks)
    from harness_awm.io import write_json
    write_json(a.output / "sft_config.json", {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(a).items()},
        "data_sha256": report["train_sha256"], "initial_special_tokens": inference_tokens,
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "pytorch_alloc_conf": os.environ.get("PYTORCH_ALLOC_CONF"),
        "loss_policy": report.get("loss_policy"), "training_rewards": False,
    })
    failed = True
    try:
        from .adaptive import trainer_class
        trainer = trainer_class()(model=model, args=args, train_dataset=dataset, callbacks=[*callbacks, tracking],
                             microbatch_tokens=a.microbatch_tokens,
                             processing_class=tokenizer, data_collator=ResponseCollator(tokenizer.pad_token_id))
        result = trainer.train(resume_from_checkpoint=str(a.resume) if a.resume else None)
        if not a.skip_final_save:
            trainer.save_model(str(a.output / "final"))
        trainer.save_state()
        trainer.save_metrics("train", result.metrics)
        failed = False
    finally:
        tracking.finish(failed=failed)


if __name__ == "__main__":
    main()
