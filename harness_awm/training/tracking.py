"""Trainer callback: only the global main process owns the SwanLab run."""
from transformers import TrainerCallback
import time
import math
from pathlib import Path

from harness_awm.io import write_json

from harness_awm.tracking import SwanLabTracker


class SwanLabCallback(TrainerCallback):
    def __init__(self):
        self.tracker = None
        self.started = None
        self.tokens = 0

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.started = time.monotonic()
            self.tracker = SwanLabTracker(args.output_dir, config={
                "stage": "sft", "learning_rate": args.learning_rate,
                "epochs": args.num_train_epochs, "max_steps": args.max_steps,
                "batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "seed": args.seed, "bf16": args.bf16,
                "microbatch_tokens": getattr(args, "microbatch_tokens", 0),
                "gradient_checkpointing": getattr(args, "gradient_checkpointing", False),
                "warmup_steps": getattr(args, "warmup_steps", 0),
                "optimizer": str(getattr(args, "optim", "unknown")),
            })
            if self.tracker.url is None or isinstance(self.tracker.url, str):
                write_json(Path(args.output_dir) / "tracking.json", {"url": self.tracker.url})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and self.tracker is not None and logs:
            if any(not math.isfinite(float(logs[key])) for key in ("loss", "grad_norm") if key in logs):
                raise ValueError("nonfinite SFT loss/gradient; stopping before selecting invalid weights")
            self.tracker.log({"train": logs}, step=state.global_step)
            import torch
            if torch.cuda.is_available():
                self.tokens = logs.get("num_tokens", self.tokens)
                gpu = {"allocated_gib": torch.cuda.memory_allocated() / 2**30,
                       "reserved_gib": torch.cuda.memory_reserved() / 2**30,
                       "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                       "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
                self.tracker.log({"gpu": gpu}, step=state.global_step)
                elapsed = time.monotonic() - self.started
                throughput = self.tokens / max(elapsed, 1e-6)
                self.tracker.log({"throughput/input_tokens_per_second": throughput}, step=state.global_step)
                write_json(Path(args.output_dir) / "progress.json", {
                    "step": state.global_step, "max_steps": state.max_steps,
                    "epoch": state.epoch, "train": logs, "gpu": gpu,
                    "elapsed_seconds": elapsed, "input_tokens_per_second": throughput})

    def finish(self, *, failed=False):
        if self.tracker is not None:
            self.tracker.finish(failed=failed)
            self.tracker = None
