"""CPU check of the actual pretrained 2B text model; discard diagnostic updates."""
import argparse
import json
import math
import os
from pathlib import Path


def run(model_path, output):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["SWANLAB_MODE"] = "disabled"
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    from trl import SFTConfig, SFTTrainer
    from .data import encode_turn, ResponseCollator
    from .tracking import SwanLabCallback
    from .doctor import check

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    environment = check(model=model_path)
    if not environment["passed"]:
        raise ValueError(environment["errors"])
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False, dtype=torch.float32,
        attn_implementation="sdpa", output_loading_info=True,
    )
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise ValueError(f"checkpoint did not load completely: {loading}")
    parameter_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameter_count != trainable_count:
        raise ValueError("some text parameters are frozen")
    name, probe = next((n, p) for n, p in model.named_parameters() if 'linear_attn.out_proj.weight' in n)
    before = probe.detach().clone()
    sample = encode_turn(tokenizer, [{"role": "user", "content": "Reply with a JSON object."}], '{"ok":true}')
    tracking = SwanLabCallback()
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer, callbacks=[tracking],
        train_dataset=Dataset.from_list([sample]), data_collator=ResponseCollator(tokenizer.pad_token_id),
        args=SFTConfig(output_dir=str(output / "trainer"), use_cpu=True, bf16=False, fp16=False,
                       per_device_train_batch_size=1, max_steps=1, learning_rate=1e-5,
                       optim="adamw_torch", gradient_checkpointing=False,
                       dataset_kwargs={"skip_prepare_dataset": True}, save_strategy="no",
                       report_to="none", dataloader_pin_memory=False, disable_tqdm=True, logging_steps=1),
    )
    try:
        result = trainer.train()
    finally:
        tracking.finish()
    report = {"test_only": True, "source_checkpoint_unchanged": True,
              "pretrained_weights_loaded": True, "missing_keys": sorted(loading.get("missing_keys", [])),
              "unexpected_keys": sorted(loading.get("unexpected_keys", [])),
              "text_parameters": parameter_count, "trainable_parameters": trainable_count,
              "input_tokens": len(sample["input_ids"]), "steps": trainer.state.global_step,
              "loss": result.training_loss, "probe_parameter": name,
              "weights_changed": not torch.equal(before, probe.detach()), "gpu_tested": False}
    report["passed"] = report["weights_changed"] and math.isfinite(result.training_loss)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise RuntimeError(report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.model, a.output), indent=2))
