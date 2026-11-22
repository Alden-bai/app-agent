"""Network-free CPU optimizer smoke checks; synthetic data, no business rewards."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path


def run(output):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from datasets import Dataset
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3_5TextConfig, Qwen3_5ForCausalLM
    from trl import SFTConfig, SFTTrainer, GRPOConfig, GRPOTrainer
    from .data import ResponseCollator, encode_turn

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(20261106)
    vocab = {s: i for i, s in enumerate(["<pad>", "<eos>", "<unk>", "system", "user", "assistant", "check", "done", "fail", "tool", "result", ":"])}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + ' ' }}{% if m['role'] == 'assistant' %}{{ eos_token + ' ' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"

    def model():
        config = Qwen3_5TextConfig(
            vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=16, linear_key_head_dim=16, linear_value_head_dim=16,
            linear_num_key_heads=2, linear_num_value_heads=2,
            layer_types=["linear_attention", "full_attention"],
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                             "partial_rotary_factor": 1.0, "mrope_section": [2, 3, 3]},
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            tie_word_embeddings=True, use_cache=False,
        )
        return Qwen3_5ForCausalLM(config)

    def changed(before, current):
        return any(not torch.equal(before[name], p.detach()) for name, p in current.named_parameters())

    common = dict(use_cpu=True, bf16=False, fp16=False, max_steps=1, learning_rate=1e-3,
                  optim="adamw_torch", report_to="none", save_strategy="no", disable_tqdm=True,
                  dataloader_pin_memory=False, logging_steps=1, seed=20261106)
    sample = encode_turn(tokenizer, [{"role": "user", "content": "check"}], "done")
    lm = model()
    before = {n: p.detach().clone() for n, p in lm.named_parameters()}
    sft = SFTTrainer(model=lm, processing_class=tokenizer,
        args=SFTConfig(output_dir=str(output / "sft"), per_device_train_batch_size=1,
                       gradient_checkpointing=False, dataset_kwargs={"skip_prepare_dataset": True}, **common),
        train_dataset=Dataset.from_list([sample, sample]), data_collator=ResponseCollator(tokenizer.pad_token_id))
    sft_result = sft.train()
    sft_changed = changed(before, lm)
    sft.save_model(str(output / "checkpoint"))
    restored = Qwen3_5ForCausalLM.from_pretrained(output / "checkpoint", local_files_only=True)
    restored_ok = all(torch.equal(p.detach(), dict(restored.named_parameters())[n].detach()) for n, p in lm.named_parameters())

    lm = model()
    before = {n: p.detach().clone() for n, p in lm.named_parameters()}
    # A deterministic diagnostic signal tests nonzero group-relative gradients.
    # It is deliberately NOT the user's terminal task reward and is never exported.
    def diagnostic_reward(completions, **kwargs):
        return [float(i % 2) for i in range(len(completions))]
    grpo = GRPOTrainer(model=lm, processing_class=tokenizer, reward_funcs=diagnostic_reward,
        args=GRPOConfig(output_dir=str(output / "grpo"), per_device_train_batch_size=2,
                        num_generations=2, max_completion_length=4, beta=0.0,
                        gradient_checkpointing=False, use_vllm=False, **common),
        train_dataset=Dataset.from_list([{"prompt": "user: check assistant:"}] * 4))
    grpo_result = grpo.train()
    grpo_changed = changed(before, lm)
    report = {
        "test_only": True, "device": "cpu", "architecture": "tiny Qwen3.5 text: linear + full attention",
        "pretrained_2b_weights_tested": False, "gpu_tested": False, "live_harness_grpo_tested": False,
        "sft": {"steps": sft.state.global_step, "loss": sft_result.training_loss, "weights_changed": sft_changed,
                "checkpoint_roundtrip": restored_ok},
        "grpo": {"steps": grpo.state.global_step, "loss": grpo_result.training_loss, "weights_changed": grpo_changed},
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "trl", "accelerate", "datasets")},
    }
    report["passed"] = (sft_changed and grpo_changed and restored_ok and math.isfinite(sft_result.training_loss)
                        and math.isfinite(grpo_result.training_loss))
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise RuntimeError(f"smoke failed: {report}")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(run(args.output), indent=2))
