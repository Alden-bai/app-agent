"""Check all eight state shards and reload a text export in BF16 on one GPU."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="SFT global_step_N or GRPO global_step_N/actor")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("output already exists")
    states = []
    for rank in range(8):
        for prefix in ("model", "optim", "extra_state"):
            path = a.checkpoint / f"{prefix}_world_size_8_rank_{rank}.pt"
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError("missing or empty state shard: " + str(path))
            states.append({"name": path.name, "bytes": path.stat().st_size})
    import torch
    from harness_awm.training.local import checkpoint_identity, load_checkpoint
    from harness_awm.training.actor import stop_token_ids
    model_path = a.checkpoint / "huggingface"
    identity = checkpoint_identity(model_path)
    model, tokenizer = load_checkpoint(model_path, dtype="bfloat16")
    for param in model.parameters():
        if not torch.isfinite(param).all():
            raise ValueError("nonfinite exported weights")
    ids = tokenizer.apply_chat_template([{"role": "user", "content": 'Return only this JSON: {"ok":true}'}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    ids = ids.to("cuda")
    eos = stop_token_ids(model, tokenizer)
    with torch.no_grad():
        output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=64,
            do_sample=False, eos_token_id=eos, pad_token_id=tokenizer.pad_token_id)
    completion = output[0, ids.shape[-1]:].tolist()
    if not completion or completion[-1] not in eos:
        raise ValueError("reload generated no EOS-terminated response within 64 tokens")
    report = {"passed": True, "hf_reload_passed": True, "state_shards": states,
        "checkpoint": identity, "completion_ids": completion, "judge_requests": 0,
        "optimizer_resume_tested": False, "effectiveness_claim": False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "state_shards": len(states), "report": str(a.output)}))


if __name__ == "__main__":
    main()
