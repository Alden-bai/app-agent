"""Local on-policy sampling with an immutable record of each real context."""
import asyncio
import inspect
from collections.abc import Mapping
from pathlib import Path

from ..io import digest, write_json


STOP_POLICY = "generation-and-tokenizer-eos-v1"


def stop_token_ids(model, tokenizer):
    """Honor both the model's EOS and the chat tokenizer's end-of-turn token."""
    tokens = set()
    for value in (getattr(model.generation_config, "eos_token_id", None),
                  getattr(tokenizer, "eos_token_id", None)):
        if value is not None:
            tokens.update(value if isinstance(value, (list, tuple)) else [value])
    if not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("missing_or_invalid_stop_tokens")
    return sorted(tokens)


class TrainingActorBackend:
    def __init__(self, model, tokenizer, *, policy_version, max_output_tokens=2048,
                 context_limit=32768, test_only=False, greedy=False, record_policy=True):
        if not policy_version or max_output_tokens < 1 or context_limit < 2:
            raise ValueError("invalid_training_actor_settings")
        if model.config.is_encoder_decoder:
            raise ValueError("training_actor_requires_causal_language_model")
        self.model, self.tokenizer = model, tokenizer
        self.policy_version = policy_version
        self.max_output_tokens, self.context_limit = max_output_tokens, context_limit
        self.test_only = test_only
        self.greedy, self.record_policy = greedy, record_policy
        self.logits_options = {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(model.forward).parameters else {}
        self.records = []

    async def generate(self, messages, folder, turn, *, output_tokens=None, **options):
        import torch
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    enable_thinking=False)
        if isinstance(prompt, Mapping):
            prompt = prompt["input_ids"]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        if not prompt or any(type(t) is not int for t in prompt):
            raise ValueError("training_prompt_must_be_flat_token_ids")
        count = output_tokens if output_tokens is not None else self.max_output_tokens
        if count < 1 or len(prompt) + count > self.context_limit:
            raise ValueError("training_context_limit_exceeded")
        ids = torch.tensor([prompt], device=next(self.model.parameters()).device)
        self.model.eval()  # Dropout must stay off in sampling AND loss evaluation.
        completion, old, cache = [], [], None
        eos_ids = stop_token_ids(self.model, self.tokenizer)
        # Sample the raw policy with a KV cache, yielding between tokens so
        # episode cancellation cannot leave a background generation using weights.
        # No inherited logits processors, forced tokens or sampling truncation.
        for _ in range(count):
            await asyncio.sleep(0)
            with torch.inference_mode():
                inputs = self.model.prepare_inputs_for_generation(ids, attention_mask=torch.ones_like(ids),
                    past_key_values=cache, next_sequence_length=1 if cache is not None else None,
                    is_first_iteration=cache is None, use_cache=True)
                output = self.model(**inputs, **self.logits_options)
                probs = output.logits[0, -1].float().log_softmax(-1)
                if not torch.isfinite(probs).all():
                    raise ValueError("nonfinite_sampling_logprobs")
                token = int(probs.argmax()) if self.greedy else int(torch.multinomial(probs.exp(), 1))
                completion.append(token)
                old.append(float(probs[token]))
                cache = output.past_key_values
                ids = torch.cat([ids, ids.new_tensor([[token]])], dim=1)
            if token in eos_ids:
                break
        content = self.tokenizer.decode(completion, skip_special_tokens=True)
        record = {"schema_version": "awm-policy-turn-v1", "turn": turn,
            "policy_version": self.policy_version, "messages_hash": digest(messages),
            "prompt_ids": prompt, "completion_ids": completion, "old_logprobs": old,
            "action_mask": [0] * len(prompt) + [1] * len(completion),
            "stop_policy": STOP_POLICY, "eos_token_ids": eos_ids,
            "sampling": ({"method": "greedy"} if self.greedy else
                         {"temperature": 1.0, "top_k": 0, "top_p": 1.0}), "test_only": self.test_only}
        if self.record_policy:
            write_json(Path(folder) / "policy_turn.json", record)
            self.records.append(record)
        return {"content": content, "finish_reason": "stop" if completion[-1] in eos_ids else "length",
                "test_only": self.test_only, "model_version": self.policy_version,
                "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(completion),
                          "total_tokens": len(prompt) + len(completion)}}
