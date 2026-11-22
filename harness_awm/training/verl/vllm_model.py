"""Expose vLLM's existing text decoder with its upstream hybrid-cache protocol.

In vLLM 0.20.2 the text decoder exists but only the multimodal wrapper is
registered. Text SFT weights use model.layers.*, not language_model.model.*.
No weight remapping or vision tower is needed for this registered subclass.
"""
import torch
from vllm.model_executor.models.interfaces import IsHybrid, SupportsMRoPE
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration


class AWMQwen35ForCausalLM(Qwen3_5ForCausalLM, IsHybrid, SupportsMRoPE):
    get_mamba_state_dtype_from_config = Qwen3_5ForConditionalGeneration.__dict__["get_mamba_state_dtype_from_config"]
    get_mamba_state_shape_from_config = Qwen3_5ForConditionalGeneration.__dict__["get_mamba_state_shape_from_config"]
    get_mamba_state_copy_func = Qwen3_5ForConditionalGeneration.__dict__["get_mamba_state_copy_func"]

    def get_mrope_input_positions(self, input_tokens, mm_features):
        # Pure text has identical temporal/height/width coordinates, as in
        # the upstream multimodal model's no-image/no-video branch.
        if mm_features:
            raise ValueError("text-only SFT checkpoint cannot accept multimodal inputs")
        return torch.arange(len(input_tokens), dtype=torch.long).repeat(3, 1), 0

