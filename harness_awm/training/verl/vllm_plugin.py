"""Opt-in registration of the text-only SFT architecture in vLLM 0.20.2."""
import os


def register():
    if os.environ.get("AWM_VERL_TEXT_MODEL") != "1":
        return
    from importlib.metadata import version
    if version("vllm") != "0.20.2":
        raise RuntimeError("AWM text-model plugin requires vllm==0.20.2")
    from vllm import ModelRegistry
    ModelRegistry.register_model("Qwen3_5ForCausalLM",
        "harness_awm.training.verl.vllm_model:AWMQwen35ForCausalLM")

