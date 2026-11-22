"""Explicit GRPO SDPA compatibility; never installs SFT batching or loss hooks."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sysconfig
import tempfile


def cuda_home(profile='local-sdpa'):
    if profile == 'local-sdpa-cu128':
        nvcc = shutil.which('nvcc')
        homes = [os.environ.get('CUDA_HOME'), str(Path(nvcc).resolve().parents[1]) if nvcc else None,
                 '/usr/local/cuda-12.8']
        for value in filter(None, homes):
            home = Path(value).resolve()
            compiler = home / 'bin/nvcc'
            if compiler.is_file() and re.search(r'release 12\.8\b', subprocess.check_output([compiler, '--version'], text=True)):
                return str(home)
        raise ValueError('Set CUDA_HOME to the prepared CUDA 12.8 toolkit (bin/nvcc)')
    home = Path(sysconfig.get_paths()['purelib']) / 'nvidia/cu13'
    if not (home / 'bin/nvcc').is_file():
        raise ValueError('local-sdpa requires cuda-toolkit[nvcc,crt,nvvm]==13.0.2')
    return str(home)


def install_local_sdpa():
    from verl.utils import attention_utils, npu_flash_attn_utils as padding
    functions = tuple(getattr(padding, name) for name in
                      ("index_first_axis", "pad_input", "rearrange", "unpad_input"))
    attention_utils._get_attention_functions = lambda: functions
    from verl.models.transformers import monkey_patch
    monkey_patch.is_trl_available = lambda: False
    from verl.workers.engine.fsdp import transformer_impl
    transformer_impl.logprobs_from_logits = fp32_logprobs
    cls = transformer_impl.FSDPEngineWithLMHead
    if not hasattr(cls, "_awm_original_outputs"):
        cls._awm_original_outputs = cls.prepare_model_outputs
        cls.prepare_model_outputs = memory_efficient_outputs


def fp32_logprobs(logits, labels, inplace_backward=True):
    if logits.device.type == "cpu":
        return logits.float().log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    if not logits.requires_grad:
        import torch
        flat = logits.reshape(-1, logits.shape[-1])
        targets = labels.reshape(-1)
        return torch.cat([chunk.float().gather(-1, targets[start:start + len(chunk), None]).squeeze(-1)
                          - torch.logsumexp(chunk.float(), -1)
                          for start in range(0, len(flat), 128)
                          for chunk in [flat[start:start + 128]]]).reshape(labels.shape)
    from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
    # Liger stores its backward gradient in the input buffer during forward.
    # float() already copies BF16, but must not alias a native FP32 logits tensor.
    scores = logits.float()
    if scores.data_ptr() == logits.data_ptr():
        scores = scores.clone()
    loss, *_ = LigerCrossEntropyFunction.apply(
        scores.reshape(-1, logits.shape[-1]), labels.reshape(-1),
        None, -100, 0.0, 0.0, "none", None, False, False, False)
    return -loss.reshape(labels.shape)


def configure_executable_cache():
    """Triton loads compiled shared objects; tmpfs may be mounted noexec."""
    temporary = Path(tempfile.gettempdir())
    if not os.statvfs(temporary).f_flag & os.ST_NOEXEC:
        return
    cache = Path(sysconfig.get_paths()['purelib']).parents[2] / 'kernel-cache'
    cache.mkdir(parents=True, exist_ok=True)
    if os.statvfs(cache).f_flag & os.ST_NOEXEC:
        raise ValueError('GRPO kernel cache also resides on a noexec filesystem')
    for variable, name in [('TRITON_CACHE_DIR', 'triton'),
                           ('TORCHINDUCTOR_CACHE_DIR', 'torchinductor'),
                           ('VLLM_CACHE_ROOT', 'vllm')]:
        os.environ[variable] = str(cache / name)
    (cache / 'tmp').mkdir(exist_ok=True)
    os.environ['TMPDIR'] = str(cache / 'tmp')
    tempfile.tempdir = None


def worker_setup():
    profile = os.environ.get('AWM_VERL_RUNTIME_PROFILE')
    if profile in {'local-sdpa', 'local-sdpa-cu128'}:
        configure_executable_cache()
        os.environ['CUDA_HOME'] = cuda_home(profile)
        install_local_sdpa()


def memory_efficient_outputs(self, output, output_args, micro_batch, logits_processor_func):
    """Avoid a second full vocabulary tensor for the single-sequence SDPA path."""
    import torch
    from verl.utils import tensordict_utils as tu
    from verl.utils.dataset.dataset_utils import DatasetPadMode
    flags = ("use_remove_padding", "use_fused_kernels", "calculate_entropy",
             "calculate_sum_pi_squared", "distillation_use_topk")
    if (any(tu.get_non_tensor_data(micro_batch, key=k, default=k == "use_remove_padding") for k in flags)
            or tu.get_non_tensor_data(micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING) != DatasetPadMode.NO_PADDING
            or output.logits.shape[0] != 1):
        return self._awm_original_outputs(output, output_args, micro_batch, logits_processor_func)
    logits = output.logits
    temperature = output_args["temperature"].unsqueeze(-1).unsqueeze(-1)
    logits.div_(temperature.clamp(min=1e-8).to(logits.dtype))
    offsets = micro_batch["input_ids"].offsets()
    length = int((offsets[1] - offsets[0]).item())
    probs = fp32_logprobs(logits[0, :length], output_args["input_ids_rmpad_rolled"])
    return {"log_probs": torch.nested.nested_tensor_from_jagged(probs, offsets)}
