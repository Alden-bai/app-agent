"""SFT-only SDPA runtime; rollout dependencies remain separately pinned."""
from importlib import metadata
from pathlib import Path

from ...contracts import ROOT
from ...io import file_hash, read_json


def verify_sft_runtime():
    import verl
    lock = read_json(ROOT / "requirements/verl-sft-runtime.json")
    for package, expected in lock["packages"].items():
        actual = metadata.version(package).split("+")[0]
        if actual != expected:
            raise ValueError(f"{package}: expected {expected}, found {actual}")
    for relative, expected in lock["source_sha256"].items():
        if file_hash(Path(verl.__file__).parent / relative) != expected:
            raise ValueError("verl source differs from pinned revision: " + relative)


def install_sdpa_padding():
    """Use upstream differentiable indexing helpers, without FlashAttention."""
    from verl.utils import attention_utils, npu_flash_attn_utils as padding
    functions = tuple(getattr(padding, name) for name in
                      ("index_first_axis", "pad_input", "rearrange", "unpad_input"))
    attention_utils._get_attention_functions = lambda: functions
    # Native SFT has no TRL value head. TRL 1.x removed this optional PPO
    # class, which the pinned verl patch otherwise imports unconditionally.
    from verl.models.transformers import monkey_patch
    monkey_patch.is_trl_available = lambda: False
    # The no-FlashAttention BF16 fallback loops over *every token* and builds
    # large slice-backward graphs. Only replace CE, never norms/MLP/attention.
    from verl.workers.engine.fsdp import transformer_impl
    transformer_impl.logprobs_from_logits = sft_logprobs
    transformer_impl.prepare_micro_batches = prepare_sft_micro_batches


def sft_logprobs(logits, labels, inplace_backward=True):
    from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
    loss, *_ = LigerCrossEntropyFunction.apply(
        logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1),
        None, -100, 0.0, 0.0, "none", None, False, False, False)
    return -loss.reshape(labels.shape)


def prepare_sft_micro_batches(data, dp_group=None, **kwargs):
    """Bound dense SDPA/logit allocation; upstream balances estimated FLOPs."""
    from verl.utils import tensordict_utils as tu
    from verl.workers.engine.utils import prepare_micro_batches
    import torch.distributed as dist
    if not tu.get_non_tensor_data(data, "use_dynamic_bsz", default=False):
        return prepare_micro_batches(data, dp_group=dp_group, **kwargs)
    if dist.is_initialized() and dist.get_world_size() != 1:
        raise ValueError("strict SFT token budgeting currently requires one GPU")
    budget = tu.get_non_tensor_data(data, "max_token_len_per_gpu", default=0)
    lengths = data["input_ids"].offsets().diff().tolist()
    groups, group, longest = [], [], 0
    for index, length in enumerate(lengths):
        if length > budget:
            raise ValueError("sample exceeds SFT token budget")
        if group and max(longest, length) * (len(group) + 1) > budget:
            groups.append(group)
            group, longest = [], 0
        group.append(index)
        longest = max(longest, length)
    if group:
        groups.append(group)
    return [tu.index_select_tensor_dict(data, group) for group in groups], groups
