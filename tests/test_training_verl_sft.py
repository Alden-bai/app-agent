"""Current-response SFT migration: frozen labels, full coverage, real verl loss."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness_awm.io import file_hash, read_json, write_json
from harness_awm.training.verl.sft_data import EncodedSFTDataset, LengthGroupedSampler, verified_rows


def data_fixture(tmp_path):
    from test_training_grpo import tiny_policy
    model, tokenizer = tiny_policy("qwen3.5")
    path = tmp_path / "model"
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    data = tmp_path / "data"
    data.mkdir()
    samples = [
        {"input_ids": [3, 4, 5, 6], "labels": [-100, -100, 5, 6]},
        {"input_ids": [7, 8, 9], "labels": [-100, -100, 9]},
        {"input_ids": [2, 3, 4, 5, 6], "labels": [-100, -100, -100, 5, 6]},
    ]
    (data / "train.jsonl").write_text(''.join(json.dumps(s) + '\n' for s in samples))
    write_json(data / "report.json", {"train_sha256": file_hash(data / "train.jsonl"),
        "tokenizer_files": {p.name: file_hash(p) for p in path.iterdir()
                            if p.name in {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}}})
    return model, tokenizer, path, data, samples


def test_data_preserves_labels_and_rejects_truncation_or_tampering(tmp_path):
    _, _, model, data, samples = data_fixture(tmp_path)
    actual, _ = verified_rows(data, model, 5)
    assert actual == samples
    with pytest.raises(ValueError, match="truncation"):
        verified_rows(data, model, 4)
    (data / "train.jsonl").write_text('{}\n')
    with pytest.raises(ValueError, match="checksum"):
        verified_rows(data, model, 5)


@pytest.mark.parametrize("size,batch,world", [(3, 4, 1), (7, 4, 2), (11, 6, 3), (1, 4, 4)])
def test_every_sample_once_per_epoch_across_ranks_with_masked_tail(size, batch, world):
    lengths = list(range(2, size + 2))
    all_indices = []
    for rank in range(world):
        sampler = LengthGroupedSampler(lengths, batch, rank, world)
        sampler.set_epoch(3)
        indices = list(sampler)
        assert len(indices) == len(sampler)
        assert len(indices) % (batch // world) == 0
        all_indices.extend(indices)
    assert sorted(i for i in all_indices if i != -1) == list(range(size))
    assert all_indices.count(-1) < batch
    dummy = EncodedSFTDataset([], 0)[-1]
    assert dummy["loss_mask"].sum() == 0


def test_real_verl_sft_loss_and_gradients_match_masked_cross_entropy(tmp_path, monkeypatch):
    pytest.importorskip("verl")
    import torch
    import torch.nn.functional as F
    from verl.utils.dataset.dataset_utils import SFTTensorCollator
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.losses import sft_loss
    # This is a CPU semantic test, even when the optional CUDA FLA is installed.
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        function = getattr(qwen, name)
        monkeypatch.setattr(qwen, name, function.__wrapped__)
    model, _, _, _, samples = data_fixture(tmp_path)
    model.eval()
    dataset = EncodedSFTDataset(samples, 0)
    rows = [dataset[i] for i in [0, 1, 2, -1]]
    collated = SFTTensorCollator("no_padding")(rows)
    td = tu.get_tensordict(collated, non_tensor_dict={"pad_mode": "no_padding", "dp_size": 1,
                                                   "batch_num_tokens": 5})
    probs = []
    for row in rows:
        ids = row["input_ids"]
        logits = model(input_ids=ids[None], use_cache=False).logits[0].float()
        targets = torch.cat([ids[1:], ids.new_zeros(1)])
        probs.append(logits.log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1))
    loss, _ = sft_loss(None, {"log_probs": torch.nested.as_nested_tensor(probs, layout=torch.jagged)}, td)
    loss.backward()
    gradients = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    reference = 0
    for sample in samples:
        ids = torch.tensor(sample["input_ids"])[None]
        logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
        reference += F.cross_entropy(logits, torch.tensor(sample["labels"][1:]), ignore_index=-100,
                                      reduction="sum") / 5
    reference.backward()
    assert torch.allclose(loss, reference, atol=1e-6)
    for p, grad in zip(model.parameters(), gradients):
        assert torch.allclose(p.grad, grad, atol=2e-6, rtol=2e-4)


def test_sft_plan_and_native_model_config(tmp_path):
    pytest.importorskip("verl")
    from harness_awm.training.verl.sft import prepare, load_plan, compose_sft_config
    from verl.utils.config import omega_conf_to_dataclass
    from verl.utils.model import get_hf_auto_model_class
    _, _, model, data, _ = data_fixture(tmp_path)
    args = SimpleNamespace(model=model, data=data, output=tmp_path / "run", gpus=1, global_batch_size=2,
        epochs=2, max_length=32, learning_rate=1e-5, warmup_ratio=.03, seed=1, max_steps=-1)
    report = prepare(args)
    assert report["optimizer_steps"] == 4 and report["supervised_tokens"] == 5
    plan = load_plan(args.output)
    config = compose_sft_config(plan)
    model_config = omega_conf_to_dataclass(config.model)
    assert model_config.hf_config.model_type == "qwen3_5_text"
    cls = get_hf_auto_model_class(model_config.hf_config)
    loaded, info = cls.from_pretrained(model_config.local_path, config=model_config.hf_config,
                                      output_loading_info=True, local_files_only=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert all(p.requires_grad for p in loaded.parameters())
    assert config.engine.strategy == "fsdp2" and not config.model.use_remove_padding
    assert config.data.val_files is None


def test_base_multimodal_checkpoint_loads_text_parameters_without_vision(tmp_path):
    pytest.importorskip("verl")
    import torch
    from safetensors.torch import save_file
    from harness_awm.training.verl.sft import prepare, load_plan, compose_sft_config
    from verl.utils.config import omega_conf_to_dataclass
    from verl.utils.model import get_hf_auto_model_class
    original, _, model, data, _ = data_fixture(tmp_path)
    raw = read_json(model / "config.json")
    write_json(model / "config.json", {"model_type": "qwen3_5", "text_config": raw,
        "architectures": ["Qwen3_5ForConditionalGeneration"], "vision_config": {}})
    # Match the official multimodal checkpoint's language-model prefix.
    weights = {(k.replace("model.", "model.language_model.", 1) if k.startswith("model.") else k):
               v.detach().clone() for k, v in original.state_dict().items()}
    save_file(weights, model / "model.safetensors", metadata={"format": "pt"})
    args = SimpleNamespace(model=model, data=data, output=tmp_path / "run", gpus=1, global_batch_size=2,
        epochs=1, max_length=32, learning_rate=1e-5, warmup_ratio=.03, seed=1, max_steps=-1)
    prepare(args)
    config = compose_sft_config(load_plan(args.output))
    model_config = omega_conf_to_dataclass(config.model)
    cls = get_hf_auto_model_class(model_config.hf_config)
    loaded, info = cls.from_pretrained(model, config=model_config.hf_config,
                                      output_loading_info=True, local_files_only=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert loaded.config.model_type == "qwen3_5_text"
    for name, value in loaded.state_dict().items():
        assert torch.equal(value, original.state_dict()[name])


def test_sdpa_padding_roundtrip_and_gradients():
    pytest.importorskip("verl")
    import torch
    from harness_awm.training.verl.sft_compat import install_sdpa_padding
    from verl.utils import attention_utils
    install_sdpa_padding()
    x = torch.randn(2, 5, 3, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    unpadded, indices, *_ = attention_utils.unpad_input(x, mask)
    restored = attention_utils.pad_input(unpadded, indices, 2, 5)
    assert torch.equal(restored, x * mask[..., None])
    restored.square().sum().backward()
    assert torch.equal(x.grad, 2 * x.detach() * mask[..., None])


def test_dynamic_sft_budget_is_frozen_and_cannot_truncate(tmp_path):
    pytest.importorskip("verl")
    from harness_awm.training.verl.sft import prepare, load_plan, compose_sft_config
    _, _, model, data, _ = data_fixture(tmp_path)
    args = SimpleNamespace(model=model, data=data, output=tmp_path / "run", gpus=1, global_batch_size=2,
        epochs=2, max_length=32, learning_rate=1e-5, warmup_ratio=.03, seed=1, max_steps=-1,
        max_tokens_per_gpu=4)
    with pytest.raises(ValueError, match="longest sample"):
        prepare(args)
    args.max_tokens_per_gpu = 12
    prepare(args)
    config = compose_sft_config(load_plan(args.output))
    assert config.data.use_dynamic_bsz
    assert config.data.max_token_len_per_gpu == 12
    assert config.data.train_batch_size == 2
    assert not config.model.use_remove_padding


def test_strict_micro_batches_preserve_mask_normalization_and_gradients():
    pytest.importorskip("verl")
    import torch
    from verl.utils import tensordict_utils as tu
    from verl.utils.dataset.dataset_utils import SFTTensorCollator
    from verl.workers.utils.losses import sft_loss
    from harness_awm.training.verl.sft_compat import prepare_sft_micro_batches
    samples = [{"input_ids": list(range(n)), "labels": [-100] * (n-2) + [n-2,n-1]}
               for n in [17, 9, 8, 7, 3]]
    dataset = EncodedSFTDataset(samples, 0)
    rows = [dataset[i] for i in [0, 1, 2, 3, 4, -1]]
    batch = tu.get_tensordict(SFTTensorCollator("no_padding")(rows), non_tensor_dict={
        "use_dynamic_bsz": True, "max_token_len_per_gpu": 20,
        "pad_mode": "no_padding", "dp_size": 1, "batch_num_tokens": 10})
    micros, groups = prepare_sft_micro_batches(batch)
    assert [i for group in groups for i in group] == list(range(6))
    for micro in micros:
        lengths = micro["input_ids"].offsets().diff()
        assert lengths.max() * len(lengths) <= 20
        assert tu.get_non_tensor_data(micro, "batch_num_tokens", default=None) == 10
    parameter = torch.tensor(0.7, requires_grad=True)
    def loss_for(data):
        probs = torch.nested.as_nested_tensor(
            [-parameter * ids.float() for ids in data["input_ids"].unbind()], layout=torch.jagged)
        return sft_loss(None, {"log_probs": probs}, data)[0]
    total = loss_for(batch)
    total.backward()
    expected_gradient = parameter.grad.clone()
    parameter.grad = None
    split = sum(loss_for(micro) for micro in micros)
    split.backward()
    assert torch.allclose(split, total)
    assert torch.allclose(parameter.grad, expected_gradient)
