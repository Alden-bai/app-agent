"""A completed schedule must not silently restore its old peak or zero LR."""
from types import SimpleNamespace
import pytest

from harness_awm.training.continuation import stage_factor, continuation_callback


def test_continuation_schedule_boundaries():
    f = lambda step: stage_factor(step, start=210, end=420, warmup=7)
    assert f(210) == 0
    assert f(217) == 1
    assert 0 < f(315) < 1
    assert f(420) == f(500) == 0
    assert all(f(i + 1) <= f(i) for i in range(217, 420))
    with pytest.raises(ValueError):
        f(209)


def test_loaded_optimizer_moments_preserved_and_resume_lr_matches(tmp_path):
    import torch
    p = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([p], lr=1e-5)
    old_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 0.)
    p.grad = torch.ones(1)
    optimizer.step()
    moments = optimizer.state[p]['exp_avg'].clone()
    callback = continuation_callback(210, 420, 7)
    args = SimpleNamespace(learning_rate=5e-6, output_dir=str(tmp_path))
    callback.on_train_begin(args, SimpleNamespace(global_step=210, max_steps=420), None,
                            optimizer=optimizer, lr_scheduler=old_scheduler)
    assert optimizer.param_groups[0]['lr'] == 0
    assert torch.equal(moments, optimizer.state[p]['exp_avg'])
    for _ in range(7):
        optimizer.step()
        old_scheduler.step()
    assert optimizer.param_groups[0]['lr'] == pytest.approx(5e-6)
    # Callback changes schedule only, leaving moments untouched.
    before = optimizer.state[p]['exp_avg'].clone()
    callback.on_train_begin(args, SimpleNamespace(global_step=315, max_steps=420), None,
                            optimizer=optimizer, lr_scheduler=old_scheduler)
    assert torch.equal(before, optimizer.state[p]['exp_avg'])
    assert optimizer.param_groups[0]['lr'] == pytest.approx(
        5e-6 * stage_factor(315, start=210, end=420, warmup=7))
    old_scheduler.step()
    assert optimizer.param_groups[0]['lr'] == pytest.approx(
        5e-6 * stage_factor(316, start=210, end=420, warmup=7))


def test_exported_prefix_is_loaded_strictly(tmp_path):
    import torch
    from safetensors.torch import save_file
    from harness_awm.training.continuation import load_resume_weights

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(3, 2)
            self.lm_head = torch.nn.Linear(2, 3, bias=False)
            self.lm_head.weight = self.model.embed_tokens.weight

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def get_output_embeddings(self):
            return self.lm_head

    model = Model()
    expected = torch.full((3, 2), 7.)
    save_file({'model.language_model.embed_tokens.weight': expected}, str(tmp_path / 'model.safetensors'))
    audit = load_resume_weights(model, tmp_path)
    assert audit['renamed_tensor_count'] == 1
    assert torch.equal(model.model.embed_tokens.weight, expected)
    save_file({'unknown.weight': expected}, str(tmp_path / 'model.safetensors'))
    with pytest.raises(RuntimeError):
        load_resume_weights(model, tmp_path)
