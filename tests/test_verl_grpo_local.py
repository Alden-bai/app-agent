"""Development admission, explicit unlimited budget, and local kernel contracts."""
import pytest

from harness_awm.contracts import RunConfig
from harness_awm.io import file_hash, read_json, write_json
from harness_awm.reward_policy import AdmissionManifest, admission_for
from harness_awm.training.grpo import preflight
from test_training_grpo import setup_sources


def test_unlimited_development_is_explicit_and_formal_stays_strict():
    raw = dict(purpose='development', actor='training', judge='api', judge_protocol='rubric_v2',
               isolation='landlock', max_physical_requests=None, max_total_tokens=None)
    assert RunConfig(**raw).max_total_tokens is None
    for change in ({'max_total_tokens': 100}, {'judge': 'mock'}, {'isolation': 'development'}, {'purpose': 'formal'}):
        with pytest.raises(ValueError):
            RunConfig(**{**raw, **change})
    with pytest.raises(ValueError, match='together'):
        RunConfig(judge_rpm=5000)


def test_landlock_requires_new_bound_calibration_and_rejects_tampering(tmp_path, monkeypatch):
    task, config, original, rubric = setup_sources(tmp_path)
    config = RunConfig.model_validate({**config.model_dump(), 'purpose': 'development', 'isolation': 'landlock',
                                      'max_physical_requests': None, 'max_total_tokens': None})
    with pytest.raises(ValueError, match='requires_strict'):
        admission_for(task, config, original, rubric)
    manifest = read_json(config.reward_admissions)
    entry = manifest['tasks'][task.task_instance_id]
    proof = tmp_path / 'calibration.json'
    write_json(proof, {'passed': True, 'task_instance_id': task.task_instance_id, 'isolation': 'landlock',
        'judge_config_hash': entry['judge_config_hash'],
        'controls': [{'matches': True, 'test_only': True, 'path': str(tmp_path)} for _ in range(4)]})
    entry.update(purpose='development', isolation='landlock', calibration_ref=str(proof), calibration_sha256=file_hash(proof))
    manifest['schema_version'] = 'awm-reward-admissions-v2'
    write_json(config.reward_admissions, manifest)
    monkeypatch.setattr('harness_awm.reward_policy.verify_seal', lambda path: {})
    assert admission_for(task, config, original, rubric).isolation == 'landlock'
    assert preflight([task], config)['checked_tasks'] == 1
    with pytest.raises(ValueError, match='profile_mismatch'):
        admission_for(task, config.model_copy(update={'isolation': 'strict'}), original, rubric)
    proof.write_text('{}')
    with pytest.raises(ValueError, match='calibration_changed'):
        admission_for(task, config, original, rubric)
    manifest['schema_version'] = 'awm-reward-admissions-v1'
    with pytest.raises(ValueError, match='schema_mismatch'):
        AdmissionManifest.model_validate(manifest)


def test_local_logprobs_preserve_values_and_gradients():
    import torch
    from harness_awm.training.verl.compat import fp32_logprobs
    torch.manual_seed(42)
    logits = torch.randn(3, 7, 19, requires_grad=True)
    labels = torch.randint(19, (3, 7))
    values = fp32_logprobs(logits, labels)
    reference = -torch.nn.functional.cross_entropy(logits.reshape(-1, 19), labels.reshape(-1), reduction='none').reshape(3, 7)
    assert torch.allclose(values, reference, atol=1e-6)
    assert torch.allclose(torch.autograd.grad(values.sum(), logits, retain_graph=True)[0],
                          torch.autograd.grad(reference.sum(), logits)[0], atol=1e-6)


def test_sdpa_compat_does_not_replace_grpo_microbatching():
    pytest.importorskip('verl')
    from verl.workers.engine.fsdp import transformer_impl
    from harness_awm.training.verl.compat import install_local_sdpa, fp32_logprobs
    original = transformer_impl.prepare_micro_batches
    install_local_sdpa()
    assert transformer_impl.prepare_micro_batches is original
    assert transformer_impl.logprobs_from_logits is fp32_logprobs


def test_text_mrope_coordinates_and_multimodal_rejection():
    pytest.importorskip('vllm')
    import torch
    from harness_awm.training.verl.vllm_model import AWMQwen35ForCausalLM
    positions, delta = AWMQwen35ForCausalLM.get_mrope_input_positions(None, [9, 4, 2], [])
    assert torch.equal(positions, torch.arange(3).repeat(3, 1)) and delta == 0
    with pytest.raises(ValueError, match='multimodal'):
        AWMQwen35ForCausalLM.get_mrope_input_positions(None, [9], [object()])


def test_rate_window_is_shared_and_retries_are_accounted(tmp_path, monkeypatch):
    import asyncio
    import sqlite3
    from types import SimpleNamespace
    from harness_awm import judge_rate_limit as rate
    now = [1000.0]
    monkeypatch.setattr(rate.time, 'time', lambda: now[0])
    gate = rate.RateLimit(tmp_path / 'rate.sqlite', 5000, 2000000)
    first, delay = gate._reserve(1000000)
    assert first and delay == 0
    second = rate.RateLimit(gate.path, 5000, 2000000)
    assert second._reserve(1000000)[0] is None
    now[0] += 31
    assert second._reserve(1000000)[0]
    now[0] += 1
    assert second._reserve(1)[0] is None
    calls = []
    class Error(Exception):
        status_code = 429
        code = 'rate_limit_exceeded'
        response = SimpleNamespace(headers={'retry-after': '1'})
    class ImmediateGate:
        async def acquire(self, tokens): return len(calls)
        def defer(self, seconds): assert seconds >= 5
        def refund(self, reservation): raise AssertionError('sent request cannot be refunded')
    async def transport(config, kwargs, budget, key, stage, folder, **options):
        calls.append(key)
        if len(calls) == 1: raise Error()
        return {'usage': {'total_tokens': 42}}
    monkeypatch.setattr(rate, 'estimated_tokens', lambda request: 1024)
    wrapped = rate.paced_transport(transport, ImmediateGate())
    result = asyncio.run(wrapped(SimpleNamespace(base_url='https://api.openai.com/v1'),
        {'model': 'gpt-6-luna'}, None, 'r', 'judge_v2', tmp_path / 'request'))
    assert result['usage']['total_tokens'] == 42 and calls == ['r', 'r/rate_retry_01']


def test_single_sequence_outputs_match_dense_values_and_gradients():
    import torch
    from types import SimpleNamespace
    from verl.utils import tensordict_utils as tu
    from harness_awm.training.verl.compat import memory_efficient_outputs
    values = torch.tensor([1, 2, 3, 4])
    offsets = torch.tensor([0, 4])
    ids = torch.nested.nested_tensor_from_jagged(values, offsets)
    batch = tu.get_tensordict({'input_ids': ids})
    tu.assign_non_tensor(batch, use_remove_padding=False)
    leaf = torch.randn(1, 4, 11, requires_grad=True)
    labels = torch.tensor([2, 3, 4, 5])
    # Non-leaf matches model-produced logits, allowing temperature scaling.
    output = memory_efficient_outputs(None, SimpleNamespace(logits=leaf * 1),
        {'temperature': torch.tensor([.8]), 'input_ids_rmpad_rolled': labels}, batch, None)
    actual = output['log_probs'].values()
    expected = (leaf[0] / .8).log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.allclose(torch.autograd.grad(actual.sum(), leaf)[0],
                          torch.autograd.grad(expected.sum(), leaf)[0], atol=1e-6)


def test_noexec_temporary_storage_redirects_compiled_kernels(tmp_path, monkeypatch):
    import os
    from types import SimpleNamespace
    from harness_awm.training.verl import compat
    noexec = tmp_path / 'noexec'
    runtime = tmp_path / 'runtime'
    noexec.mkdir()
    monkeypatch.setattr(compat.tempfile, 'gettempdir', lambda: str(noexec))
    monkeypatch.setattr(compat.tempfile, 'tempdir', str(noexec))
    monkeypatch.setattr(compat.sysconfig, 'get_paths', lambda: {'purelib': str(runtime / 'lib/python3.12/site-packages')})
    monkeypatch.setattr(compat.os, 'statvfs', lambda p: SimpleNamespace(f_flag=os.ST_NOEXEC if p == noexec else 0))
    for name in ['TMPDIR', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'VLLM_CACHE_ROOT']:
        monkeypatch.setenv(name, str(noexec))
    compat.configure_executable_cache()
    assert compat.tempfile.tempdir is None
    assert os.environ['TMPDIR'] == str(runtime / 'kernel-cache/tmp')
    assert os.environ['TRITON_CACHE_DIR'] == str(runtime / 'kernel-cache/triton')
    assert (runtime / 'kernel-cache/tmp').is_dir()


def test_executable_temporary_storage_keeps_existing_cache(tmp_path, monkeypatch):
    import os
    from types import SimpleNamespace
    from harness_awm.training.verl import compat
    monkeypatch.setattr(compat.tempfile, 'gettempdir', lambda: str(tmp_path))
    monkeypatch.setattr(compat.os, 'statvfs', lambda p: SimpleNamespace(f_flag=0))
    monkeypatch.setenv('TRITON_CACHE_DIR', '/existing/kernel-cache')
    compat.configure_executable_cache()
    assert os.environ['TRITON_CACHE_DIR'] == '/existing/kernel-cache'
