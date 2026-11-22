"""Portable skill handoff and held-out split propagation; no model or API calls."""
import importlib.util
from pathlib import Path
import shutil
import sys

import pytest

from harness_awm.contracts import RunConfig
from harness_awm.io import file_hash, read_json, write_json
from harness_awm.training import skill_eval
from harness_support import fixture
from test_skillbank import example_skill
from test_harness_v2 import TOOLS

ROOT = Path(__file__).resolve().parents[1]


def prepare_module():
    spec = importlib.util.spec_from_file_location('prepare_delivery_skills', ROOT / 'scripts/prepare_delivery_skills.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bank_relocated_and_tampering_rejected(tmp_path, monkeypatch):
    module = prepare_module()
    bundle = tmp_path / 'other-machine' / 'skills'
    shutil.copytree(ROOT / 'delivery/skillrl', bundle)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / 'work' / 'inference_skills.json'
    report = module.prepare(bundle, ROOT / 'data/training_runtime_20261106_v1', output)
    assert read_json(output)['bank_path'] == str(bundle / 'bank.json')
    assert (report['skills'], report['validation_tasks'], report['validation_tasks_with_skills']) == (173, 589, 147)
    assert report['training_evidence_checked'] and not report['evaluation_run']
    with pytest.raises(FileExistsError):
        module.prepare(bundle, ROOT / 'data/training_runtime_20261106_v1', output)
    (bundle / 'bank.json').write_text('{}')
    with pytest.raises(ValueError, match='skill_bank_hash_mismatch'):
        module.prepare(bundle, ROOT / 'data/training_runtime_20261106_v1', tmp_path / 'bad.json')
    assert not (tmp_path / 'bad.json').exists()


@pytest.mark.parametrize('split,task_split', [('validation', 'dev'), ('test', 'test')])
def test_paired_entry_uses_requested_split_and_same_model(tmp_path, monkeypatch, split, task_split):
    task = fixture(tmp_path / 'assets').model_copy(update={'split': task_split})
    write_json(Path(task.environment_ref).with_name('tools.json'), TOOLS)
    cfg = RunConfig(protocol_id='awm-single-json-v2', purpose='evaluation', actor='training',
                    judge='api', judge_protocol='rubric_v2', isolation='landlock', context_limit=32768)
    write_json(tmp_path / 'config.json', cfg.model_dump())
    bank = tmp_path / 'bank.json'
    write_json(bank, {'schema_version': 'awm-skillbank-v1', 'source_split': 'grpo_train', 'skills': [example_skill()]})
    write_json(tmp_path / 'skills.json', {'bank_path': str(bank), 'bank_sha256': file_hash(bank),
                                        'top_k': 3, 'max_context_bytes': 5000})
    calls = []
    def tasks(path, selected_split):
        assert selected_split == split
        return [task], {}, None
    def preflight(tasks, config, selected_split):
        assert selected_split == split and tasks[0].split == task_split
    monkeypatch.setattr(skill_eval, 'load_tasks', tasks)
    monkeypatch.setattr(skill_eval, 'preflight', preflight)
    monkeypatch.setattr(skill_eval, 'checkpoint_identity', lambda _: {'sha256': 'frozen-checkpoint'})
    model, tokenizer = object(), object()
    monkeypatch.setattr(skill_eval, 'load_checkpoint', lambda *a, **k: (model, tokenizer))
    async def evaluate(m, tok, tasks, config, output, **kwargs):
        assert m is model and tok is tokenizer
        assert kwargs['split'] == split and kwargs['sampling'] == 'greedy'
        assert kwargs['checkpoint']['sha256'] == 'frozen-checkpoint'
        calls.append(config.skills is not None)
        write_json(output / 'episodes/t00000/a0/episode_result.json',
                   {'task_instance_id': task.task_instance_id, 'evaluation_valid': True,
                    'task_status': 'confirmed_pass'})
    monkeypatch.setattr(skill_eval, 'evaluate', evaluate)
    out = tmp_path / 'paired'
    monkeypatch.setattr(sys, 'argv', ['skill_eval', '--config', str(tmp_path / 'config.json'),
        '--tasks', str(tmp_path), '--model', str(tmp_path / 'model'), '--skills', str(tmp_path / 'skills.json'),
        '--split', split, '--limit', '0', '--output', str(out), '--execute'])
    skill_eval.main()
    assert calls == [False, True]
    assert read_json(out / 'plan.json')['split'] == split
    report = read_json(out / 'comparison.json')
    assert report['split'] == split and report['planned_pairs'] == 1
    assert report['success_over_planned'] == {'baseline': 1.0, 'skills': 1.0}
