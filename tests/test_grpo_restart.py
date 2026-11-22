from pathlib import Path
import pytest
from harness_awm.io import file_hash, read_json, seal, verify_seal, write_json
import importlib.util
spec = importlib.util.spec_from_file_location('awm_restart_grpo', Path(__file__).parents[1] / 'scripts/restart_grpo_local.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
stage_admissions = module.stage_admissions


def fixture_admission(root):
    control = root / 'control'
    write_json(control / 'task.json', {'task': 'preserve original bytes'})
    seal(control)
    proof = root / 'calibration.json'
    write_json(proof, {'controls': [{'path': str(control)}],
                      'trajectory_controls': [{'path': str(control)}]})
    manifest = root / 'reward_admissions.json'
    write_json(manifest, {'tasks': {'example': {'calibration_ref': str(proof),
                                             'calibration_sha256': file_hash(proof)}}})
    return manifest, control, proof


def test_stage_preserves_evidence_and_relocates_only_references(tmp_path):
    manifest, control, proof = fixture_admission(tmp_path / 'source')
    old_proof = proof.read_bytes()
    staged = stage_admissions(manifest, tmp_path / 'local')
    entry = read_json(staged)['tasks']['example']
    local_proof = read_json(entry['calibration_ref'])
    assert file_hash(entry['calibration_ref']) == entry['calibration_sha256']
    local = Path(local_proof['controls'][0]['path'])
    assert local.is_relative_to(tmp_path / 'local')
    assert local_proof['trajectory_controls'][0]['path'] == str(local)
    assert verify_seal(local) == verify_seal(control)
    assert (local / 'task.json').read_bytes() == (control / 'task.json').read_bytes()
    assert proof.read_bytes() == old_proof


def test_stage_rejects_corrupt_evidence(tmp_path):
    manifest, control, _ = fixture_admission(tmp_path / 'source')
    write_json(control / 'task.json', {'task': 'changed'})
    with pytest.raises(ValueError, match='evidence hash mismatch'):
        stage_admissions(manifest, tmp_path / 'local')


def test_stage_rejects_modified_calibration(tmp_path):
    manifest, _, proof = fixture_admission(tmp_path / 'source')
    write_json(proof, {'controls': []})
    with pytest.raises(ValueError, match='calibration hash mismatch'):
        stage_admissions(manifest, tmp_path / 'local')
