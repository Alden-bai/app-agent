from pathlib import Path
import pytest
from harness_awm.io import write_json, read_json, file_hash, verify_seal
from scripts.sft_eval_local_recovery import archive_episode, retry_reason


def test_complete_result_overrides_queue_storage_error():
    result = {'evaluation_valid': True, 'task_status': 'confirmed_pass'}
    assert retry_reason(result, {'state': 'error'}, 'database is locked') is None
    assert retry_reason({'evaluation_valid': False, 'error': None}, {'state': 'done'}, '') is None


def test_only_evidenced_infrastructure_failure_is_retried():
    assert retry_reason(None, {'state': 'held_storage'}, '') == 'unstarted'
    assert retry_reason(None, {'state': 'error'}, 'OperationalError: database is locked') == 'recorded_sqlite_lock_without_result'
    with pytest.raises(ValueError, match='without proven'):
        retry_reason(None, {'id': 3, 'state': 'claimed'}, '')
    oom = {'evaluation_valid': False, 'execution_integrity': 'runtime_error',
           'termination_reason': 'runtime_error', 'error': 'CUDA out of memory'}
    assert retry_reason(oom, {'state': 'done'}, '') == 'recorded_cuda_oom'


def test_archive_preserves_bytes_and_seal_containment(tmp_path):
    local, archive = tmp_path / 'local/a0', tmp_path / 'remote/a0'
    write_json(local / 'episode_result.json', {'evaluation_valid': True})
    write_json(local / 'model_turns/0.json', {'prompt': 'example'})
    write_json(local / 'evidence_seal.json', {'model_turns/0.json': file_hash(local / 'model_turns/0.json')})
    expected = file_hash(local / 'model_turns/0.json')
    archive_episode(local, archive)
    assert not (local / 'episode_result.json').is_symlink()
    assert local.is_symlink()
    assert (local / 'model_turns/0.json').resolve().is_relative_to(local.resolve())
    assert file_hash(local / 'model_turns/0.json') == expected
    assert verify_seal(local)['model_turns/0.json'] == expected
    assert (archive / 'episode_result.json').exists()
    archive_episode(local, archive)


def test_archive_mismatch_preserves_original(tmp_path):
    local, archive = tmp_path / 'local', tmp_path / 'archive'
    write_json(local / 'episode_result.json', {'value': 1})
    write_json(archive / 'episode_result.json', {'value': 2})
    with pytest.raises(RuntimeError, match='differs'):
        archive_episode(local, archive)
    assert read_json(local / 'episode_result.json')['value'] == 1
