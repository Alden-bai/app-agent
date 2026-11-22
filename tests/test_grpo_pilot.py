import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from harness_awm.io import write_json, read_json
from harness_awm.training.verl.pilot import archive_group, group_health


def test_background_archive_overlaps_work_and_manifest_follows_verification(tmp_path, monkeypatch):
    from harness_awm.training.verl import pilot
    started, release = threading.Event(), threading.Event()
    original = pilot.archive_group
    def delayed(*args):
        started.set()
        assert release.wait(5)
        return original(*args)
    monkeypatch.setattr(pilot, 'archive_group', delayed)
    source, dest = tmp_path / 'group', tmp_path / 'archive/group.tar.gz'
    write_json(source / 'evidence.json', {'value': 'evidence'})
    archiver = pilot.GroupArchiver()
    try:
        assert archiver.submit(source, dest) == str(dest)
        assert started.wait(5)
        archiver.wait_for_slot()  # Another rollout can start while archival is blocked.
        assert not dest.with_suffix('.manifest.json').exists()
        assert (source / 'evidence.json').exists()
    finally:
        release.set()
        archiver.close()
    assert read_json(dest.with_suffix('.manifest.json'))['files_verified'] == 1


def test_background_archive_bounds_backlog(tmp_path, monkeypatch):
    from harness_awm.training.verl import pilot
    started, release, waiting, slot = (threading.Event() for _ in range(4))
    def delayed(*args):
        started.set()
        assert release.wait(5)
    monkeypatch.setattr(pilot, 'archive_group', delayed)
    archiver = pilot.GroupArchiver(max_pending=1)
    def request_slot():
        waiting.set()
        archiver.wait_for_slot()
        slot.set()
    with ThreadPoolExecutor(max_workers=1) as caller:
        future = None
        try:
            archiver.submit(tmp_path, tmp_path / 'archive.tar.gz')
            assert started.wait(5)
            future = caller.submit(request_slot)
            assert waiting.wait(5)
            assert not slot.wait(.05)
        finally:
            release.set()
            if future is not None:
                future.result(timeout=5)
            archiver.close()
    assert slot.is_set()


@pytest.mark.parametrize('drain', [False, True])
def test_background_archive_failure_is_propagated_and_originals_retained(tmp_path, monkeypatch, drain):
    from harness_awm.training.verl import pilot
    source, dest = tmp_path / 'group', tmp_path / 'archive.tar.gz'
    write_json(source / 'evidence.json', {'value': 1})
    def broken(*args):
        raise OSError('archive disk failed')
    monkeypatch.setattr(pilot, 'archive_group', broken)
    archiver = pilot.GroupArchiver(max_pending=1)
    archiver.submit(source, dest)
    try:
        with pytest.raises(OSError, match='archive disk failed'):
            archiver.close() if drain else archiver.wait_for_slot()
    finally:
        with pytest.raises(OSError, match='archive disk failed'):
            archiver.close()
    assert (source / 'evidence.json').exists()
    assert not dest.with_suffix('.manifest.json').exists()


def test_archive_preserves_original_and_checks_all_bytes(tmp_path):
    source = tmp_path / 'group-000001'
    write_json(source / 'nested' / 'result.json', {'value': '证据'})
    archive = tmp_path / 'archive' / 'group.tar.gz'
    archive_group(source, archive)
    assert (source / 'nested' / 'result.json').is_file()
    assert read_json(archive.with_suffix('.manifest.json'))['files_verified'] == 1
    with tarfile.open(archive) as handle:
        assert handle.extractfile('group-000001/nested/result.json').read() == (source / 'nested/result.json').read_bytes()


def test_majority_null_and_api_fault_are_distinct(tmp_path):
    for i in range(8):
        write_json(tmp_path / f'candidate-{i:03d}' / 'episode_result.json',
                   {'reward': None if i < 4 else 1.0})
    assert not group_health(tmp_path, 8)['quarantine']
    write_json(tmp_path / 'candidate-004/episode_result.json', {'reward': None})
    health = group_health(tmp_path, 8)
    assert not health['quarantine'] and not health['api_fault']
    write_json(tmp_path / 'candidate-000/episode_result.json', {'reward': None, 'errors': ['APITimeoutError']})
    assert group_health(tmp_path, 8)['api_fault']
    for i in range(5):
        write_json(tmp_path / f'candidate-{i:03d}/episode_result.json',
                   {'reward': None, 'terminal_state_status': 'untrusted'})
    assert group_health(tmp_path, 8)['quarantine']


def test_rollout_budget_reserves_actor_and_context_memory():
    import pytest
    from harness_awm.training.verl.pilot import rollout_fraction
    fraction = rollout_fraction(.82, total_gb=102.64, actor_gb=7.53, target_gb=85)
    assert fraction == pytest.approx((85 - 7.53 - 2) / 102.64)
    assert fraction * 102.64 + 7.53 + 2 == pytest.approx(85)
    assert rollout_fraction(.6, total_gb=102.64, actor_gb=7.53, target_gb=85) == .6
    with pytest.raises(ValueError):
        rollout_fraction(.82, total_gb=80, actor_gb=7.53, target_gb=85)
