"""Exercise namespace-free restrictions in a child, never restrict pytest itself."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from harness_awm.contracts import RunConfig
from harness_awm.isolation import clean_env, command
from harness_awm.landlock import abi_version
from harness_awm.training.baseline import isolation_probe


def supported():
    try:
        assert abi_version() >= 1
    except (OSError, RuntimeError):
        pytest.skip('Landlock unavailable on this host')


def test_real_filesystem_policy_denies_outside_and_readonly_writes(tmp_path):
    supported()
    writable = tmp_path / 'worker'
    writable.mkdir()
    outside = tmp_path / 'outside.txt'
    outside.write_text('secret sentinel')
    readonly = tmp_path / 'input.txt'
    readonly.write_text('allowed')
    (writable / 'escape').symlink_to(outside)
    config = RunConfig(isolation='landlock', runtime_python=sys.executable)
    code = '''
import pathlib, sqlite3, sys, tempfile
outside, readonly = map(pathlib.Path, sys.argv[1:])
assert readonly.read_text() == 'allowed'
for path, mode in [(outside, 'r'), (outside, 'w'), (readonly, 'w'), (pathlib.Path('escape'), 'r')]:
    try:
        with path.open(mode):
            pass
    except PermissionError:
        pass
    else:
        raise AssertionError((path, mode))
with sqlite3.connect('working.db') as db:
    db.execute('CREATE TABLE sample (value INTEGER)')
with tempfile.NamedTemporaryFile() as f:
    assert pathlib.Path(f.name).parent == pathlib.Path.cwd() / '.tmp'
print('ok')
'''
    argv = command(config, [sys.executable, '-c', code, str(outside), str(readonly)], [writable], [readonly])
    run = subprocess.run(argv, env=clean_env(config.timezone), capture_output=True, text=True, timeout=20)
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == 'ok'
    assert outside.read_text() == 'secret sentinel'
    assert readonly.read_text() == 'allowed'


def test_probe_checks_inherited_restrictions(tmp_path):
    supported()
    config = RunConfig(isolation='landlock', runtime_python=sys.executable)
    result = isolation_probe(config, tmp_path / 'probe')
    assert result['passed'], result
    assert result['landlock_abi'] >= 1
    assert result['pid_namespace'] is False


def test_unavailable_landlock_fails_closed(tmp_path, monkeypatch):
    def unavailable():
        raise OSError('not supported')
    monkeypatch.setattr('harness_awm.landlock.abi_version', unavailable)
    config = RunConfig(isolation='landlock', runtime_python=sys.executable)
    result = isolation_probe(config, tmp_path)
    assert not result['passed']
    assert 'not supported' in result['error']


def test_landlock_is_explicit_and_does_not_enable_training_rewards():
    assert RunConfig().isolation == 'strict'
    config = RunConfig(purpose='evaluation', actor='training', judge='api',
                       judge_protocol='rubric_v2', isolation='landlock')
    assert config.isolation == 'landlock'
    with pytest.raises(ValueError, match='formal runs require strict'):
        RunConfig(purpose='formal', actor='training', judge='api', isolation='landlock')
    with pytest.raises(ValueError, match='scoring-only'):
        RunConfig(**{**config.model_dump(), 'reward_admissions': 'admission.json'})
