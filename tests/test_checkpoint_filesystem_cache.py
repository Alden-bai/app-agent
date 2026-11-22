"""Verified plans avoid duplicate network reads without ignoring model changes."""
import os

import pytest

from harness_awm.io import digest, write_json
from harness_awm.training.local import checkpoint_filesystem, checkpoint_identity
from harness_awm.training.verl import VERL_REVISION
from harness_awm.training.verl import __main__ as cli


def frozen_plan(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"weights")
    run = tmp_path / "run"
    run.mkdir()
    plan = {"output": str(run), "verl_revision": VERL_REVISION, "model": str(model),
            "checkpoint": checkpoint_identity(model), "checkpoint_filesystem": checkpoint_filesystem(model),
            "harness_source_sha256": {}}
    write_json(run / "plan.json", plan)
    write_json(run / "plan.sha256.json", {"sha256": digest(plan)})
    return run, model, plan


def test_unchanged_frozen_checkpoint_does_not_rehash_weights(tmp_path, monkeypatch):
    run, _, expected = frozen_plan(tmp_path)
    def unnecessary_read(path):
        raise AssertionError("unchanged weights should not be read again")
    monkeypatch.setattr(cli, "checkpoint_identity", unnecessary_read)
    assert cli.load_plan(run) == expected


@pytest.mark.parametrize("change", ["same_size_restored_mtime", "replacement", "new_shard"])
def test_changed_checkpoint_cannot_use_cached_identity(tmp_path, change):
    run, model, _ = frozen_plan(tmp_path)
    path = model / "model.safetensors"
    before = path.stat()
    if change == "replacement":
        path.unlink()
    if change == "new_shard":
        (model / "model-extra.safetensors").write_bytes(b"extra")
    else:
        path.write_bytes(b"changed")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match="checkpoint changed"):
        cli.load_plan(run)
