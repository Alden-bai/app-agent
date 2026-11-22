"""Verify an extracted delivery or stream-verify its archive without loading models."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def hash_stream(stream):
    h = hashlib.sha256()
    for part in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(part)
    return h.hexdigest()


def verify(target):
    target = Path(target)
    if target.is_file():
        actual = {}
        manifest = None
        prefix = None
        with tarfile.open(target, "r|gz") as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                if prefix is None and path.parts:
                    prefix = path.parts[0]
                if (path.is_absolute() or ".." in path.parts or not member.isfile() or len(path.parts) < 2
                        or prefix not in {'awm-qwen35-27b', 'awm-nioflow-cu128', 'awm-nioflow-cu128-grpo2000'}
                        or path.parts[0] != prefix):
                    raise ValueError("unsafe archive member: " + member.name)
                name = str(PurePosixPath(*path.parts[1:]))
                if name in actual or (name == "DELIVERY_MANIFEST.json" and manifest is not None):
                    raise ValueError("duplicate archive member")
                with archive.extractfile(member) as f:
                    if name == "DELIVERY_MANIFEST.json":
                        manifest = json.load(f)
                    else:
                        actual[name] = {"bytes": member.size, "sha256": hash_stream(f)}
        if manifest is None:
            raise ValueError("delivery manifest missing")
        expected = {e["path"]: {k: e[k] for k in ("bytes", "sha256")} for e in manifest["files"]}
        if actual != expected:
            raise ValueError("archive entries/checksums differ from manifest")
    else:
        manifest = json.loads((target / "DELIVERY_MANIFEST.json").read_text())
        for entry in manifest["files"]:
            path = target / entry["path"]
            if not path.resolve().is_relative_to(target.resolve()) or path.is_symlink():
                raise ValueError("delivery reference escapes root")
            with path.open("rb") as f:
                if path.stat().st_size != entry["bytes"] or hash_stream(f) != entry["sha256"]:
                    raise ValueError("delivery checksum mismatch: " + entry["path"])
    if manifest["weights_included"] is not False:
        raise ValueError("unexpected weights flag")
    return {"passed": True, "files": len(manifest["files"]), "bytes": manifest["total_bytes"],
            "weights_included": False, "gpu_validation": manifest["gpu_validation"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("target", type=Path)
    print(json.dumps(verify(p.parse_args().target), indent=2))
