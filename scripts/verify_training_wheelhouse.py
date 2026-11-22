"""Validate a complete Linux/Python wheelhouse and freeze exact offline wheels."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--wheelhouse", type=Path, required=True)
    a = p.parse_args()
    requirements = {}
    for line in a.lock.read_text().replace("\\\n", " ").splitlines():
        match = re.match(r"([\w.-]+)==([^\s]+)", line)
        if match:
            requirements[canonicalize_name(match[1])] = (match[2], set(re.findall(r"--hash=sha256:([0-9a-f]+)", line)))
    compatible = set(sys_tags())
    wheels = {}
    for path in a.wheelhouse.glob("*.whl"):
        name, version, _, tags = parse_wheel_filename(path.name)
        if tags & compatible:
            wheels[(canonicalize_name(name), str(version))] = path
    output, manifest = [], {}
    for name, (version, hashes) in sorted(requirements.items()):
        path = wheels.get((name, version))
        if path is None:
            raise ValueError(f"missing compatible wheel: {name}=={version}")
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        digest = h.hexdigest()
        # pwinput 1.0.3 publishes only an sdist. Its locally built pure-Python
        # wheel receives a new hash; freeze that exact build for offline transfer.
        if digest not in hashes and name != "pwinput":
            raise ValueError(f"wheel hash not approved by dependency lock: {path}")
        output.append(f"{name}=={version} --hash=sha256:{digest}")
        manifest[path.name] = {"sha256": digest, "size": path.stat().st_size,
                               "locally_built": name == "pwinput" and digest not in hashes}
    destination = a.wheelhouse / a.lock.name
    destination.write_text("# Exact wheels for Python 3.12/Linux x86_64 offline installation.\n" + "\n".join(output) + "\n")
    destination.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"passed": True, "packages": len(output), "lock": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
