"""Build a weight-free archive from explicit sources, without copying all runs/."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SPLIT = "data/training_split_20261106_v1"
RUNTIME = "data/training_runtime_20261106_v1"
REVISION = "7aed6b230776f963fa09509c10d9c3a767d1102c"
IGNORED = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules"}
WEIGHT = re.compile(r"(?:\.safetensors$|\.gguf$|\.pt$|\.pth$|\.ckpt$|pytorch_model.*\.bin$|optimizer.*\.bin$)", re.I)
SECRET = re.compile(rb"(?:sk-(?:proj-)?[A-Za-z0-9_-]{32,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def fetch_verl():
    dest = ROOT / "vendor/verl"
    lock = json.loads((ROOT / "requirements/verl-runtime.json").read_text())
    if not dest.exists():
        dest.mkdir(parents=True)
        url = f"https://codeload.github.com/verl-project/verl/tar.gz/{REVISION}"
        with urllib.request.urlopen(url, timeout=180) as response, tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                parts = PurePosixPath(member.name).parts[1:]
                if not parts or ".." in parts or member.issym() or member.islnk():
                    continue
                target = dest.joinpath(*parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as src, target.open("wb") as out:
                        import shutil
                        shutil.copyfileobj(src, out)
    for name, expected in lock["source_sha256"].items():
        if sha(dest / "verl" / name) != expected:
            raise ValueError("vendored veRL differs from pinned revision: " + name)
    (dest / "UPSTREAM_REVISION").write_text(REVISION + "\n")


def allowed(name):
    parts = PurePosixPath(name).parts
    return (not any(p in IGNORED or p.startswith(".venv") for p in parts)
            and not any(p.endswith((".egg-info", ".dist-info")) for p in parts)
            and not WEIGHT.search(name) and not name.endswith((".pyc", ".pyo"))
            and Path(name).name not in {".env", ".git", "credentials", "id_rsa", "id_ed25519"})


def inventory(calibration_run):
    files = {}
    def add(source, name=None, expected=None):
        source = Path(source)
        name = str(name or source.relative_to(ROOT))
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or not allowed(name):
            raise ValueError("excluded or unsafe delivery member: " + name)
        if not source.is_file():
            raise FileNotFoundError(source)
        if name in files and files[name][0].resolve() != source.resolve():
            raise ValueError("delivery path collision: " + name)
        files[name] = (source, expected)
    def tree(folder, prefix=None):
        folder = Path(folder)
        for base, dirs, names in os.walk(folder, followlinks=False):
            dirs[:] = [n for n in sorted(dirs) if n not in IGNORED and not n.startswith(".venv")]
            for name in sorted(names):
                path = Path(base) / name
                rel = str(Path(prefix) / path.relative_to(folder)) if prefix else str(path.relative_to(ROOT))
                if allowed(rel):
                    add(path, rel)
    for name in ("harness_awm", "scripts", "tests", "configs", "requirements", "docs", "vendor/verl", "assets/qwen35-27b", "delivery"):
        tree(ROOT / name)
    for name in ("pyproject.toml", "README.md"):
        add(ROOT / name)
    for name in ("awm",):
        tree(ROOT / "agent-world-model" / name)
    for name in ("pyproject.toml", "README.md"):
        add(ROOT / "agent-world-model" / name)
    for path in (ROOT / "agent-world-model").glob("LICENSE*"):
        add(path)
    tree(ROOT / SPLIT)
    tree(ROOT / RUNTIME)
    tree(ROOT / "data/training_qwen35_27b_v1")
    split = json.loads((ROOT / SPLIT / "manifest.json").read_text())
    sources = json.loads((ROOT / SPLIT / "source_hashes.json").read_text())
    for ref, expected in sources.items():
        rel = Path(ref).relative_to(split["source_root"])
        add(ROOT / rel, str(rel), expected)
    # Preserve old calibration bytes, seals and provenance; never certify them
    # for changed Judge code. Fresh calibration is a separate target-host command.
    calibration_run = Path(calibration_run)
    suite = json.loads((calibration_run / "admissions-shared-calibration.json").read_text())
    for name in ("admissions-shared-calibration.json", "admissions-reward_admissions.json", "admissions-audit.json"):
        add(calibration_run / name, "calibration/history/" + name)
    for entry in suite["calibrated_controls"]:
        proof = Path(entry["calibration_ref"])
        if sha(proof) != entry["calibration_sha256"]:
            raise ValueError("calibration proof changed: " + str(proof))
        tree(proof.parent, "calibration/history/controls/" + proof.parent.name)
        report = json.loads(proof.read_text())
        for control in report["controls"] + report.get("trajectory_controls", []):
            episode = Path(control["path"])
            if not episode.is_dir():
                raise FileNotFoundError("missing calibration episode: " + str(episode))
            tree(episode, "calibration/history/episodes/" + proof.parent.name + "/" + episode.name)
    return files


def build(output, calibration_run):
    files = inventory(calibration_run)
    # Read credential values only to detect accidental inclusion. Never print them.
    credentials = []
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            key, sep, value = line.partition("=")
            value = value.strip().strip("\"'")
            if sep and any(x in key.upper() for x in ("KEY", "TOKEN", "SECRET", "PASSWORD")) and len(value) >= 16:
                credentials.append(value.encode())
    entries = []
    for name, (path, expected) in sorted(files.items()):
        digest = sha(path)
        if expected and digest != expected:
            raise ValueError("frozen source hash mismatch: " + name)
        # Scan all bytes, including DBs; overlap chunks for tokens spanning blocks.
        with path.open("rb") as f:
            tail = b""
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                data = tail + chunk
                if SECRET.search(data) or any(token in data for token in credentials):
                    raise ValueError("credential-like content in delivery member: " + name)
                tail = data[-4096:]
        entries.append({"path": name, "bytes": path.stat().st_size, "sha256": digest})
    manifest = {"schema_version": "awm-training-delivery-v1", "model": "Qwen/Qwen3.5-27B",
        "hardware": "1 node, 8 x A100 80GB", "weights_included": False,
        "files": entries, "total_bytes": sum(e["bytes"] for e in entries),
        "calibration_status": "historical_only; fresh calibration required for current Judge implementation",
        "gpu_validation": "pending on target hardware",
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_status": subprocess.check_output(["git", "status", "--short", "--untracked-files=normal"], cwd=ROOT, text=True).splitlines()}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    partial = output.with_suffix(output.suffix + ".partial")
    with tarfile.open(partial, "w:gz", compresslevel=6) as archive:
        for entry in entries:
            path = files[entry["path"]][0]
            if sha(path) != entry["sha256"]:
                raise ValueError("source changed while packaging: " + entry["path"])
            info = tarfile.TarInfo("awm-qwen35-27b/" + entry["path"])
            info.size, info.mode, info.mtime = entry["bytes"], 0o644, 0
            with path.open("rb") as f:
                archive.addfile(info, f)
        data = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
        info = tarfile.TarInfo("awm-qwen35-27b/DELIVERY_MANIFEST.json")
        info.size, info.mode = len(data), 0o644
        archive.addfile(info, io.BytesIO(data))
    partial.rename(output)
    output.with_suffix(output.suffix + ".sha256").write_text(sha(output) + "  " + output.name + "\n")
    output.with_suffix(output.suffix + ".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"archive": str(output), "files": len(entries), "uncompressed_bytes": manifest["total_bytes"],
                      "archive_bytes": output.stat().st_size, "sha256": sha(output)}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "dist/awm-qwen35-27b-a100x8-delivery.tar.gz")
    p.add_argument("--calibration-run", type=Path, required=True)
    p.add_argument("--fetch-verl", action="store_true")
    p.add_argument("--inventory-only", action="store_true")
    a = p.parse_args()
    if a.fetch_verl:
        fetch_verl()
    if a.inventory_only:
        files = inventory(a.calibration_run)
        print(json.dumps({"files": len(files), "bytes": sum(p.stat().st_size for p, _ in files.values())}))
    else:
        build(a.output, a.calibration_run)


if __name__ == "__main__":
    main()
