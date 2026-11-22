"""Fetch checksum-pinned Qwen assets from the model author's ModelScope repo."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(lock, output, weights=False):
    manifest = json.loads(Path(lock).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    def fetch(row):
        name = row["name"]
        if Path(name).name != name:
            raise ValueError("model asset name must not contain a directory")
        path = output / name
        if path.exists() and digest(path) == row["sha256"]:
            return name + ": cached and verified"
        url = f"https://modelscope.cn/models/{manifest['repo_id']}/resolve/{row['revision']}/{name}"
        part = path.with_name(path.name + ".part")
        if part.exists() and part.stat().st_size == row["size"] and digest(part) == row["sha256"]:
            part.replace(path)
            return name + ": resumed completed download and verified"
        for attempt in range(3):
            try:
                offset = part.stat().st_size if part.exists() else 0
                req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
                with urllib.request.urlopen(req, timeout=60) as response:
                    if offset and response.status == 206:
                        if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise ValueError("unexpected partial response")
                        mode = "ab"
                    else:
                        mode = "wb"
                    with part.open(mode) as stream:
                        while block := response.read(1024 * 1024):
                            stream.write(block)
                if part.stat().st_size != row["size"] or digest(part) != row["sha256"]:
                    part.unlink()
                    raise ValueError(f"model checksum mismatch: {name}")
                part.replace(path)
                return name + ": downloaded and verified"
            except Exception:
                if attempt == 2:
                    raise

    selected = [r for r in manifest["files"] if weights or not r["name"].endswith(".safetensors")]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for message in pool.map(fetch, selected):
            print(message, flush=True)
    (output / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lock", type=Path, default=Path("configs/training/qwen3.5-2b-model.lock.json"))
    p.add_argument("--output", type=Path, default=Path("models/Qwen3.5-2B"))
    p.add_argument("--weights", action="store_true")
    a = p.parse_args()
    download(a.lock, a.output, a.weights)
