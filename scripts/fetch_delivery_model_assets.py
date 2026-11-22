"""Download ONLY pinned non-weight Qwen3.5-27B assets (standard library)."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def fetch(url):
    with urllib.request.urlopen(url, timeout=180) as response:
        return response.read()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("assets/qwen35-27b"))
    a = p.parse_args()
    lock_path = a.output / "model-assets.lock.json"
    repo = "Qwen/Qwen3.5-27B"
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else None
    revision = lock["revision"] if lock else json.loads(fetch(f"https://huggingface.co/api/models/{repo}"))["sha"]
    a.output.mkdir(parents=True, exist_ok=True)
    records = {}
    for name in FILES:
        path = a.output / name
        data = path.read_bytes() if lock and path.exists() else fetch(f"https://huggingface.co/{repo}/resolve/{revision}/{name}")
        checksum = hashlib.sha256(data).hexdigest()
        if lock and checksum != lock["files"][name]["sha256"]:
            raise ValueError("asset checksum mismatch: " + name)
        path.write_bytes(data)
        records[name] = {"sha256": checksum, "bytes": len(data)}
    lock_path.write_text(json.dumps({"repository": repo, "revision": revision, "weights_included": False,
        "files": records}, indent=2) + "\n")
    print(json.dumps({"revision": revision, "bytes": sum(r["bytes"] for r in records.values()), "weights_downloaded": False}))


if __name__ == "__main__":
    main()
