"""Install independent A100 environments; never reuse paths from the source host."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("profile", choices=("sft", "grpo", "env"))
    p.add_argument("--venv", type=Path)
    a = p.parse_args()
    if sys.version_info[:2] != (3, 12):
        p.error("Use Python 3.12")
    folder = (a.venv or ROOT / f".venv-delivery-{a.profile}").absolute()
    folder.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(folder.parent).free < (1 if a.profile == "env" else 30) * 2**30:
        p.error("Insufficient free disk: env needs 1 GiB; each training environment needs 30 GiB")
    python = str(folder / "bin/python")
    if not (folder / "pyvenv.cfg").is_file():
        venv.EnvBuilder(with_pip=True, symlinks=True).create(folder)
    else:
        subprocess.run([python, "-m", "ensurepip", "--upgrade"], check=True)
    env = dict(os.environ, PIP_NO_CACHE_DIR="1", TOKENIZERS_PARALLELISM="false")
    def pip(*args):
        subprocess.run([python, "-m", "pip", *map(str, args)], check=True, env=env)
    pip("install", "pip==25.3", "wheel", "setuptools==80.10.2")
    if a.profile == "env":
        pip("install", "-r", ROOT / "requirements/delivery-env.lock.txt",
            "-e", ROOT / "agent-world-model", "-e", str(ROOT) + "[test]")
    else:
        lock = json.loads((ROOT / "requirements/verl-runtime.json").read_text())
        source = ROOT / "vendor/verl"
        for name, sha in lock["source_sha256"].items():
            if hashlib.sha256((source / "verl" / name).read_bytes()).hexdigest() != sha:
                raise ValueError("vendored veRL source checksum mismatch: " + name)
        build = folder / "build/verl"
        if build.exists():
            shutil.rmtree(build)
        shutil.copytree(source, build, ignore=shutil.ignore_patterns(".git", "__pycache__"))
        if a.profile == "grpo":
            setup = build / "setup.py"
            old = setup.read_text()
            if '"numpy<2.0.0"' not in old:
                raise ValueError("upstream numpy metadata changed")
            setup.write_text(old.replace('"numpy<2.0.0"', '"numpy==2.3.5"'))
            (build / "verl/version/version").write_text("0.8.0.dev0+awm.numpy2\n")
        # One resolver transaction applies all pins to veRL's transitive dependencies.
        pip("install", "-r", ROOT / f"requirements/delivery-{a.profile}.lock.txt", build,
            "-e", str(ROOT) + "[test]")
        code = ("from harness_awm.training.verl.sft_compat import verify_sft_runtime; verify_sft_runtime()"
                if a.profile == "sft" else
                "from harness_awm.training.verl.__main__ import verify_runtime; verify_runtime('local-sdpa')")
        subprocess.run([python, "-c", code], cwd=ROOT, check=True, env=env)
    pip("check")
    (folder / "installed.freeze.txt").write_text(subprocess.check_output([python, "-m", "pip", "freeze"], text=True))
    print(f"Installed {a.profile}: {python}. GPU execution still requires preflight and smoke tests.")


if __name__ == "__main__":
    main()
