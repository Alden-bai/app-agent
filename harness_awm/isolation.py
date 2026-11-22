"""Linux filesystem isolation. Networking is loopback/shared, not a network sandbox."""
import asyncio
import json
import os
import shutil
import signal
import subprocess
from pathlib import Path


def runtime_info(python):
    import json
    output = subprocess.check_output([python, "-c", "import json,sys; print(json.dumps([sys.prefix,sys.base_prefix]))"], text=True)
    return [Path(p).resolve() for p in json.loads(output)]


def private_root_command(config, folder, operation, *args):
    status = Path('/proc/self/status').read_text().splitlines()
    caps = int(next(line.split()[1] for line in status if line.startswith('CapEff:')), 16)
    prefix = [] if caps & (1 << 18) else ['unshare', '--user', '--map-root-user', '--']
    return [*prefix, config.runtime_python, '-I', '-B', str(Path(__file__).with_name('private_root.py')),
            operation, str(folder), *map(str, args)]


def command(config, argv, writable, readonly=()):
    if config.isolation == "development":
        return argv
    # Never bind /, /home, the repository root, run root or credentials directory.
    roots = [Path("/usr"), Path("/lib"), Path("/lib64"), *runtime_info(config.runtime_python)]
    # uv venv interpreters can point through an additional version-alias directory.
    interpreter = Path(config.runtime_python)
    if interpreter.is_symlink():
        link = Path(os.readlink(interpreter))
        if not link.is_absolute():
            link = interpreter.parent / link
        roots.append(link.parent.parent.absolute())
    roots += [Path(p).absolute() for p in readonly]
    if config.isolation == "landlock":
        writable = [str(Path(p).resolve()) for p in writable]
        scratch = Path(writable[0]) / ".tmp"
        scratch.mkdir(mode=0o700, exist_ok=True)
        roots += [Path("/dev/urandom"), Path("/dev/random")]
        policy = {"readonly": [str(p) for p in dict.fromkeys(roots) if p.exists()],
                  "writable": [*writable, "/dev/null"],
                  "cwd": writable[0], "tmp": str(scratch),
                  # Python discovers awm by listing its parent; grant no file reads there.
                  "listdirs": [str(Path(config.official_root))] if Path(config.official_root) / "awm" in roots else []}
        return [config.runtime_python, str(Path(__file__).with_name("landlock.py")),
                json.dumps(policy), *argv]
    if config.isolation != "strict":
        raise ValueError("unknown isolation backend")
    bwrap = shutil.which("bwrap") or "bwrap"
    cmd = [bwrap, "--die-with-parent", "--new-session", "--unshare-user", "--unshare-pid",
           "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for path in dict.fromkeys(roots):
        if path.exists():
            cmd += ["--ro-bind", str(path), str(path)]
    for path in writable:
        path = Path(path).resolve()
        cmd += ["--bind", str(path), str(path)]
    cmd += ["--chdir", str(Path(writable[0]).resolve()), "--", *argv]
    return cmd


def clean_env(timezone):
    return {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "TZ": timezone, "OPENBLAS_NUM_THREADS": "1",
            "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "HOME": "/tmp"}


async def wait_process(process, timeout):
    # Popen.poll uses waitpid; works on hosts where asyncio pidfd notifications are restricted.
    async with asyncio.timeout(timeout):
        while process.poll() is None:
            await asyncio.sleep(.05)
    return process.returncode


async def stop_process(process, timeout=5):
    if process is None:
        return True
    # Kill the entire session group even if its original leader already exited.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            await wait_process(process, timeout)
        except asyncio.TimeoutError:
            continue
        # bwrap's PID namespace destroys descendants when its init exits.
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return True
    return False

