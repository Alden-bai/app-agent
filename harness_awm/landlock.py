"""Namespace-free filesystem restrictions for evaluation workers.

ABI 1 cannot mediate truncate/chmod/stat, and there is no PID or network
namespace. This backend is explicitly distinct from strict isolation.
Executed as a standalone, single-threaded launcher before worker imports.
"""
import ctypes
import json
import os
import platform
import sys
from pathlib import Path


class Ruleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class PathRule(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def _libc():
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "aarch64"}:
        raise RuntimeError("Landlock launcher supports Linux x86_64/aarch64 only")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    return libc


def _checked(value):
    if value < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return value


def abi_version():
    return _checked(_libc().syscall(444, 0, 0, 1))


def restrict(readonly, writable, listdirs=()):
    libc = _libc()
    abi = abi_version()
    if abi < 1:
        raise RuntimeError("Landlock unavailable")
    # Handle all filesystem rights through ABI 3; never assume newer support.
    handled = (1 << (15 if abi >= 3 else 14 if abi >= 2 else 13)) - 1
    attr = Ruleset(handled)
    ruleset = _checked(libc.syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0))
    try:
        for paths, access in ((readonly, 1 | 4 | 8), (writable, handled), (listdirs, 8)):
            for name in dict.fromkeys(paths):
                path = Path(name).resolve(strict=True)
                allowed = access
                if not path.is_dir():
                    allowed &= 1 | 2 | 4 | ((1 << 14) if abi >= 3 else 0)
                fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
                try:
                    rule = PathRule(allowed, fd)
                    _checked(libc.syscall(445, ruleset, 1, ctypes.byref(rule), 0))
                finally:
                    os.close(fd)
        _checked(libc.prctl(38, 1, 0, 0, 0))  # PR_SET_NO_NEW_PRIVS
        _checked(libc.syscall(446, ruleset, 0))
    finally:
        os.close(ruleset)
    return abi


def main():
    policy = json.loads(sys.argv[1])
    argv = sys.argv[2:]
    if not argv:
        raise ValueError("missing worker command")
    os.chdir(policy["cwd"])
    os.umask(0o077)
    for key in ("HOME", "TMPDIR", "TMP", "TEMP"):
        os.environ[key] = policy["tmp"]
    restrict(policy["readonly"], policy["writable"], policy.get("listdirs", []))
    os.execv(argv[0], argv)


if __name__ == "__main__":
    main()
