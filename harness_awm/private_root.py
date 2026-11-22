"""Run a frozen V2 tool server in its own directory on mount-restricted Linux.

Only the trusted framework is imported before confinement. World code runs with
an empty capability set, a syscall allowlist and one pre-bound loopback listener.
"""
import ctypes as C
import errno
import importlib
import os
from pathlib import Path
import resource
import socket
import sys


def checked(value):
    if value < 0:
        error = C.get_errno()
        raise OSError(error, os.strerror(error))
    return value


class Comparison(C.Structure):
    _fields_ = [('arg', C.c_uint), ('op', C.c_uint), ('mask', C.c_uint64), ('value', C.c_uint64)]


def confine(root, keep=()):
    libc = C.CDLL(None, use_errno=True)
    lib = C.CDLL('libseccomp.so.2', use_errno=True)
    lib.seccomp_init.argtypes = [C.c_uint32]
    lib.seccomp_init.restype = C.c_void_p
    lib.seccomp_rule_add_array.argtypes = [C.c_void_p, C.c_uint32, C.c_int, C.c_uint, C.POINTER(Comparison)]
    lib.seccomp_load.argtypes = lib.seccomp_release.argtypes = [C.c_void_p]
    # Unknown calls fail closed, including new kernel interfaces. ENOSYS also
    # lets newer glibc versions fall back from clone3 to filtered pthread clone.
    policy = lib.seccomp_init(0x00050000 | errno.ENOSYS)
    if not policy:
        raise MemoryError('seccomp_init')

    def allow(name, comparison=None):
        nr = lib.seccomp_syscall_resolve_name(name.encode('ascii'))
        if nr < 0:  # not present in this architecture/libseccomp version
            return
        rc = lib.seccomp_rule_add_array(policy, 0x7fff0000, nr, int(comparison is not None),
                                        C.byref(comparison) if comparison is not None else None)
        if rc != 0:
            raise OSError(-rc, f'seccomp: {name}')

    for name in ('''read write readv writev pread64 pwrite64 close close_range lseek open openat
        access faccessat faccessat2 stat lstat fstat newfstatat statx statfs fstatfs getdents64
        getcwd chdir mkdir mkdirat rmdir unlink unlinkat rename renameat renameat2 truncate
        ftruncate fsync fdatasync sync_file_range umask fcntl flock dup dup2 dup3 pipe pipe2
        mmap mprotect munmap mremap madvise brk getrandom futex futex_waitv set_robust_list rseq
        clock_gettime clock_getres gettimeofday time nanosleep clock_nanosleep sched_yield
        sched_getaffinity getcpu getpid getppid gettid getuid geteuid getgid getegid getgroups
        getrlimit uname sysinfo getrusage times getitimer setitimer restart_syscall
        rt_sigaction rt_sigprocmask rt_sigreturn sigaltstack exit exit_group
        poll ppoll select pselect6 epoll_create epoll_create1 epoll_ctl epoll_wait epoll_pwait
        eventfd2 listen accept accept4 getsockname getpeername getsockopt setsockopt shutdown
        recvfrom sendmsg recvmsg''').split():
        allow(name)
    # pthreads used by FastAPI sync endpoints. No process or namespace creation.
    pthread_flags = 0x013d0f00  # VM, FS, FILES, SIGHAND, THREAD, SYSVSEM, TLS, parent/child TID
    clone_thread = 0x00010000
    allow('clone', Comparison(0, 7, (0xffffffff ^ pthread_flags) | clone_thread, clone_thread))
    allow('socketpair', Comparison(0, 4, socket.AF_UNIX, 0))
    allow('tgkill', Comparison(0, 4, os.getpid(), 0))
    # Connected sockets only. No destination address can be supplied to sendto.
    allow('sendto', Comparison(4, 4, 0, 0))
    # Only read the process's resource limits.
    allow('prlimit64', Comparison(2, 4, 0, 0))
    # asyncio queries the number of pending bytes and nonblocking status.
    for request in (0x541b, 0x5421):
        allow('ioctl', Comparison(1, 4, request, 0))
    # Close framework-opened descriptors, including any external directories.
    if len(os.listdir('/proc/self/task')) != 1:
        raise RuntimeError('private-root bootstrap must be single-threaded')
    keep = set(keep) | {0, 1, 2}
    for entry in os.listdir('/proc/self/fd'):
        fd = int(entry)
        if fd not in keep:
            try:
                os.close(fd)
            except OSError as exc:
                if exc.errno != errno.EBADF:
                    raise
    os.chdir(root)
    os.chroot('.')
    os.chdir('/')
    os.umask(0o077)
    checked(libc.prctl(38, 1, 0, 0, 0))  # PR_SET_NO_NEW_PRIVS
    header = (C.c_uint32 * 2)(0x20080522, 0)
    checked(libc.capset(C.byref(header), C.byref((C.c_uint32 * 6)())))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    rc = lib.seccomp_load(policy)
    if rc:
        raise OSError(-rc, 'seccomp_load')
    lib.seccomp_release(policy)


def warm_server():
    # Only fixed, trusted module names. Never use world text to select imports.
    import asyncio
    from fastapi import FastAPI
    from fastapi_mcp import FastApiMCP
    import sqlalchemy
    import uvicorn
    import time
    from zoneinfo import ZoneInfo, available_timezones
    for module in '''ast base64 collections csv datetime decimal fractions hashlib httpx io ipaddress
        json math random secrets statistics string typing urllib.parse uuid zoneinfo encodings.idna _strptime
        anyio._backends._asyncio sqlalchemy.orm sqlalchemy.sql sqlalchemy.types sqlalchemy.ext.declarative sqlalchemy.dialects.sqlite
        sqlalchemy.pool uvicorn.loops.asyncio uvicorn.protocols.http.h11_impl uvicorn.lifespan.on'''.split():
        importlib.import_module(module)
    sqlalchemy.create_engine('sqlite:///:memory:').dispose()
    app = FastAPI()
    FastApiMCP(app).mount_http()
    cfg = uvicorn.Config(app, loop='asyncio', http='h11', ws='none')
    cfg.load()
    with asyncio.Runner() as runner:
        runner.run(asyncio.sleep(0))
    time.tzset()
    # Four frozen worlds accept user-supplied IANA zone names. Keep strong
    # references so ZoneInfo's weak cache can resolve them inside the root.
    return [ZoneInfo(key) for key in available_timezones()]


def main():
    kind, directory, *args = sys.argv[1:]
    root = Path(directory)
    if kind == 'probe':
        import sqlite3
        confine(root)
        outside = Path(args[0])
        for mode in ('r', 'w'):
            try:
                outside.open(mode).close()
            except (PermissionError, FileNotFoundError):
                pass
            else:
                raise RuntimeError('outside access permitted')
        try:
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except OSError:
            pass
        else:
            raise RuntimeError('new network socket permitted')
        with sqlite3.connect('/probe.sqlite') as db:
            db.execute('CREATE TABLE probe(value)')
            db.execute('INSERT INTO probe VALUES (1)')
        print('private_root_ok')
    elif kind == 'server':
        zones = warm_server()
        import uvicorn
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(('127.0.0.1', int(os.environ['PORT'])))
        listener.listen(128)
        original_run = uvicorn.run

        def serve(app, **kwargs):
            return original_run(app, **kwargs, fd=listener.fileno(), loop='asyncio', http='h11', ws='none')

        uvicorn.run = serve
        confine(root, [listener.fileno()])
        # The world module is first opened and compiled inside its private root.
        import types
        world = types.ModuleType('__main__')
        world.__file__ = '/server.py'
        sys.modules['__main__'] = world
        exec(compile(Path('/server.py').read_bytes(), '/server.py', 'exec'), world.__dict__)
    else:
        raise ValueError('unknown private-root operation')


if __name__ == '__main__':
    main()
