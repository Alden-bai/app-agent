"""Fail-closed resource checks and non-destructive pilot evidence archives."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from ...io import read_json, write_json


def resource_check(path, *, free_gib=35, free_inodes=3500, gpu_limit_gb=77):
    path = Path(path)
    while not path.exists():
        path = path.parent
    disk = shutil.disk_usage(path)
    inodes = os.statvfs(path).f_favail
    if disk.free < free_gib * 1024**3 or inodes < free_inodes:
        raise RuntimeError(f'pilot_storage_reserve: bytes={disk.free}, inodes={inodes}')
    used = int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.used',
        '--format=csv,noheader,nounits'], text=True).strip()) * 1024**2
    if used >= gpu_limit_gb * 10**9:
        raise RuntimeError(f'pilot_gpu_boundary: {used / 10**9:.2f} GB')
    return {'free_bytes': disk.free, 'free_inodes': inodes, 'gpu_bytes': used}


def group_health(folder, group_size):
    results = [read_json(Path(folder) / f'candidate-{i:03d}' / 'episode_result.json')
               for i in range(group_size)]
    missing = [r for r in results if r.get('reward') is None]
    markers = ('429', 'RateLimitError', 'APITimeoutError', 'APIConnectionError',
               'AuthenticationError', 'InternalServerError', 'Connection error')
    errors = []
    for i, result in enumerate(results):
        if result.get('reward') is not None:
            continue
        errors.extend([result.get('error'), result.get('errors'),
                       (result.get('trajectory_quality') or {}).get('error')])
        for path in (Path(folder) / f'candidate-{i:03d}' / 'evaluations_v2').glob('*/attempt_*/result.json'):
            errors.append(read_json(path).get('errors'))
    broken_environments = sum(r.get('execution_integrity') in {'runtime_error', 'outcome_unknown'}
        or r.get('terminal_state_status') in {'unavailable', 'untrusted'} for r in results)
    return {'null_count': len(missing), 'quarantine': broken_environments > group_size // 2,
            'broken_environment_count': broken_environments,
            'api_fault': any(any(marker in str(error) for marker in markers) for error in errors),
            'reasons': [r.get('reward_exclusion') or r.get('errors') or r.get('evaluation_status')
                        for r in missing]}


def archive_group(folder, destination):
    """Compare archived bytes to source, retaining all original evidence."""
    folder, destination = Path(folder), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.partial')
    if destination.exists() or temporary.exists():
        raise FileExistsError(destination)
    sources = {str(p.relative_to(folder)): p for p in folder.rglob('*') if p.is_file()}
    if any(p.is_symlink() for p in folder.rglob('*')):
        raise ValueError('pilot_archive_contains_symlink')
    with tarfile.open(temporary, 'w:gz', compresslevel=1) as archive:
        archive.add(folder, arcname=folder.name)
    with tarfile.open(temporary, 'r:gz') as archive:
        members = {str(Path(m.name).relative_to(folder.name)): m
                   for m in archive.getmembers() if m.isfile()}
        if members.keys() != sources.keys():
            raise ValueError('pilot_archive_file_set_mismatch')
        for name, source in sources.items():
            with source.open('rb') as handle:
                expected = hashlib.file_digest(handle, 'sha256').hexdigest()
            with archive.extractfile(members[name]) as handle:
                actual = hashlib.file_digest(handle, 'sha256').hexdigest()
            if expected != actual:
                raise ValueError('pilot_archive_hash_mismatch:' + name)
    with temporary.open('rb') as handle:
        os.fsync(handle.fileno())
    temporary.replace(destination)
    write_json(destination.with_suffix('.manifest.json'), {
        'archive': str(destination), 'files_verified': len(sources), 'bytes': destination.stat().st_size,
        'originals_retained': True})
    return str(destination)


class GroupArchiver:
    """Bounded background archives; manifests alone authorize source cleanup.

    Call wait_for_slot before collecting another group. Only immutable, finished
    groups are submitted. close joins every job and propagates archive failures.
    """

    def __init__(self, max_pending=2):
        if max_pending < 1:
            raise ValueError('max_pending must be positive')
        self.max_pending = max_pending
        self._pending = deque()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='awm-archive')

    def check(self):
        while self._pending and self._pending[0].done():
            self._pending[0].result()
            self._pending.popleft()

    def wait_for_slot(self):
        self.check()
        while len(self._pending) >= self.max_pending:
            self._pending[0].result()
            self._pending.popleft()

    def submit(self, folder, destination):
        self.wait_for_slot()
        self._pending.append(self._executor.submit(archive_group, folder, destination))
        # This is a destination, not proof of completion. Consumers must wait
        # for archive_group's verified manifest before removing originals.
        return str(destination)

    def close(self):
        self._executor.shutdown(wait=True)
        self.check()


def rollout_fraction(requested, *, total_gb, actor_gb, target_gb, overhead_gb=2.0):
    """Reserve the resident actor plus CUDA contexts outside vLLM's own budget."""
    import math
    values = (requested, total_gb, actor_gb, target_gb, overhead_gb)
    if not all(math.isfinite(x) for x in values) or not (0 < requested < 1 and total_gb > 0
            and actor_gb > 0 and overhead_gb >= 0 and actor_gb + overhead_gb < target_gb < total_gb):
        raise ValueError("invalid aggregate GPU budget")
    return min(requested, (target_gb - actor_gb - overhead_gb) / total_gb)
