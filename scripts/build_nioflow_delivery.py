"""Package this checkout, the frozen training source closure and NIOFlow entry point."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import zipfile

from build_training_delivery import ROOT, SECRET, allowed, sha
from verify_training_delivery import verify

PREFIX = 'awm-nioflow-cu128-grpo2000'
SPLIT = ROOT / 'data/training_split_20261106_v1'
RUNTIME = ROOT / 'data/training_runtime_20261106_v1'


def inventory():
    files = {}
    def add(path, name=None):
        name = name or str(path.relative_to(ROOT))
        if (not allowed(name) or not path.is_file() or path.is_symlink()
                or not path.resolve().is_relative_to(ROOT)):
            raise ValueError('unexpected source: ' + name)
        files[name] = path

    for folder in ('harness_awm', 'scripts', 'configs', 'tests', 'requirements', 'assets/qwen35-27b',
                   'docs', 'skills', 'evidence/legacy',
                   'agent-world-model/awm', 'vendor/verl', 'delivery/nioflow', 'delivery/skillrl',
                   'data/awm-dde80a0-repaired-v1', 'data/training_qwen35_27b_v1', SPLIT, RUNTIME):
        for base, dirs, names in os.walk(ROOT / folder):
            dirs[:] = sorted(n for n in dirs if allowed(n) and n not in {'.github', '.codex', '.claude', '.agent',
                              '.gemini', '.vscode', 'mcp_smoke'} and not (Path(base) / n).is_symlink())
            for name in sorted(names):
                path = Path(base) / name
                if allowed(str(path.relative_to(ROOT))) and not path.is_symlink():
                    add(path)
    for name in ('pyproject.toml', 'agent-world-model/README.md', 'agent-world-model/pyproject.toml',
                 'delivery/run-config.example.json', 'delivery/env.example', 'delivery/nioflow-worlds.zip'):
        add(ROOT / name)
    for path in (ROOT / 'agent-world-model').glob('LICENSE*'):
        add(path)
    for source_name, packaged_name in [('README.md', 'README.md'), ('CODEX.md', 'AGENTS.md')]:
        source = ROOT / 'delivery/nioflow' / source_name
        add(source if source.is_file() else ROOT / packaged_name, packaged_name)
    task_book = '任务_全量GRPO.md'
    source = ROOT / 'dist' / task_book
    add(source if source.is_file() else ROOT / task_book, task_book)
    files.pop('delivery/nioflow/README.md', None)
    files.pop('delivery/nioflow/CODEX.md', None)
    files.pop('delivery/skillrl/README_技能库交接.md', None)
    source_root = Path(json.loads((SPLIT / 'manifest.json').read_text())['source_root'])
    pins = {str(Path(ref).relative_to(source_root)): expected for ref, expected in
            json.loads((SPLIT / 'source_hashes.json').read_text()).items()}
    for rel in pins:
        add(ROOT / rel)
    return files, pins


def build(output):
    if output.exists():
        raise FileExistsError(output)
    files, pins = inventory()
    worlds = set()
    for split in ('grpo_train', 'validation', 'test'):
        with (RUNTIME / (split + '.jsonl')).open() as rows:
            worlds.update(json.loads(row)['environment_code_hash'] + '.py' for row in rows if row.strip())
    with zipfile.ZipFile(ROOT / 'delivery/nioflow-worlds.zip') as bundle:
        if worlds != set(bundle.namelist()) or len(bundle.namelist()) != len(worlds) or bundle.testzip():
            raise ValueError('frozen world/script coverage mismatch')
    secrets = []
    if (ROOT / '.env').is_file():
        for line in (ROOT / '.env').read_text().splitlines():
            key, _, value = line.partition('=')
            value = value.strip().strip('\"\'')
            if any(s in key.upper() for s in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD')) and len(value) >= 16:
                secrets.append(value.encode())
    entries = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.partial') as temp:
        with gzip.GzipFile(fileobj=temp, filename='', mode='wb', mtime=0, compresslevel=4) as gz, tarfile.open(fileobj=gz, mode='w|') as tar:
            def write(name, raw):
                item = tarfile.TarInfo(PREFIX + '/' + name)
                item.size, item.mode = len(raw), 0o755 if name.endswith('.sh') else 0o644
                tar.addfile(item, io.BytesIO(raw))

            for name, path in sorted(files.items()):
                raw = path.read_bytes()
                value = hashlib.sha256(raw).hexdigest()
                if name in pins and pins[name] != value:
                    raise ValueError('changed frozen input: ' + name)
                if SECRET.search(raw) or any(secret in raw for secret in secrets):
                    raise ValueError('credential-like content: ' + name)
                write(name, raw)
                entries.append(dict(path=name, bytes=len(raw), sha256=value))
            launcher = b'#!/usr/bin/env bash\nset -eu\nexec bash "$(dirname -- "$0")/delivery/nioflow/run.sh" "$@"\n'
            write('run.sh', launcher)
            entries.append(dict(path='run.sh', bytes=len(launcher), sha256=hashlib.sha256(launcher).hexdigest()))
            manifest = dict(schema_version='awm-nioflow-cu128-grpo2000-v2', model='Qwen/Qwen3.5-27B',
                            target='Linux x86_64, Python 3.12, CUDA 12.8, one node with 8 A100 80GB',
                            grpo_task_selection='delivery/nioflow/grpo-2000.json',
                            weights_included=False, gpu_validation='pending on target allocation',
                            source_evidence_files=len(pins), frozen_worlds=len(worlds), files=entries,
                            total_bytes=sum(e['bytes'] for e in entries))
            write('DELIVERY_MANIFEST.json', json.dumps(manifest, ensure_ascii=False, separators=(',', ':')).encode())
        temp.flush()
        verify(temp.name)
        os.link(temp.name, output)
    output.chmod(0o644)
    value = sha(output)
    Path(str(output) + '.sha256').write_text(f'{value}  {output.name}\n')
    print(json.dumps(dict(archive=str(output), bytes=output.stat().st_size, sha256=value,
                          files=len(entries), frozen_worlds=len(worlds), source_files=len(pins))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'dist' / (PREFIX + '.tar.gz'))
    build(p.parse_args().output)
