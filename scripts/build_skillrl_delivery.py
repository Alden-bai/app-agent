"""Add the frozen skill handoff to a verified training archive, preserving its snapshot."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

from build_training_delivery import ROOT, SECRET, allowed, sha
from verify_training_delivery import verify

PREFIX = 'awm-qwen35-27b/'


def payload(archive, name, data):
    info = tarfile.TarInfo(PREFIX + name)
    info.size, info.mode, info.mtime = len(data), 0o644, 0
    archive.addfile(info, io.BytesIO(data))


def build(base, full, addon):
    for path in (full, addon):
        if path.exists() or path.with_suffix(path.suffix + '.partial').exists():
            raise FileExistsError(path)
    verify(base)
    base_hash = sha(base)
    with tarfile.open(base, 'r:gz') as archive:
        manifest = json.load(archive.extractfile(PREFIX + 'DELIVERY_MANIFEST.json'))
    names = ['scripts/a100_delivery.py', 'scripts/prepare_delivery_skills.py',
             'scripts/build_skillrl_delivery.py', 'harness_awm/training/skill_eval.py',
             'tests/test_skillrl_delivery.py', 'delivery/README_交接说明.md']
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT / 'delivery/skillrl').iterdir()) if p.is_file()]
    replacements = {}
    for name in names:
        if not allowed(name):
            raise ValueError('excluded supplement file: ' + name)
        raw = (ROOT / name).read_bytes()
        if SECRET.search(raw):
            raise ValueError('credential-like content: ' + name)
        replacements[name] = raw
    entries = {e['path']: e for e in manifest['files']}
    for name, raw in replacements.items():
        entries[name] = dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    manifest.update(files=[entries[n] for n in sorted(entries)],
        total_bytes=sum(e['bytes'] for e in entries.values()),
        skillrl_handoff=dict(base_archive_sha256=base_hash,
            scope='Frozen inference skills and paired evaluation; no training changes or weights',
            bank_sha256=sha(ROOT / 'delivery/skillrl/bank.json'),
            replaced_or_added_files=sorted(replacements), raw_skill_source_archives_included=False,
            gpu_evaluation='not run; pending on target hardware'))
    manifest_raw = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
    full.parent.mkdir(parents=True, exist_ok=True)
    partial = full.with_suffix(full.suffix + '.partial')
    with tarfile.open(partial, 'w:gz', compresslevel=6) as dest, tarfile.open(base, 'r|gz') as source:
        for member in source:
            name = member.name[len(PREFIX):]
            if name == 'DELIVERY_MANIFEST.json' or name in replacements:
                continue
            with source.extractfile(member) as stream:
                dest.addfile(member, stream)
        for name, raw in sorted(replacements.items()):
            payload(dest, name, raw)
        payload(dest, 'DELIVERY_MANIFEST.json', manifest_raw)
    verify(partial)
    partial.rename(full)
    addon.parent.mkdir(parents=True, exist_ok=True)
    partial = addon.with_suffix(addon.suffix + '.partial')
    with tarfile.open(partial, 'w:gz', compresslevel=6) as dest:
        for name, raw in sorted(replacements.items()):
            payload(dest, name, raw)
        payload(dest, 'DELIVERY_MANIFEST.json', manifest_raw)
    partial.rename(addon)
    full.with_suffix(full.suffix + '.manifest.json').write_bytes(manifest_raw)
    for path in (full, addon):
        path.with_suffix(path.suffix + '.sha256').write_text(sha(path) + '  ' + path.name + '\n')
    print(json.dumps(dict(full=str(full), addon=str(addon), full_bytes=full.stat().st_size,
                         addon_bytes=addon.stat().st_size, files=len(entries), base_sha256=base_hash), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=ROOT / 'dist/awm-qwen35-27b-a100x8-delivery.tar.gz')
    p.add_argument('--full', type=Path, default=ROOT / 'dist/awm-qwen35-27b-a100x8-with-skillrl.tar.gz')
    p.add_argument('--addon', type=Path, default=ROOT / 'dist/awm-skillrl-addon.tar.gz')
    args = p.parse_args()
    build(args.base, args.full, args.addon)
