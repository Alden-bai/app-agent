"""Repair the separate CUDA 13 GRPO environment without changing SFT installs."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import venv
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--venv', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    fresh = not (args.venv / 'bin/python').exists()
    for target, minimum_gib in ((args.venv, 30 if fresh else 2), (args.work, 1)):
        parent = target.absolute()
        while not parent.exists():
            parent = parent.parent
        if shutil.disk_usage(parent).free < minimum_gib * 1024**3 or os.statvfs(parent).f_favail < 2000:
            raise ValueError(f'insufficient space/inodes for {target}: need {minimum_gib} GiB and 2000 inodes')
    args.work.mkdir(parents=True, exist_ok=True)
    if fresh:
        venv.EnvBuilder(with_pip=True).create(args.venv)
    python = str(args.venv / 'bin/python')
    env = dict(os.environ, PIP_NO_CACHE_DIR='1', TMPDIR=str(args.work / 'tmp'), MAX_JOBS='2')
    Path(env['TMPDIR']).mkdir(exist_ok=True)
    def run(*argv):
        subprocess.run([python, '-m', 'pip', *argv], env=env, check=True)
    lock = json.loads((ROOT / 'requirements/verl-runtime.json').read_text())
    for name, expected in lock['source_sha256'].items():
        assert hashlib.sha256((args.source / 'verl' / name).read_bytes()).hexdigest() == expected, name
    source = args.work / 'verl-source'
    if not source.exists():
        shutil.copytree(args.source, source, ignore=shutil.ignore_patterns('.git', '__pycache__', 'build', '*.egg-info'))
    setup = source / 'setup.py'
    setup.write_text(setup.read_text().replace('"numpy<2.0.0"', '"numpy==2.3.5"'))
    (source / 'verl/version/version').write_text('0.8.0.dev0+awm.numpy2\n')
    patch = ''
    for name in ('setup.py', 'verl/version/version'):
        patch += ''.join(difflib.unified_diff((args.source / name).read_text().splitlines(True),
            (source / name).read_text().splitlines(True), fromfile='upstream/' + name, tofile='local/' + name))
    (args.work / 'metadata.patch').write_text(patch)
    constraints = args.work / 'constraints.txt'
    pins = {k: v for k, v in lock['packages'].items() if k not in {'verl', 'flash-attn'}}
    pins.update({'numpy': '2.3.5', 'liger-kernel': '0.8.5', 'flash-linear-attention': '0.6.0',
                 'swanlab': '0.10.4', 'setuptools': '80.10.2'})
    constraints.write_text(''.join(f'{k}=={v}\n' for k, v in pins.items()))
    run('install', '-c', str(constraints), 'hatchling>=1.25', 'wheel', 'setuptools==80.10.2')
    wheels = args.work / 'wheels'
    wheels.mkdir(exist_ok=True)
    run('wheel', '--no-deps', '--no-build-isolation', '-w', str(wheels), str(source))
    wheel = next(wheels.glob('verl-0.8.0.dev0+awm.numpy2-*.whl'))
    source_hashes = {}
    with zipfile.ZipFile(wheel) as archive:
        for path in sorted((args.source / 'verl').rglob('*.py')):
            name = str(path.relative_to(args.source))
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError('wheel modified upstream Python source: ' + name)
            source_hashes[name] = expected
    (args.work / 'all-source-sha256.json').write_text(json.dumps(source_hashes, indent=2) + '\n')
    run('install', '-c', str(constraints), *[f'{k}=={v}' for k, v in pins.items()], str(wheel),
        'cuda-toolkit[nvcc,crt,nvvm]==13.0.2',
        'swanlab==0.10.4', 'pytest>=8,<10', '-e', str(ROOT))
    run('check')
    freeze = subprocess.check_output([python, '-m', 'pip', 'freeze'], text=True)
    (args.work / 'installed.txt').write_text(freeze)
    report = {'upstream_revision': lock['revision'], 'wheel': str(wheel),
              'wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(),
              'metadata_patch': {'numpy<2.0.0': 'numpy==2.3.5', 'version': '0.8.0.dev0+awm.numpy2'},
              'source_sha256': lock['source_sha256']}
    (args.work / 'install-report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
