"""Render the frozen worlds once with AWM's official generator for private-root workers."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'agent-world-model')]
from awm.core.server import Config, _generate_server_code
from harness_awm.io import digest


def build(output):
    worlds = ROOT / 'data/training_runtime_20261106_v1/worlds'
    seen = set()
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(tmp / 'worlds.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(worlds.glob('*/environment.json')):
                env = json.loads(path.read_text())
                key = digest(env)
                if key in seen:
                    continue
                seen.add(key)
                cfg = Config(scenario=env['scenario'], envs_load_path=str(path), db_path='/working.db',
                             port=8001, temp_server_path=str(tmp / 'server.py'))
                # tools_jsonl_load expects one JSON object per line.
                (tmp / 'envs.jsonl').write_text(json.dumps(env) + '\n')
                cfg.envs_load_path = str(tmp / 'envs.jsonl')
                _generate_server_code(cfg, '/working.db')
                script = (tmp / 'server.py').read_bytes()
                compile(script, 'server.py', 'exec')
                archive.writestr(key + '.py', script)
        if not seen:
            raise ValueError('missing frozen worlds')
        (tmp / 'worlds.zip').replace(output)
    print(f'{len(seen)} distinct world scripts in {output}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'delivery/nioflow-worlds.zip')
    build(p.parse_args().output)
