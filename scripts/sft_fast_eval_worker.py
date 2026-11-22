"""Audited fast-kernel worker; retains frozen model, prompts, sampling and Judge."""
import argparse
import inspect
from importlib.metadata import version
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness_awm.io import read_json, write_json, file_hash, now


def bindings():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    result = {}
    for name in ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule',
                 'causal_conv1d_fn', 'causal_conv1d_update'):
        cells = inspect.getclosurevars(getattr(q, name)).nonlocals
        result[name] = {'fast': cells.get('is_new_implementation', False),
                        'module': getattr(cells.get('implementation'), '__module__', None)}
    if not all(v['fast'] for v in result.values()):
        raise RuntimeError('required fast kernel fell back to reference')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=('validation', 'test'), required=True)
    p.add_argument('--index', type=int, required=True)
    a = p.parse_args()
    root = a.output.resolve()
    runtime = {'schema_version': 'awm-inference-kernel-runtime-v1', 'mode': 'fla-and-causal-conv1d',
        'bindings': bindings(), 'versions': {k: version(k) for k in ('torch','transformers','flash-linear-attention','fla-core','causal-conv1d')},
        'switch_manifest_sha256': file_hash(root / 'kernel_switch.json'),
        'worker_script_sha256': file_hash(__file__), 'at': now()}
    if file_hash(root / 'kernel_switch.json') != read_json(root / 'kernel_switch.sha256.json')['sha256']:
        raise ValueError('kernel switch manifest changed')
    print(runtime, flush=True)
    from harness_awm.training import evaluate as evaluation_module
    from harness_awm.training.selection import worker
    original = evaluation_module.evaluate
    async def audited(*args, **kwargs):
        output = Path(args[4] if len(args)>4 else kwargs['output'])
        info = {**runtime, 'task_started_at': now(), 'output': str(output)}
        write_json(output.with_name(output.name + '.kernel-runtime.json'), info)
        try:
            return await original(*args, **kwargs)
        finally:
            if output.exists():
                write_json(output / 'kernel_runtime.json', info)
    evaluation_module.evaluate = audited
    worker(root, a.split, a.index)


if __name__ == '__main__':
    main()
