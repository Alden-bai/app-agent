"""Verify four independent real MCP sessions and committed SQLite changes on CPU."""
import argparse
import asyncio
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness_awm.contracts import RunConfig
from harness_awm.environment import AWMEnvAdapter
from harness_awm.io import file_hash
from harness_awm.training.baseline import isolation_probe
from harness_awm.training.tasks import load_tasks


async def smoke(config, folder):
    probe = isolation_probe(config, folder / 'probe')
    if not probe['passed']:
        raise RuntimeError(f'worker isolation failed: {probe}')
    tasks, _, _ = load_tasks(ROOT / 'data/training_runtime_20261106_v1', 'grpo_train')
    task = next(task for task in tasks if task.scenario == 'accounting_2')
    names = [f'awm_smoke_role_{i}' for i in range(4)]

    async def one(index):
        env = AWMEnvAdapter(task, config, folder / f'candidate-{index}')
        try:
            await env.prepare()
            result = await env.call('create_role', {'name': names[index]})
            if result.get('isError'):
                raise RuntimeError(f'create_role failed: {result}')
        finally:
            if not await env.close():
                raise RuntimeError('task server failed to stop')
        env.freeze()
        with sqlite3.connect(env.folder / 'final.db') as db:
            observed = [r[0] for r in db.execute('SELECT name FROM roles WHERE name IN (?,?,?,?)', names)]
        if observed != [names[index]]:
            raise ValueError(f'wrong database state in candidate {index}: {observed}')
        return len(env.tools)

    counts = await asyncio.gather(*(one(i) for i in range(4)))
    if file_hash(task.initial_state_ref) != task.initial_state_hash:
        raise ValueError('frozen initial database changed')
    return dict(passed=True, backend=config.isolation, task=task.task_instance_id,
                catalog_sizes=counts, independent_committed_databases=4, model_calls=0, judge_requests=0)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    config = RunConfig.model_validate_json(a.config.read_text())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=a.output.parent, prefix='worker-smoke-') as temporary:
        folder = Path(temporary)
        try:
            result = asyncio.run(smoke(config, folder))
        except BaseException:
            for log in sorted(folder.rglob('*.log')):
                print(log.relative_to(folder), log.read_text(errors='replace')[-4500:], file=sys.stderr)
            raise
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
