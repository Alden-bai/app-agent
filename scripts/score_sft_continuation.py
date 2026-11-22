"""Authorized Judge continuation; waits for GPU evidence without interrupting workers."""
import argparse
import asyncio
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_sft_continuation as evaluation


def score(step):
    root = evaluation.CONTROL / f'checkpoint-{step}'
    evaluation.RUN = root
    with (root / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plan = evaluation.selection.checked_plan(root)
        assert evaluation.read_json(root / 'gpu_status.json')['stage'] == 'complete'
        config = evaluation.read_json(root / 'judge_config.json')['judge_api']
        authorization = evaluation.read_json(ROOT / 'runs/judge_scoring_authorization.json')
        assert authorization['authorized'] and config['base_url'] == authorization['base_url']
        tracker = evaluation.SwanLabTracker(root / 'judge_tracking', mode='offline', config={'stage':'continuation-validation-judge'})
        try:
            asyncio.run(evaluation.judge(plan, tracker))
            evaluation.persist(step, 'scored')
            assert evaluation.read_json(root / 'status.json')['stage'] == 'complete'
        finally:
            tracker.finish()


def run():
    root = evaluation.CONTROL
    with (root / 'authorized_judge.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for step in evaluation.STEPS:
                # The GPU parent must have observed its child's completion before
                # Judge changes that child's status file.
                while True:
                    state = evaluation.read_json(root / 'status.json')
                    if state['stage'] == 'needs_attention':
                        raise RuntimeError(str(state))
                    ready = (state.get('checkpoint') == 420 if step == 315 else False)
                    ready = ready or state['stage'] == 'gpu_complete_awaiting_judge'
                    evaluation.write_json(root / 'scoring_status.json', dict(stage='waiting_gpu', checkpoint=step, at=evaluation.now()))
                    if ready:
                        break
                    time.sleep(15)
                evaluation.write_json(root / 'scoring_status.json', dict(stage='scoring', checkpoint=step, at=evaluation.now()))
                subprocess.run([sys.executable, __file__, '--step', str(step)], check=True)
            reports={str(step):evaluation.read_json(root / f'checkpoint-{step}' / 'summary.json') for step in evaluation.STEPS}
            comparison=dict(at=evaluation.now(), validation=reports,
                historical_baseline=evaluation.read_json(root / 'historical_baseline.json')['summary']['lr1e-5-epoch2'])
            evaluation.write_json(root / 'comparison.json', comparison)
            evaluation.write_json(evaluation.PERSISTENT / 'comparison.json', comparison)
            evaluation.write_json(root / 'scoring_status.json', dict(stage='complete', at=evaluation.now()))
            evaluation.write_json(root / 'status.json', dict(stage='complete', at=evaluation.now()))
        except BaseException as exc:
            evaluation.write_json(root / 'scoring_status.json', dict(stage='needs_attention', error=str(exc), at=evaluation.now()))
            raise


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    parser=argparse.ArgumentParser()
    parser.add_argument('--step', type=int, choices=evaluation.STEPS)
    args=parser.parse_args()
    score(args.step) if args.step else run()
