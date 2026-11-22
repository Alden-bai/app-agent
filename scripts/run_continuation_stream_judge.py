"""Stream both validation candidates through one shared 2M TPM / 5000 RPM gate."""
import fcntl
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import evaluate_sft_continuation as evaluation
root=evaluation.CONTROL
with (root/'authorized_judge.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        for step in evaluation.STEPS:
            evaluation.write_json(root/'scoring_status.json',dict(stage='streaming',checkpoint=step,tpm=2000000,rpm=5000,workers=32,at=evaluation.now()))
            subprocess.run([sys.executable,str(evaluation.ROOT/'scripts/sft_continuation_stream_judge.py'),'--step',str(step)],check=True)
        with (root/'supervisor.lock').open('a') as gpu_lock:
            fcntl.flock(gpu_lock,fcntl.LOCK_EX)
            reports={str(step):evaluation.read_json(root/f'checkpoint-{step}'/'summary.json') for step in evaluation.STEPS}
            comparison=dict(at=evaluation.now(),validation=reports,historical_baseline=evaluation.read_json(root/'historical_baseline.json')['summary']['lr1e-5-epoch2'])
            evaluation.write_json(root/'comparison.json',comparison)
            evaluation.write_json(evaluation.PERSISTENT/'comparison.json',comparison)
            evaluation.write_json(root/'scoring_status.json',dict(stage='complete',at=evaluation.now()))
            evaluation.write_json(root/'status.json',dict(stage='complete',at=evaluation.now()))
    except BaseException as exc:
        evaluation.write_json(root/'scoring_status.json',dict(stage='needs_attention',error=str(exc),at=evaluation.now()))
        raise
