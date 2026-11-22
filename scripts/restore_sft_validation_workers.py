"""Replace dead validation slots without interrupting surviving workers."""
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness_awm.io import read_json, write_json, now
from scripts.resume_sft_validation import SOURCE, CONTROL, ARCHIVE
from scripts.sft_eval_local_recovery import alive, archive_episode


def main():
    stage = SOURCE / 'validation'
    directory = CONTROL / 'replacement_workers'
    directory.mkdir(exist_ok=False)
    original = read_json(stage / 'launch.json')
    write_json(directory / 'original_launch.json', original)
    env = dict(os.environ, SWANLAB_MODE='disabled', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        PYTORCH_ALLOC_CONF='expandable_segments:True', TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    replacements = {}
    for index in (3, 13):
        launch = read_json(stage / 'launch.json')
        assert not alive(launch['worker_pids'][index]), 'slot is still active'
        with sqlite3.connect(stage / 'queue.sqlite') as db:
            assert not db.execute("SELECT 1 FROM jobs WHERE state='claimed' AND owner=?", (index,)).fetchone()
        assert index not in read_json(CONTROL / 'primary/worker_control.json')['retire_indices']
        with (directory / f'worker_{index:02d}.log').open('x') as log:
            process = subprocess.Popen([sys.executable, str(ROOT / 'scripts/resume_sft_validation.py'),
                'worker', '--root', str(SOURCE), '--split', 'validation', '--index', str(index),
                '--control', str(CONTROL / 'primary')], cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        replacements[index] = process
        launch['worker_pids'][index] = process.pid
        launch['replacements_at'] = now()
        write_json(stage / 'launch.json', launch)
        write_json(directory / 'launch.json', {'manager_pid': os.getpid(),
            'workers': {str(k): v.pid for k, v in replacements.items()}, 'at': now()})
        if index == 3:
            time.sleep(30)
            assert process.poll() is None, 'first replacement failed'
    supervisor = read_json(CONTROL / 'launch.json')['primary']
    while True:
        for process in replacements.values():
            process.poll()
        launch = read_json(stage / 'launch.json')
        live = [pid for pid in launch['worker_pids'] if alive(pid)]
        with sqlite3.connect(stage / 'queue.sqlite') as db:
            counts = dict(db.execute('SELECT state,count(*) FROM jobs GROUP BY state'))
        write_json(directory / 'status.json', {'at': now(), 'live_workers': len(live),
            'queue': counts, 'replacement_exit_codes': {str(k): p.poll() for k,p in replacements.items()}})
        if not live and not alive(supervisor):
            if any(v for k,v in counts.items() if k != 'done'):
                raise RuntimeError('unfinished queue requires inspection')
            # The original supervisor knows the dead PIDs, so it can report an
            # obsolete worker failure after the replacement jobs have completed.
            with sqlite3.connect(stage / 'queue.sqlite') as db:
                outputs = db.execute("SELECT output FROM jobs WHERE state='done'").fetchall()
            for (output,) in outputs:
                folder = Path(output) / 'episodes/t00000/a0'
                archive_episode(folder, ARCHIVE / SOURCE.name / folder.relative_to(SOURCE))
            status = read_json(SOURCE / 'status.json')
            if status.get('error') == 'validation incomplete; evidence retained, automatic selection blocked':
                write_json(directory / 'original_supervisor_exit.json', status)
                write_json(SOURCE / 'status.json', {'stage': 'awaiting_judge_recovery',
                    'validation_complete': True, 'updated_at': now(), 'worker_recovery': str(directory)})
            return
        time.sleep(10)


if __name__ == '__main__':
    main()
