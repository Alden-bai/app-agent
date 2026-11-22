"""Requeue only an idle test scorer behind a validation scorer's lock handoff."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/root/awm-agent-train')
RUN = ROOT / 'runs/sft_server_recovery_20261110_v1'
TEST = ROOT / 'runs/base_test_20261110_v1'


def read(path):
    return json.loads(path.read_text())


def write(value):
    path = RUN / 'test_judge_handoff.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(at=time.time(), **value)))
    tmp.replace(path)


def main():
    status = read(TEST / 'judge_status.json')
    pid = status['pid']
    process = Path('/proc') / str(pid)
    args = [x.decode() for x in (process / 'cmdline').read_bytes().split(b'\0') if x]
    assert args[1:] == [str(ROOT / 'scripts/base_test_20261110.py'), 'run'], args
    assert status['stage'] == 'waiting_for_validation_judge'
    assert read(TEST / 'gpu_status.json')['stage'] == 'complete'
    assert (process / 'wchan').read_text().strip() == 'locks_lock_inode_wait'
    assert read(RUN / 'optimized_launch.json')['stage'] == 'waiting_for_previous_judge'
    # Capture the existing environment in memory only; never write credentials.
    env = dict(x.decode().split('=', 1) for x in (process / 'environ').read_bytes().split(b'\0') if x)
    write({'stage': 'requeue_idle_test_waiter', 'previous_pid': pid, 'args': args})
    os.kill(pid, signal.SIGTERM)
    # The old process is waiting for flock: no API or GPU work is interrupted.
    for _ in range(100):
        if not process.exists() or (process / 'stat').read_text().split()[2] == 'Z':
            break
        time.sleep(.1)
    else:
        raise RuntimeError('idle test waiter did not exit')
    while True:
        launch = read(RUN / 'optimized_launch.json')
        successor = Path('/proc') / str(launch['pid'])
        if launch['stage'] == 'running' or not successor.exists():
            break
        time.sleep(2)
    with (TEST / 'optimization_handoff_resume.log').open('ab', buffering=0) as log:
        restarted = subprocess.Popen(args, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write({'stage': 'test_waiter_restarted', 'previous_pid': pid, 'pid': restarted.pid,
           'validation_launch': launch, 'args': args})


if __name__ == '__main__':
    main()
