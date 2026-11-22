"""Keep a veRL SFT run and GPU telemetry alive after the launching shell exits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from harness_awm.io import write_json

p = argparse.ArgumentParser()
p.add_argument('--run', type=Path, required=True)
a = p.parse_args()
run = a.run.resolve()
with (run / 'monitor_started.json').open('x') as f:
    json.dump({'pid':os.getpid(), 'started_at':time.time()}, f)
command = [sys.executable, '-m', 'harness_awm.training.verl.sft', 'train', '--run', str(run)]
with (run / 'train.log').open('a') as log, (run / 'gpu.jsonl').open('a', buffering=1) as telemetry:
    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    maximum = 0
    while child.poll() is None:
        try:
            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'], text=True, timeout=10).strip().split(',')
            memory, utilization = int(raw[0]), int(raw[1])
            maximum = max(maximum, memory)
            sample = {'time':time.time(), 'memory_gib':memory/1024, 'utilization':utilization}
            telemetry.write(json.dumps(sample)+'\n')
            write_json(run / 'status.json', {'stage':'training', 'monitor_pid':os.getpid(),
                'train_pid':child.pid, 'gpu':sample, 'peak_sampled_gib':maximum/1024})
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            log.write(f'GPU monitoring warning: {error}\n'); log.flush()
        time.sleep(2)
    write_json(run / 'status.json', {'stage':'complete' if child.returncode==0 else 'failed',
        'exit_code':child.returncode, 'finished_at':time.time(), 'peak_sampled_gib':maximum/1024})
sys.exit(child.returncode)
