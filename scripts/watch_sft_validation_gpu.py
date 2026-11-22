"""GPU scheduling monitor independent of potentially slow evidence archiving."""
import json
from pathlib import Path
import subprocess
import time

from harness_awm.io import read_json,write_json,now
from scripts.sft_eval_local_recovery import alive
from scripts.resume_sft_validation import SOURCE,CONTROL,gpu_sample

root=CONTROL/'primary'
supervisor=read_json(CONTROL/'launch.json')['primary']
workers=read_json(SOURCE/'validation/launch.json')['worker_pids']
control=root/'worker_control.json'
high=0;peak=read_json(root/'gpu_live.json').get('peak_sampled_gib',0) if (root/'gpu_live.json').exists() else 0
with (root/'gpu_live.jsonl').open('a',buffering=1) as log:
    while alive(supervisor) or any(alive(pid) for pid in workers):
        workers=read_json(SOURCE/'validation/launch.json')['worker_pids']
        sample=gpu_sample();peak=max(peak,sample['memory_gib'])
        sample['peak_sampled_gib']=peak
        log.write(json.dumps(sample)+'\n');write_json(root/'gpu_live.json',sample)
        retired=set(read_json(control)['retire_indices'])
        high=high+1 if sample['memory_gib']>86 else 0
        if high>=3 and len(retired)<4 and not any(alive(workers[i]) for i in retired):
            index=next((i for i in reversed(range(len(workers))) if i not in retired and alive(workers[i])),None)
            if index is not None:
                retired.add(index)
                write_json(control,{'retire_indices':sorted(retired),'reason':'GPU above 86 GiB; retire after current episode','at':now()})
            high=0
        time.sleep(2)
