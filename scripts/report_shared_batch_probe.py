"""Summarize bounded batch probes separately from formal evaluation scores."""
from pathlib import Path
import sys
from statistics import median
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from harness_awm.io import read_json,write_json,file_hash,now


def main():
    from dotenv import load_dotenv
    from harness_awm.tracking import SwanLabTracker
    load_dotenv(ROOT/'.env',override=False)
    sources={
        'mixed_sft':ROOT/'runs/sft_shared_batch_probe_20261109_v1/summary.json',
        'short_sft':ROOT/'runs/sft_shared_batch_short_20261109_v1/summary.json',
        'base_compatibility':ROOT/'runs/base_shared_batch_short_20261109_v1/summary.json'}
    trials={k:read_json(v) for k,v in sources.items()}
    compact={}
    for name,trial in trials.items():
        compact[name]={'path':str(sources[name]),'sha256':file_hash(sources[name]),
            'prompt_lengths':trial['prompt_lengths'],'results':trial.get('results'),
            'memory_limited_batch_sizes':trial.get('memory_limited_batch_sizes',[])}
        if trial.get('compatibility_only'):
            compact[name].update(batch_size=trial['batch_size'],
                identical_requests=trial['natural_identical_requests'],
                reference_seconds=trial['reference_actor_seconds'],natural_batch_seconds=trial['natural_batch_seconds'])
    out=ROOT/'runs/shared_batch_report_20261109';out.mkdir(exist_ok=False)
    report={'at':now(),'trials':compact,'formal_runtime_changed':False,'judge_requests':0,
        'speedup_denominator':'One shared model running the same frozen replay workload serially (batch=1). Not the 18-process production pipeline.',
        'engine':'Transformers reference kernels, explicit static batches; no vLLM installation or continuous-batching service.',
        'constraints':'5 GiB PyTorch allocator cap per probe while production continued; rejected larger batches for mixed-length input reflect this cap, not the capacity of the full 96GB GPU.',
        'sampling':'Same checkpoint, BF16, SDPA, greedy, no thinking, same EOS set. Probe output cap 96.',
        'decision':'Diagnostic only. Preserve production run; any production batching change needs an explicit unified runtime and end-to-end comparison.'}
    write_json(out/'summary.json',report)
    tracker=SwanLabTracker(out,mode='online',config={k:v for k,v in report.items() if k!='trials'})
    try:
        for step,(name,trial) in enumerate(trials.items()):
            if trial.get('compatibility_only'):
                tracker.log({'batch_compatibility':{'requests':trial['batch_size'],'identical_requests':trial['natural_identical_requests']}},step=step)
            else:
                tracker.log({name:trial['results']},step=step)
        write_json(out/'tracking.json',{'url':tracker.url})
        print('REPORT',out/'summary.json');print('SWANLAB',tracker.url)
    finally:tracker.finish()


if __name__=='__main__':main()
