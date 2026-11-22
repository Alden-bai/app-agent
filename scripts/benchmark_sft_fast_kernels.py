"""Bounded, no-Judge paired benchmark; never changes running evaluation workers."""
import argparse
import asyncio
import gc
import inspect
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/sft_kernel_deps'))
os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
from harness_awm.io import read_json, write_json, file_hash, now


def main():
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    from harness_awm.training.local import load_checkpoint
    from harness_awm.training.actor import TrainingActorBackend
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--candidate', default='lr1e-5-epoch2')
    parser.add_argument('--kernel-mode', choices=('all','fla-only','conv-only'), default='all')
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--repetitions', type=int, default=2)
    a = parser.parse_args()
    a.output.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    free, total = torch.cuda.mem_get_info()
    deadline = time.monotonic() + 120
    while free < 8 * 1024**3:
        write_json(a.output / 'status.json', {'stage': 'waiting_for_gpu_headroom', 'free_GiB': free / 1024**3, 'at': now()})
        if time.monotonic() > deadline:
            raise RuntimeError('Need at least 8 GiB free for bounded benchmark')
        time.sleep(5)
        free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(5 * 1024**3 / total)
    names = ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule',
             'causal_conv1d_fn', 'causal_conv1d_update')
    fast = {name: getattr(q, name) for name in names}
    slow = {name: inspect.unwrap(func) for name, func in fast.items()}
    if a.kernel_mode == 'fla-only':
        for name in names[2:]: fast[name] = slow[name]
    elif a.kernel_mode == 'conv-only':
        for name in names[:2]: fast[name] = slow[name]
    bindings = {}
    for name, func in fast.items():
        cells = inspect.getclosurevars(func).nonlocals
        impl = cells.get('implementation')
        bindings[name] = {'fast': cells.get('is_new_implementation', False),
                          'module': getattr(impl, '__module__', None)}
    expected = names[:2] if a.kernel_mode == 'fla-only' else names[2:] if a.kernel_mode == 'conv-only' else names
    if not all(bindings[n]['fast'] for n in expected):
        raise RuntimeError('requested fast kernel is not actually bound')
    candidate = a.candidate
    identity = read_json(a.source / 'validation/candidates.json')[candidate]
    write_json(a.output / 'status.json', {'stage': 'loading', 'at': now(), 'bindings': bindings})
    model, tokenizer = load_checkpoint(identity['path'], device='cuda', dtype='bfloat16')
    model.requires_grad_(False).eval()
    inputs = []
    with sqlite3.connect(a.source / 'validation/queue.sqlite') as db:
        outputs = [Path(x[0]) for x in db.execute("select output from jobs where state='done' and candidate=? order by id", (candidate,))]
    # Freeze three observed contexts covering short and longer multi-turn requests.
    for output in outputs:
        turns = sorted((output / 'episodes/t00000/a0/model_turns').glob('*/input.json'))
        for f in turns[::max(1,len(turns)//3)]:
            messages = read_json(f)['messages']
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            if hasattr(ids, 'keys'):
                ids = ids['input_ids']
            if len(ids) <= 5000 and not any(abs(len(ids)-x['tokens']) < 500 for x in inputs):
                inputs.append({'messages':messages,'tokens':len(ids),'source':str(f), 'sha256':file_hash(f)})
            if len(inputs) == 3: break
        if len(inputs) == 3: break
    if a.inputs:
        inputs = read_json(a.inputs)
    if len(inputs) < 2:
        raise RuntimeError('not enough representative bounded inputs')
    write_json(a.output / 'inputs.json', inputs)
    records = []
    def mode(which):
        for name, func in (fast if which=='fast' else slow).items():setattr(q, name, func)
    async def sample(which, item, repetition, warmup=False):
        mode(which)
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        actor = TrainingActorBackend(model, tokenizer, policy_version=identity['sha256'],
            max_output_tokens=96, context_limit=32768, greedy=True, record_policy=True)
        folder = a.output / ('warmup' if warmup else 'samples') / f'{repetition}_{which}_{item["tokens"]}'
        torch.cuda.synchronize(); started=time.perf_counter()
        response = await actor.generate(item['messages'], folder, 1)
        torch.cuda.synchronize(); seconds=time.perf_counter()-started
        row={'mode':which,'prompt_tokens':item['tokens'],'source':item['source'],'repetition':repetition,
            'seconds':seconds,'completion_ids':actor.records[-1]['completion_ids'],
            'completion_tokens':response['usage']['completion_tokens'],'content':response['content'],
            'finish_reason':response['finish_reason'], 'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,
            'warmup':warmup}
        records.append(row)
        write_json(a.output / 'measurements.json', records)
        print(json.dumps({k:v for k,v in row.items() if k not in ('content','completion_ids','source')}),flush=True)
    async def run():
        # Warm both paths for every context shape before timed alternating pairs.
        for item in inputs:
            for which in ('fast','slow'):await sample(which,item,-1,True)
        for repetition in range(a.repetitions):
            for item in inputs:
                for which in (('slow','fast') if repetition==0 else ('fast','slow')):
                    await sample(which,item,repetition)
    write_json(a.output / 'status.json', {'stage':'benchmarking','at':now(),'bindings':bindings})
    asyncio.run(run())
    pairs=[]
    for repetition in range(a.repetitions):
        for item in inputs:
            pair={r['mode']:r for r in records if not r['warmup'] and r['repetition']==repetition and r['source']==item['source']}
            s,f=pair['slow'],pair['fast']
            pairs.append({'source':item['source'],'prompt_tokens':item['tokens'],'repetition':repetition,
                'identical_completion_ids':s['completion_ids']==f['completion_ids'],
                'slow_seconds':s['seconds'],'fast_seconds':f['seconds'],
                'slow_completion_tokens':s['completion_tokens'],'fast_completion_tokens':f['completion_tokens'],
                'wall_speedup':s['seconds']/f['seconds']})
    result={'at':now(),'checkpoint':identity,'bindings':bindings,'pairs':pairs,
        'all_outputs_identical':all(p['identical_completion_ids'] for p in pairs),
        'median_paired_wall_speedup':median(p['wall_speedup'] for p in pairs),
        'caveat':'Shared GPU with 18 active evaluation workers; 96-token cap; warmup excluded; not an end-to-end evaluation speed claim',
        'harness_modified':False,'judge_requests':0}
    write_json(a.output / 'summary.json',result)
    write_json(a.output / 'status.json',{'stage':'finished','at':now()})
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
