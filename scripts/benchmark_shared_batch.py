"""One shared model, batched greedy decoding; diagnostics only, no Judge calls."""
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
from harness_awm.io import read_json, write_json, file_hash, digest, now
from harness_awm.training.actor import TrainingActorBackend, stop_token_ids


def batch_decode(model, tokenizer, prompts, max_tokens=96, replay=None,
                 on_complete=None, cancelled=None):
    """Left-pad independent contexts. A finished row never appends further output.

    replay teacher-forces frozen reference completions for equal-work timing;
    it is deliberately separate from the natural greedy compatibility check.
    """
    import torch
    if not prompts or any(not x for x in prompts):
        raise ValueError('nonempty prompts required')
    if max_tokens < 1:
        raise ValueError('positive token limit required')
    if replay is not None and (len(replay)!=len(prompts) or any(not x or len(x)>max_tokens for x in replay)):
        raise ValueError('invalid replay lengths')
    device=next(model.parameters()).device
    eos=set(stop_token_ids(model,tokenizer));pad=tokenizer.pad_token_id
    if pad is None:pad=min(eos)
    width=max(map(len,prompts));batch=len(prompts)
    ids=torch.tensor([[pad]*(width-len(x))+list(x) for x in prompts],device=device)
    mask=torch.tensor([[0]*(width-len(x))+[1]*len(x) for x in prompts],device=device)
    outputs=[[] for _ in prompts]; predictions=[[] for _ in prompts]
    active=[True]*batch;cache=None
    logits_options={'logits_to_keep':1} if 'logits_to_keep' in inspect.signature(model.forward).parameters else {}
    model.eval()
    with torch.inference_mode():
        for step in range(max_tokens):
            if cancelled is not None:
                for i in range(batch):
                    if active[i] and cancelled(i): active[i] = False
                if not any(active): break
            positions=mask.long().cumsum(-1)-1
            positions.masked_fill_(mask==0,0)
            inputs=model.prepare_inputs_for_generation(ids,attention_mask=mask,position_ids=positions,
                past_key_values=cache,next_sequence_length=1 if cache is not None else None,
                is_first_iteration=cache is None,use_cache=True)
            response=model(**inputs,**logits_options)
            logprobs=response.logits[:,-1].float().log_softmax(-1)
            if not torch.isfinite(logprobs).all():raise ValueError('nonfinite_sampling_logprobs')
            tokens=logprobs.argmax(-1)
            # The production actor also extracts selected logprobs, even for scoring-only runs.
            selected=logprobs.gather(1,tokens[:,None]).squeeze(-1).tolist()
            token_list=tokens.tolist()
            cache=response.past_key_values
            feed=[];extension=[]
            for i, token in enumerate(token_list):
                if not active[i]:feed.append(pad);extension.append(0);continue
                predictions[i].append(token)
                chosen=replay[i][step] if replay is not None else token
                outputs[i].append(chosen)
                done=(step+1==len(replay[i])) if replay is not None else chosen in eos
                active[i]=not done
                feed.append(chosen);extension.append(1)
                if on_complete is not None and (done or step + 1 == max_tokens):
                    on_complete(i, list(outputs[i]))
            if not any(active):break
            ids=torch.cat([ids,torch.tensor(feed,device=device)[:,None]],dim=-1)
            mask=torch.cat([mask,torch.tensor(extension,device=device)[:,None]],dim=-1)
    return {'completion_ids':outputs,'predicted_ids':predictions}


def collect_inputs(source,tokenizer,count,max_prompt_tokens=5000):
    inputs=read_json(ROOT/'runs/sft_fast_kernel_benchmark_20261109_v2/inputs.json')
    inputs=[x for x in inputs if x['tokens'] <= max_prompt_tokens]
    seen={digest(x['messages']) for x in inputs}
    with sqlite3.connect(source/'validation/queue.sqlite') as db:
        outputs=[Path(r[0]) for r in db.execute("SELECT output FROM jobs WHERE state='done' AND candidate='lr1e-5-epoch2' ORDER BY id")]
    for output in outputs:
        for path in sorted((output/'episodes/t00000/a0/model_turns').glob('*/input.json'))[:1]:
            messages=read_json(path)['messages'];key=digest(messages)
            if key in seen:continue
            ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
            if hasattr(ids,'keys'):ids=ids['input_ids']
            if 1 <= len(ids) <= max_prompt_tokens:
                inputs.append({'messages':messages,'tokens':len(ids),'source':str(path),'sha256':file_hash(path)})
                seen.add(key)
        if len(inputs)>=count:break
    if len(inputs)<count:raise RuntimeError('not enough distinct real prompts')
    # Freeze prompt ordering; group adjacent lengths to reduce padding waste.
    return sorted(inputs[:count],key=lambda x:x['tokens'])


def main():
    import torch
    from harness_awm.training.local import load_checkpoint
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--candidate',default='lr1e-5-epoch2');p.add_argument('--count',type=int,default=8)
    p.add_argument('--sizes',type=int,nargs='+',default=[1,2,4,8]);p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--max-prompt-tokens',type=int,default=5000)
    p.add_argument('--compatibility-only',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    for name in ('torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule','causal_conv1d_fn','causal_conv1d_update'):
        if inspect.getclosurevars(getattr(q,name)).nonlocals.get('is_new_implementation'):
            raise RuntimeError('benchmark must use the production reference kernels')
    deadline=time.monotonic()+120
    free,total=torch.cuda.mem_get_info()
    while free < 8*1024**3:
        write_json(a.output/'status.json',{'stage':'waiting_for_headroom','at':now(),'free_GiB':free/1024**3})
        if time.monotonic()>deadline:raise RuntimeError('not enough GPU headroom')
        time.sleep(5);free,total=torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(5*1024**3/total)
    identity=read_json(a.source/'validation/candidates.json')[a.candidate]
    write_json(a.output/'status.json',{'stage':'loading','at':now()})
    model,tokenizer=load_checkpoint(identity['path'],device='cuda',dtype='bfloat16')
    model.requires_grad_(False).eval()
    items=collect_inputs(a.source,tokenizer,a.count,a.max_prompt_tokens)
    prompts=[]
    for item in items:
        ids=tokenizer.apply_chat_template(item['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False)
        if hasattr(ids,'keys'):ids=ids['input_ids']
        prompts.append(ids)
    write_json(a.output/'inputs.json',items)
    rows=[];refs=[]
    async def actor_ref(index,warmup=False):
        actor=TrainingActorBackend(model,tokenizer,policy_version=identity['sha256'],max_output_tokens=96,
            context_limit=32768,greedy=True,record_policy=True)
        torch.cuda.synchronize();start=time.perf_counter()
        await actor.generate(items[index]['messages'],a.output/('warmup' if warmup else 'reference')/str(index),1)
        torch.cuda.synchronize()
        return {'seconds':time.perf_counter()-start,'tokens':actor.records[-1]['completion_ids']}
    asyncio.run(actor_ref(0,True))
    write_json(a.output/'status.json',{'stage':'serial_reference','at':now()})
    for i in range(len(items)):
        value=asyncio.run(actor_ref(i));refs.append(value)
        write_json(a.output/'reference.json',refs)
    ref_tokens=[x['tokens'] for x in refs]
    # Verify the custom decoder reproduces the production actor in batch=1.
    one=batch_decode(model,tokenizer,[prompts[0]])
    if one['completion_ids'][0]!=ref_tokens[0]:raise RuntimeError('batch=1 differs from production actor')
    if a.compatibility_only:
        size=max(a.sizes)
        torch.cuda.synchronize();start=time.perf_counter()
        natural=batch_decode(model,tokenizer,prompts)
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        report={'at':now(),'checkpoint':identity,'prompt_lengths':list(map(len,prompts)),
            'batch_size':len(prompts),'reference_actor_seconds':sum(x['seconds'] for x in refs),
            'natural_batch_seconds':seconds,'natural_identical_requests':sum(x==y for x,y in zip(natural['completion_ids'],ref_tokens)),
            'reference_ids':ref_tokens,'batched_ids':natural['completion_ids'],
            'formal_runtime_changed':False,'judge_requests':0,'compatibility_only':True,
            'limitations':['Only output compatibility; no warmed equal-work speedup claim','Shared GPU; 96-token cap']}
        write_json(a.output/'summary.json',report);write_json(a.output/'status.json',{'stage':'finished','at':now()})
        print(json.dumps({k:v for k,v in report.items() if k not in ('reference_ids','batched_ids','checkpoint')}),flush=True)
        return
    supported=[];skipped=[]
    for size in a.sizes:
        try:
            for offset in range(0,len(prompts),size):
                batch_decode(model,tokenizer,prompts[offset:offset+size],replay=ref_tokens[offset:offset+size])
            supported.append(size)
        except torch.OutOfMemoryError:
            gc.collect();torch.cuda.empty_cache();skipped.append(size)
    write_json(a.output/'status.json',{'stage':'paired_batch_benchmark','at':now(),'supported_sizes':supported,'memory_limited_sizes':skipped})
    def measure(size,mode,repetition):
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        outputs=[];predictions=[];padding_tokens=0
        torch.cuda.synchronize();start=time.perf_counter()
        for offset in range(0,len(prompts),size):
            batch_prompts=prompts[offset:offset+size]
            padding_tokens+=max(map(len,batch_prompts))*len(batch_prompts)-sum(map(len,batch_prompts))
            result=batch_decode(model,tokenizer,batch_prompts,
                replay=ref_tokens[offset:offset+size] if mode=='fixed_work_replay' else None)
            outputs+=result['completion_ids'];predictions+=result['predicted_ids']
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        row={'batch_size':size,'mode':mode,'repetition':repetition,'seconds':seconds,
            'requests':len(prompts),'requests_per_second':len(prompts)/seconds,
            'completion_tokens':sum(map(len,outputs)),'output_tokens_per_second':sum(map(len,outputs))/seconds,
            'identical_requests':sum(x==y for x,y in zip(predictions,ref_tokens)),
            'completion_ids':outputs,'predicted_ids':predictions,'padding_tokens':padding_tokens,
            'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,
            'peak_reserved_GiB':torch.cuda.max_memory_reserved()/1024**3}
        rows.append(row);write_json(a.output/'measurements.json',rows)
        print(json.dumps({k:v for k,v in row.items() if k not in ('completion_ids','predicted_ids')}),flush=True)
    for repetition in range(a.repeats):
        sizes=supported if repetition%2==0 else list(reversed(supported))
        for size in sizes:measure(size,'fixed_work_replay',repetition)
    for size in supported:measure(size,'natural_greedy',0)
    baseline=median(r['seconds'] for r in rows if r['batch_size']==1 and r['mode']=='fixed_work_replay')
    summary={}
    for size in supported:
        timed=[r for r in rows if r['batch_size']==size and r['mode']=='fixed_work_replay']
        natural=next(r for r in rows if r['batch_size']==size and r['mode']=='natural_greedy')
        seconds=median(r['seconds'] for r in timed)
        summary[str(size)]={'fixed_work_median_seconds':seconds,'fixed_work_speedup_vs_shared_serial':baseline/seconds,
            'natural_seconds':natural['seconds'],'natural_identical_requests':natural['identical_requests'],
            'requests':len(prompts),'peak_allocated_GiB':max(r['peak_allocated_GiB'] for r in rows if r['batch_size']==size),
            'peak_reserved_GiB':max(r['peak_reserved_GiB'] for r in rows if r['batch_size']==size),
            'padding_tokens':natural['padding_tokens']}
    report={'at':now(),'checkpoint':identity,'prompt_lengths':list(map(len,prompts)),
        'reference_actor_seconds':sum(r['seconds'] for r in refs),'results':summary,'memory_limited_batch_sizes':skipped,
        'reference_kernels_preserved':True,'formal_runtime_changed':False,'judge_requests':0,
        'limitations':['Static batches using one model copy; not a vLLM/continuous-batching deployment',
          'Shared GPU with 18 production workers; not a controlled comparison against those 18 processes',
          'Fixed-work replay measures identical scheduled output lengths; natural greedy checked separately',
          '96-token cap; no Judge or environment execution in timing; warmup excluded']}
    write_json(a.output/'summary.json',report);write_json(a.output/'status.json',{'stage':'finished','at':now()})
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    main()
