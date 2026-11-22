"""Launch the explicitly authorized full frozen training pool with shared Judge calibration."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from dotenv import load_dotenv
from harness_awm.contracts import ROOT
from harness_awm.io import file_hash, read_json, write_json
from harness_awm.training.local import checkpoint_identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--scratch',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--calibrated',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--control',type=Path,required=True)
    target=p.add_mutually_exclusive_group()
    target.add_argument('--target-updates',type=int)
    target.add_argument('--steps',type=int,help='Total rollout groups, including groups with no parameter update')
    p.add_argument('--save-every',type=int,default=200)
    args=p.parse_args()
    if args.steps is None and args.target_updates is None:args.target_updates=800
    limit=args.steps if args.steps is not None else args.target_updates
    if not 1 <= limit <= 3200 or args.save_every < 1:
        p.error('steps/target-updates must be 1..3200 and save-every must be positive')
    target_flags=['--steps',args.steps] if args.steps is not None else ['--target-updates',args.target_updates]
    run,scratch,control=args.run.resolve(),args.scratch.resolve(),args.control.resolve()
    run.mkdir(parents=True,exist_ok=False);scratch.mkdir(parents=True,exist_ok=False)
    (control/'persistent').symlink_to(run,target_is_directory=True)
    (control/'runtime').symlink_to(scratch,target_is_directory=True)
    (run/'training').symlink_to(scratch/'training',target_is_directory=True)
    load_dotenv(ROOT/'.env',override=False)
    cache=Path(sys.prefix).parent/'kernel-cache'
    (cache/'tmp').mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             TOKENIZERS_PARALLELISM='false',SWANLAB_MODE='disabled',TMPDIR=str(cache/'tmp'),
             RAY_TMPDIR='/dev/shm/awm-gfull-ray',TRITON_CACHE_DIR=str(cache/'triton'),
             TORCHINDUCTOR_CACHE_DIR=str(cache/'torchinductor'),VLLM_CACHE_ROOT=str(cache/'vllm'))
    state=dict(pid=os.getpid(),status='running',stage='bootstrap',training_tasks=2746,target_updates=args.target_updates,save_every=args.save_every,
               target_steps=args.steps,save_unit="groups" if args.steps is not None else "updates",
               max_groups=args.steps if args.steps is not None else 3200,calibration_scope='shared-suite',per_task_calibrated=False)
    def status(**kw):
        state.update(kw,updated_at=time.time());write_json(run/'status.json',state)
    def persist():
        train=scratch/'training'
        if not train.exists():return
        for path in train.glob('*.json'):
            try:write_json(run/'records'/path.name,read_json(path))
            except (FileNotFoundError,json.JSONDecodeError):pass
        progress=train/'progress.json'
        if not progress.exists():return
        d=read_json(progress)
        status(groups=d['groups'],optimizer_steps=d['optimizer_steps'])
        for group in sorted((train/'groups').glob('group-*')):
            if int(group.name.split('-')[1])>=d['groups']-2:continue
            archive=run/'archives'/(group.name+'.tar.gz');manifest=archive.with_suffix('.manifest.json')
            if not manifest.exists():continue
            info=read_json(manifest)
            if archive.is_file() and archive.stat().st_size==info['bytes'] and info['files_verified']>0:
                write_json(run/'group-records'/(group.name+'.json'),read_json(group/'update.json'))
                shutil.rmtree(group)
    def command(name,argv,*,gpu=False,timeout=7200):
        status(stage=name)
        with (run/(name+'.log')).open('x') as log:
            proc=subprocess.Popen([sys.executable,*map(str,argv)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                  stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status(child_pid=proc.pid)
            start=time.monotonic()
            try:
                while proc.poll() is None:
                    if time.monotonic()-start>timeout:raise RuntimeError('stage time limit: '+name)
                    if gpu:
                        raw=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
                        used,util=map(int,raw.strip().split(','));row=dict(at=time.time(),used_mib=used,used_gb=used*1024**2/1e9,utilization=util)
                        with (run/'gpu.jsonl').open('a') as h:h.write(json.dumps(row)+'\n')
                        status(gpu=row)
                        if row['used_gb']>=90:raise RuntimeError('GPU memory guard: '+str(row))
                    persist();time.sleep(5)
                persist()
                if proc.returncode:raise RuntimeError(name+' exited '+str(proc.returncode)+'; inspect '+name+'.log')
            except BaseException:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                raise
    with (ROOT/'runs/verl_grpo_smoke.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            status()
            from harness_awm.training.verl.pilot import resource_check
            resource_check(run,free_gib=100,gpu_limit_gb=1)
            resource_check(scratch,free_gib=8,gpu_limit_gb=1)
            selection=read_json(ROOT/'runs/sft_best_test_20261110_v1/selection.json')
            actual=checkpoint_identity(args.model)
            if actual['files']!=selection['checkpoint']['files']:raise ValueError('selected SFT checkpoint bytes changed')
            write_json(run/'model-verification.json',dict(source=selection['checkpoint'],cache=actual,matched=True))
            config=read_json(args.config)
            config.update(recover_tool_errors=True,judge_rate_db=str(scratch/'judge-rate.sqlite'),judge_rpm=5000,judge_tpm=2000000)
            write_json(run/'config.json',config)
            for source in [Path(__file__),ROOT/'scripts/prepare_full_grpo_admissions.py']:
                shutil.copy2(source,run/source.name)
            shutil.copy2(control/'authorization.json',run/'authorization.json')
            command('admissions',[ROOT/'scripts/prepare_full_grpo_admissions.py','--tasks',ROOT/'data/training_runtime_20261106_v1',
                    '--config',run/'config.json','--calibrated',args.calibrated,'--output',scratch/'admissions','--expected-tasks','2746'])
            for source in (scratch/'admissions').glob('*.json'):shutil.copy2(source,run/('admissions-'+source.name))
            command('prepare',['-m','harness_awm.training.verl','prepare','--preset','single-gpu-800',
                    '--config',scratch/'admissions/config.json','--tasks',ROOT/'data/training_runtime_20261106_v1',
                    '--model',args.model,'--output',scratch/'training','--checkpoint-dir',run/'checkpoints',
                    *target_flags,'--save-every',args.save_every,
                    '--shuffle-tasks','--gpu-memory-utilization','.82','--allocator-gb','85','--gpu-stop-gb','90','--pilot-guards'])
            plan=read_json(scratch/'training/plan.json')
            assert len(plan['tasks'])==2746 and len({t['task_instance_id'] for t in plan['tasks']})==2746
            assert all(t['split']=='train' for t in plan['tasks'])
            assert (plan['group_size'],plan['target_updates'],plan['save_every'],plan['null_reward_policy'])==(4,args.target_updates,args.save_every,'zero')
            if args.steps is not None:assert plan['groups']==args.steps and plan['save_unit']=='groups'
            assert plan['task_order']=='seeded-shuffle' and plan['run_config']['recover_tool_errors']
            command('check',['-m','harness_awm.training.verl','check','--run',scratch/'training'])
            # The same SFT bytes, local-sdpa kernels and aggregate memory budget
            # passed the 32K forward/backward + weight-sync diagnostic this session.
            proof=ROOT/'runs/grpo_800_20261113_v1/persistent/early-probe-report.json'
            d=read_json(proof)
            assert d['passed'] and d['weight_sync'] and d['optimizer_steps']==1 and d['sequence_tokens']>=32736
            write_json(run/'gpu-diagnostic-reference.json',dict(source=str(proof.resolve()),sha256=file_hash(proof),report=d,
                scope='GPU/kernel diagnostic only. Full-pool reward admissions are checked independently.'))
            command('train',['-u','-m','harness_awm.training.verl','train','--run',scratch/'training'],gpu=True,timeout=1209600)
            command('verify',[ROOT/'scripts/verify_verl_grpo_checkpoint.py','--run',scratch/'training'],gpu=True)
            status(status='complete',stage='complete')
        except BaseException as exc:
            status(status='needs_attention',error=repr(exc));raise
        finally:persist()


if __name__=='__main__':main()
