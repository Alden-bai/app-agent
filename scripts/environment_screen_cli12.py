"""Continue the frozen pilot using 12 independent Codex CLI verification sessions."""
import asyncio
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

from environment_screen_pilot100 import RUN, ROOT
from harness_awm.io import file_hash, now, read_json, write_json

MODEL='gpt-6-luna'
EFFORT='high'
WORKERS=12
PYTHON=ROOT/'agent-world-model/.venv/bin/python'
HELPER=ROOT/'scripts/environment_screen_pilot100.py'

SUPPLEMENT='''
本轮执行补充（与前面判定原则一致）：
- 只处理当前 case 一题。禁止启动其他 agent、模型或 CLI 子会话。禁止读取相邻 case、private、历史运行或选择标签。
- “初态已有目标记录”不能自动替代明确的“新建/登记”义务。核查必要写入路径可用性；若公开创建能力可用，已有记录也不自动排除。缺少 actor 可公开发现的必需 ID 不能用 SQL 偷填后宣称通过。
- 这是模拟环境；不要求模拟工具真的访问现实第三方。核查契约内是否存在等价模拟能力及有效结果；不要仅因没有真实外网请求就排除。
- 只读取当前目录材料和指定 probe 助手。允许用 Python sqlite3 以 mode=ro 查询初态库并将 SQL/结果记录为证据；不得执行环境源码，实际调用只能通过指定 probe 助手的隔离环境。
- 每题优先 1–3 个探测计划；已有相关 baseline 实测或确定代码+数据证据时收束，避免穷尽。可修正自己构造的错误参数。
- 完成后同时保存 decision.first_pass.json 和 decision.json。已有 first_pass 时勿覆盖。不要修改冻结输入，不要读取主 agent 汇总结果。
'''


def usage(thread_id):
    if not thread_id:return {'actual_config_verified':False,'reason':'missing thread id'}
    with sqlite3.connect('file:/home/bjx/.codex/state_5.sqlite?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        row=db.execute('SELECT model,reasoning_effort,rollout_path FROM threads WHERE id=?',(thread_id,)).fetchone()
    if row is None:return {'actual_config_verified':False,'reason':'missing metadata'}
    total=None;contexts=[]
    for line in Path(row['rollout_path']).open():
        event=json.loads(line);payload=event.get('payload',{})
        if event.get('type')=='turn_context':contexts.append((payload.get('model'),payload.get('effort')))
        if event.get('type')=='event_msg' and payload.get('type')=='token_count' and payload.get('info'):
            total=payload['info'].get('total_token_usage') or total
    return dict(thread_id=thread_id,model=row['model'],reasoning_effort=row['reasoning_effort'],
                actual_config_verified=(row['model'],row['reasoning_effort'])==(MODEL,EFFORT)
                    and bool(contexts) and all(x==(MODEL,EFFORT) for x in contexts),actual_usage=total)


async def main():
    run=RUN/'cli12';run.mkdir(exist_ok=True)
    if (run/'execution.json').exists():raise RuntimeError('existing execution: reconcile before restart')
    pending=[r for r in read_json(RUN/'cases.json') if not (Path(r['case_dir'])/'decision.json').exists()]
    write_json(run/'execution.json',dict(started_at=now(),model=MODEL,reasoning_effort=EFFORT,
        requested_concurrency=WORKERS,cases=[r['case_id'] for r in pending],
        mechanism='independent codex exec per case; no resume/fork; same account authentication',
        permission_mode='--approve-for-me; workspace-write with case as cwd; isolated runtime probes',
        prior_decisions_preserved=True,prompt_sha256=file_hash(RUN/'prompt_zh.txt'),implementation_sha256=file_hash(__file__)))
    (run/'supplement_zh.txt').write_text(SUPPLEMENT)
    active={};completed=[];peak=0;stop=False
    queue=asyncio.Queue()
    for row in pending:queue.put_nowait(row)

    def status():
        write_json(run/'status.json',dict(updated_at=now(),active=active,completed=completed,
            peak_concurrency=peak,requested_concurrency=WORKERS,remaining=queue.qsize(),stop_dispatch=stop))

    async def worker():
        nonlocal peak,stop
        while not queue.empty() and not stop:
            row=queue.get_nowait();cid=row['case_id'];case=Path(row['case_dir'])
            if (case/'decision.json').exists():raise RuntimeError('concurrent ownership conflict:'+cid)
            folder=run/cid;folder.mkdir(exist_ok=False)
            message=(RUN/'prompt_zh.txt').read_text()+SUPPLEMENT+f'''
当前唯一任务目录：{case}
先读 task.json、contract.json 和必要代码/数据库。baseline/ 已包含实际工具探测证据。
实际探测：将计划 JSON 保存到当前 case，运行：
{PYTHON} {HELPER} probe --case {case} --plan <本题计划绝对路径>
计划格式：{{"calls":[{{"tool":"公开工具名","arguments":{{}}}}],"queries":[{{"sql":"SELECT ...","params":[]}}]}}
calls 最多 16 项，按顺序共享同一新副本；后续参数可用 {{"$result":0,"path":["id"]}} 引用先前公开结果。
每次调用自动生成 probes/pNNN/result.json。宿主工具如受 sandbox 阻止启动运行时，可按现有工具权限机制请求提升执行这个助手，环境内部仍由 bwrap 隔离。
返回结果后检查实际响应/写入，保存判定与 SQL 证据到本题目录。最终消息简述 case ID、判定和证据路径即可。
'''
            (folder/'prompt.txt').write_text(message)
            command=['codex','exec','--skip-git-repo-check','--ignore-user-config','-m',MODEL,
                '-c','model_reasoning_effort="high"','--approve-for-me','--json','-C',str(case),
                '-o',str(folder/'response.txt'),'-']
            active[cid]=dict(started_at=now());peak=max(peak,len(active));status()
            started=time.monotonic();error=None;returncode=None
            with (folder/'events.jsonl').open('w') as stdout,(folder/'stderr.txt').open('w') as stderr:
                proc=await asyncio.create_subprocess_exec(*command,stdin=asyncio.subprocess.PIPE,stdout=stdout,stderr=stderr,cwd=case,start_new_session=True)
                active[cid]['pid']=proc.pid;status()
                try:
                    await asyncio.wait_for(proc.communicate(message.encode()),timeout=1200)
                    returncode=proc.returncode
                except asyncio.TimeoutError:
                    import signal
                    os.killpg(proc.pid,signal.SIGTERM)
                    await proc.wait();error='session timeout; no automatic retry'
            thread_id=None;turn_done=False;errors=[]
            for line in (folder/'events.jsonl').read_text().splitlines():
                try:event=json.loads(line)
                except ValueError:continue
                if event.get('type')=='thread.started':thread_id=event.get('thread_id')
                if event.get('type')=='turn.completed':turn_done=True
                if event.get('type') in ['error','turn.failed']:errors.append(event)
            actual=usage(thread_id)
            valid=returncode==0 and turn_done and actual.get('actual_config_verified') and (case/'decision.json').exists()
            for name,sha in read_json(case/'asset_hashes.json').items():
                if file_hash(case/name)!=sha:valid=False;error='frozen input changed:'+name
            result=dict(case_id=cid,valid=bool(valid),returncode=returncode,error=error,errors=errors,
                elapsed_seconds=round(time.monotonic()-started,2),finished_at=now(),**actual)
            write_json(folder/'completed.json',result)
            if not valid:
                stop=True
            completed.append(dict(case_id=cid,valid=bool(valid)));active.pop(cid);status()
            print(json.dumps(dict(case_id=cid,valid=bool(valid),completed=len(completed),active=len(active),remaining=queue.qsize()),ensure_ascii=False),flush=True)
    await asyncio.gather(*(worker() for _ in range(WORKERS)))
    status()
    if stop:sys.exit(2)


if __name__=='__main__':asyncio.run(main())
