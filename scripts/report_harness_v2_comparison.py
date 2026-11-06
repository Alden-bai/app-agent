#!/usr/bin/env python3
"""Read paired execution evidence and report business outcomes separately from liveness."""
import argparse
from collections import Counter
from pathlib import Path
import sqlite3
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness_awm.io import file_hash, read_json, verify_seal, write_json

RESOURCE_STOPS = {'token_limit', 'context_limit'}


def report(run):
    run = Path(run).resolve()
    m = read_json(run/'manifest.json')
    if file_hash(run/'manifest.json') != read_json(run/'manifest.sha256.json')['sha256']:
        raise ValueError('experiment_manifest_changed')
    failed = [t for t in m['tasks'] if t['baseline']['task_status']=='failed']
    ready = [t for t in m['tasks'] if t['ready']]
    data = {'planned_tasks_per_arm':len(m['tasks']), 'frozen_ready_per_arm':len(ready),
        'baseline':{'confirmed_pass':sum(t['baseline']['task_status']=='confirmed_pass' for t in m['tasks']),
            'failed':len(failed),
            'failed_zero_calls':sum(t.get('baseline_execution',{}).get('tool_calls')==0 for t in failed),
            'failed_empty_answers':sum(t.get('baseline_empty_answer') is True for t in failed),
            'failed_resource_stops':sum(t.get('baseline_execution',{}).get('termination_reason') in RESOURCE_STOPS for t in failed)},
        'arms':{}}
    for arm in m['configs']:
        rows=[]
        for t in m['tasks']:
            p=run/arm/t['id']/'result.json'
            if not p.exists(): continue
            r=read_json(p); e=r.get('execution'); ev=r.get('evaluation',{})
            if e:
                verify_seal(p.parent/'episode')
                trace=read_json(p.parent/'episode/trajectory.json')
                if trace['finished'] and not trace['final_answer'].strip():
                    raise ValueError('empty_finished_answer:'+str(p))
            rows.append({'id':t['id'],'ready':t['ready'],'old':t['baseline']['task_status'],
                'new':ev.get('task_status','undetermined'),'evaluation_status':ev.get('evaluation_status'),
                'execution':e,'empty_answer':r.get('empty_answer'),'error':r.get('error'),
                'judge_usage':ev.get('usage',{}),'result_path':str(p)})
        executed=[r for r in rows if r['execution']]
        old_failed=[r for r in executed if r['old']=='failed']
        statuses=Counter(r['new'] for r in rows if r['ready'])
        changes=[{'id':r['id'],'new_status':r['new'],'evaluation_status':r['evaluation_status'],
                  'result_path':r['result_path']} for r in rows if r['old']=='confirmed_pass' and r['new']!='confirmed_pass']
        a={'executed':len(executed),'recorded':len(rows),'settled':sum(not r['ready'] or r['evaluation_status'] is not None or bool(r['error']) for r in rows),'evaluated':sum(r['evaluation_status'] is not None for r in rows),
           'ready_task_statuses':dict(statuses),'confirmed_pass':statuses['confirmed_pass'],
           'empty_answers':sum(r['empty_answer'] is True for r in executed),
           'zero_calls':sum(r['execution']['tool_calls']==0 for r in executed),
           'failed_cohort_covered':len(old_failed),
           'failed_zero_calls':sum(r['execution']['tool_calls']==0 for r in old_failed),
           'failed_empty_answers':sum(r['empty_answer'] is True for r in old_failed),
           'failed_resource_stops':sum(r['execution']['termination_reason'] in RESOURCE_STOPS or
                r['execution'].get('execution_metrics',{}).get('finalization_trigger') in RESOURCE_STOPS for r in old_failed),
           'terminations':dict(Counter(r['execution']['termination_reason'] for r in executed)),
           'finalization_triggers':dict(Counter(r['execution']['execution_metrics']['finalization_trigger'] for r in executed
                                                if r['execution']['execution_metrics'].get('finalization_trigger'))),
           'completion_claims':dict(Counter(r['execution'].get('completion_status') or 'not_finished' for r in executed)),
           'previous_pass_changes':changes,
           'actor_reported_tokens':sum(r['execution']['execution_metrics']['reported_tokens'] for r in executed),
           'actor_estimated_tokens':sum(r['execution']['execution_metrics']['estimated_tokens'] for r in executed),
           'judge_tokens_accounted_or_reserved':sum(r['judge_usage'].get('tokens_accounted_or_reserved',0) for r in rows),
           'errors':[{'id':r['id'],'error':r['error']} for r in rows if r['error']],
           'execution_seals_verified':len(executed)}
        data['arms'][arm]=a
    with sqlite3.connect((run/'budget.sqlite').as_uri()+'?mode=ro',uri=True) as db:
        count,total,unknown=db.execute("SELECT count(*),coalesce(sum(reserved),0),coalesce(sum(status!='response'),0) FROM requests").fetchone()
        data['budget']={'physical_requests':count,'tokens_accounted_or_reserved':total,'unknown_or_inflight':unknown,
                        'stages':[{'stage':s,'requests':n,'tokens_accounted_or_reserved':v} for s,n,v in db.execute('SELECT stage,count(*),sum(reserved) FROM requests GROUP BY stage')]}
    snapshots=read_json(run/'implementation_snapshots.json') if (run/'implementation_snapshots.json').exists() else {}
    data['implementation_hashes_verified']=all(file_hash(p)==h or (p in snapshots and snapshots[p]['sha256']==h and file_hash(run/snapshots[p]['path'])==h) for p,h in m['implementation_hashes'].items())
    data['working_tree_matches_executed_driver']=all(file_hash(p)==h for p,h in m['implementation_hashes'].items())
    data['complete']=all(a['settled']==len(m['tasks']) and a['executed']==len(ready) for a in data['arms'].values())
    write_json(run/'comparison.json',data)
    b=data['baseline']; o=data['arms']['original_budget']; x=data['arms']['expanded_budget']
    lines=['# Harness V2 执行对照','', ('完整对照已结束。' if data['complete'] else '对照进行中；当前结果仅覆盖部分样本。'), '',
        f"计划每组 {data['planned_tasks_per_arm']} 题，其中 {len(ready)} 题有可信冻结初态；其余样本保留分母并单列，不重建世界。",'',
        '| 指标 | 历史基线 | V2 原预算 | V2 扩大预算 |','|---|---:|---:|---:|',
        f"| 已执行 | {len(ready)} | {o['executed']} | {x['executed']} |",
        f"| Judge 确认通过 | {b['confirmed_pass']} | {o['confirmed_pass']} | {x['confirmed_pass']} |",
        f"| 原 46 条失败中的零调用 | {b['failed_zero_calls']} | {o['failed_zero_calls']} | {x['failed_zero_calls']} |",
        f"| 原 46 条失败中的空答案 | {b['failed_empty_answers']} | {o['failed_empty_answers']} | {x['failed_empty_answers']} |",
        f"| 原 46 条失败中的 token／上下文停止或收尾 | {b['failed_resource_stops']} | {o['failed_resource_stops']} | {x['failed_resource_stops']} |",'',
        '资源停止包括提前收尾触发原因，不能因最终输出 finish 就把预算瓶颈隐藏掉。系统终止报告不是模型答案；模型自报 completed 不是 Judge 确认通过。', '',
        f"原失败队列覆盖：原预算 {o['failed_cohort_covered']}/46，扩大预算 {x['failed_cohort_covered']}/46。",'',
        '## 原成功样本复核','']
    for arm,a in data['arms'].items():
        lines += [f"- {arm}: " + (', '.join(c['id']+' → '+c['new_status']+' ('+str(c['evaluation_status'])+')' for c in a['previous_pass_changes']) or '暂无未确认通过的原成功样本。')]
    lines += ['', '未确定／暂定通过属于待复核，不等于已证明业务失败。逐题结果路径和细分状态见 comparison.json、summary.json。', '',
        '## 用量与限制','',f"共享预算已占用 {total:,} token，{count:,} 次物理请求；未知或进行中用量请求 {unknown} 条。",'',
        '固定了任务、初态和模型配置；实际时钟未冻结，模型采样不确定，历史 Judge 标签也不是人工真值。该对照用于诊断 harness 改善，不声称独立因果准确率。', '',
        '代码与离线验证：119 项相关测试通过；80 份历史工具目录中的 2,803 个工具通过名称可达、精确搜索及 schema 无损回读验证。']
    (run/'comparison.md').write_text('\n'.join(lines)+'\n')
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    value=report(args.run)
    print({k:v for k,v in value.items() if k not in ('arms','budget')})
