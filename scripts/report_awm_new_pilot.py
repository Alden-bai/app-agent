#!/usr/bin/env python3
"""Offline, source-bound export and readable report for the new-world pilot."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from harness_awm.io import canonical, file_hash, read_json, write_json
from awm_new_pilot import DEFAULT_RUN, freeze, load_manifest


def export(run):
    run=Path(run).resolve()
    m=load_manifest(run,False)
    audit=read_json(run/'final_audit.json') if (run/'final_audit.json').exists() else {}
    exclusions=audit.get('excluded_tasks',{})
    root=Path(m['data'])
    worlds=sorted((root/'worlds').glob('*/tasks.json'))
    built_worlds=0
    assets={k:[] for k in ['gen_scenario','gen_tasks','gen_db','gen_sample','gen_spec','gen_envs','gen_verifier','accepted_rubrics','question_bank']}
    if (root/'scenarios.json').exists():
        assets['gen_scenario']=read_json(root/'scenarios.json')
    diagnostics=[]
    for tasks_path in worlds:
        folder=tasks_path.parent
        ready=folder/'ready.json'
        if ready.exists():
            built_worlds+=1
            for rel,expected in read_json(ready)['hashes'].items():
                if file_hash(folder/rel)!=expected:
                    raise ValueError('export_source_changed:'+str(folder/rel))
        scenario=folder.name
        tasks=read_json(folder/'tasks.json')['tasks']
        assets['gen_tasks'].append(dict(scenario=scenario,tasks=tasks))
        if (folder/'schema.json').exists():
            assets['gen_db'].append(dict(scenario=scenario,db_schema=read_json(folder/'schema.json'),db_path=str(folder/'initial.db')))
        if (folder/'sample.json').exists():
            assets['gen_sample'].append(dict(scenario=scenario,sample_data=read_json(folder/'sample.json')))
        if (folder/'spec.json').exists():
            assets['gen_spec'].append(dict(scenario=scenario,api_spec=read_json(folder/'spec.json')))
        if (folder/'environment.json').exists():
            assets['gen_envs'].append(read_json(folder/'environment.json'))
        assets['gen_verifier'].extend(read_json(p) for p in sorted((folder/'verifiers').glob('*.json')))
        for index,instruction in enumerate(tasks):
            rp=folder/'rubrics'/f'{index:02d}'
            review=read_json(rp/'result.json') if (rp/'result.json').exists() else {'status':'not_generated'}
            ep=run/'worlds'/scenario/'tasks'/f'{index:02d}'
            result=read_json(ep/'result.json') if (ep/'result.json').exists() else {}
            row=dict(task_id=f'{scenario}:{index}',scenario=scenario,index=index,task=instruction,
                     rubric_status=review['status'],rubric_review=review,execution=result.get('execution'),
                     evaluation=result.get('evaluation'),error=result.get('error'),
                     original_ref=str(rp/'original.json'),rubric_ref=str(rp/'rubric.json'),episode_ref=str(ep/'episode'))
            if row['task_id'] in exclusions:
                row['audit_override']=exclusions[row['task_id']]
                row['rubric_status']='audit_rejected'
            if (rp/'original.json').exists():
                row['original_sha256']=file_hash(rp/'original.json')
            if row['rubric_status']=='passed':
                row['rubric_sha256']=file_hash(rp/'rubric.json')
                assets['accepted_rubrics'].append(dict(task_id=row['task_id'],rubric=read_json(rp/'rubric.json'),
                    source_artifact_ref=str(rp/'rubric.json'),source_artifact_sha256=row['rubric_sha256']))
            assets['question_bank'].append({k:v for k,v in row.items() if k not in ('execution','evaluation')})
            diagnostics.append(row)
    # The identity includes current task results so later audited versions never overwrite exports.
    from harness_awm.io import digest
    version=digest(dict(assets=assets,diagnostics=diagnostics))[:16]
    dest=root/'exports'/version
    dest.mkdir(parents=True,exist_ok=True)
    for name,records in assets.items():
        path=dest/(name+'.jsonl')
        content=''.join(canonical(r)+'\n' for r in records)
        if path.exists() and path.read_text()!=content:
            raise ValueError('export_collision')
        path.write_text(content)
    freeze(dest/'manifest.json',dict(schema_version='awm-new-pilot-export-v1',split='development',
        scenario_candidates=len(assets['gen_scenario']),built_environments=built_worlds,tasks=len(diagnostics),
        rubric_accepted=len(assets['accepted_rubrics']),formal_admission=False,
        reward_ready=False,source_manifest_sha256=file_hash(run/'manifest.json'),
        files={p.name:dict(path=str(p),sha256=file_hash(p)) for p in sorted(dest.glob('*.jsonl'))}))
    write_json(run/'task_diagnostics.json',diagnostics)
    with sqlite3.connect((run/'budget.sqlite').as_uri()+'?mode=ro',uri=True) as db:
        usage=[dict(stage=s,status=status,category=category,calls=n,tokens=t) for s,status,category,n,t in
               db.execute("SELECT stage,status,CASE WHEN request_key LIKE '%/completion_repair%' OR request_key LIKE '%/r1%' OR request_key LIKE '%/json_contract_v1%' THEN 'repair' ELSE 'initial' END AS category,count(*),sum(reserved) FROM requests GROUP BY stage,status,category")]
        reported=[json.loads(row[0]) for row in db.execute("SELECT usage FROM requests WHERE status='response' AND usage IS NOT NULL")]
    actual_tokens=sum(u.get('total_tokens',0) for u in reported)
    cached_tokens=sum(u.get('prompt_tokens_details',{}).get('cached_tokens',0) for u in reported)
    evaluations=[r['evaluation'] for r in diagnostics if r['evaluation'] and not r.get('audit_override')]
    valid=[e for e in evaluations if e['evaluation_status']=='valid']
    labels=Counter(e['business_label'] for e in valid)
    statuses=Counter(e['evaluation_status'] for e in evaluations)
    statuses['audited_rubric_error']=sum(bool(r.get('audit_override')) for r in diagnostics if r['evaluation'])
    unknown_reasons=Counter(item['unknown']['kind'] for e in evaluations for item in e.get('items',[])
                            if item.get('unknown'))
    gate_path=run/('pilot_gate.audited.json' if (run/'pilot_gate.audited.json').exists() else 'pilot_gate.json')
    gate=read_json(gate_path) if gate_path.exists() else {'passed':False,'reasons':['pilot_not_finished']}
    accepted=[r for r in diagnostics if r['rubric_status']=='passed']
    all_twos=sum(set(r['rubric_review']['scores'].values())=={2} for r in accepted)
    ones=len(accepted)-all_twos
    lines=['# AWM 新场景端到端试跑结果','',
        f"目标 **10 场景、100 题**；候选场景 {len(assets['gen_scenario'])} 个，已构建环境 {built_worlds} 个，已生成 {len(diagnostics)} 题，rubric 审查通过 {len(assets['accepted_rubrics'])} 题。",
        f"有效判分 {len(valid)} 题：确认通过 {labels['pass']}，失败 {labels['fail']}，未知/暂定通过 {labels['unknown']}。评测状态：`{dict(statuses)}`。",
        f"逐条 unknown 原因分布：`{dict(unknown_reasons)}`。",
        f"已知或预留总用量 **{sum(r['tokens'] for r in usage):,} / {m['max_total_tokens']:,} tokens**；未返回用量保守预留，不能视为实际账单。",
        f"其中 API 已返回实际用量 {actual_tokens:,} tokens，缓存输入 {cached_tokens:,} tokens（已包含在实际用量中）；未知用量预留 {sum(r['tokens'] for r in usage)-actual_tokens:,} tokens。",
        f"Rubric 通过率 {len(accepted)}/{len(diagnostics)}；五维全 2 分 {all_twos} 题，含 1 分 {ones} 题。有效判定成功率 {labels['pass']}/{len(valid)}；unknown 不计成功。",
        f"首批扩量门槛：`{gate}`。",'',
        '本批全部为开发数据。未知不算成功；故障不算模型业务失败。未训练 GRPO、未导出奖励，也不声称人工金标准准确率。',
        f"\n[兼容格式导出]({dest}/manifest.json) · [逐题机器可读诊断]({run}/task_diagnostics.json) · [回归日志]({run}/regression.log)",
        '\n## 分阶段用量','', '| 阶段 | 首次/修复 | 请求状态 | 调用数 | tokens（实际或预留） |','|---|---|---|---:|---:|']
    lines += [f"| {r['stage']} | {r['category']} | {r['status']} | {r['calls']} | {r['tokens']:,} |" for r in usage]
    if audit:
        lines += ['\n## 验收复核与停止原因','',audit['conclusion'],
                  f"\n[完整复核记录]({run}/final_audit.json) · [最终回归]({run}/regression.final.log)"]
        lines += ['\n'+item for item in audit.get('findings',[])]
        lines += ['\n## 具体案例']
        for example in audit.get('examples',[]):
            lines += ['\n**'+example['kind']+'**：'+example['description'],f"\n[证据]({example['evidence_ref']})"]
    lines += ['\n## 逐题结果','', '| 题目 | Rubric | 评测 | 业务结果 |','|---|---|---|---|']
    for row in diagnostics:
        e=row['evaluation'] or {}
        lines.append(f"| {row['task_id']} | {row['rubric_status']} | {'audited_rubric_error' if row.get('audit_override') else e.get('evaluation_status','not_run')} | {'不计入业务结果' if row.get('audit_override') else e.get('business_label','—')} |")
    lines += ['\n## 题面、规则与判分证据']
    for row in diagnostics:
        lines += [f"\n### {row['task_id']}",'',row['task'],
            f"\nRubric 审查：`{row['rubric_review']}`。"]
        if Path(row['original_ref']).exists():
            lines.append(f"\n[来源]({row['original_ref']})")
        if (Path(row['episode_ref'])/'trajectory.json').exists():
            lines.append(f"\n[执行轨迹]({row['episode_ref']}/trajectory.json)")
        if row['rubric_status']=='passed':
            r=read_json(row['rubric_ref'])
            lines.append(f"\n[完整五字段 rubric]({row['rubric_ref']})")
            for item in r['rubric']:
                lines += [f"\n**{item['id']}**",item['criterion']]
            if r['unresolved']:
                lines += ['\n运行时待核查条件：', '```json',json.dumps(r['unresolved'],ensure_ascii=False,indent=2),'```']
        e=row['evaluation']
        if e:
            if row.get('audit_override'):
                lines += [f"\n复核排除：{row['audit_override']}。下列原始 Judge 输出保留供追溯，不计入有效业务结果。"]
            lines += [f"\nJudge：**{e['task_status']} / {e['business_label']}**，评测状态 `{e['evaluation_status']}`。"]
            if e.get('attempt'):
                lines += [f"\n[逐项判定与证明]({e['attempt']}/result.json) · [完整证据]({e['attempt']}/evidence.json)"]
            if e.get('errors'):
                lines += [f"\n评测错误：`{e['errors']}`"]
            for item in e.get('items',[]):
                if item.get('unknown'):
                    u=item['unknown']
                    lines += [f"\n- {item.get('id',item.get('item_id',''))}：{u.get('kind')}；缺口：{u.get('missing_fact')}；影响：{u.get('impact')}；下一步：{u.get('next_action')}"]
    (run/'results_zh.md').write_text('\n'.join(lines)+'\n')
    return dict(export=str(dest),report=str(run/'results_zh.md'),tasks=len(diagnostics),rubric_accepted=len(assets['accepted_rubrics']),valid=len(valid),labels=dict(labels))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN)
    print(canonical(export(parser.parse_args().run)))
