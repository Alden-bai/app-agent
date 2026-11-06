#!/usr/bin/env python3
"""Frozen 50+50 blind native-subagent evaluation; no model API client."""
import argparse
import asyncio
from collections import Counter
import csv
import json
from pathlib import Path
import random
import re
import shutil
import sqlite3
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from harness_awm.io import db_fingerprint,digest,file_hash,now,read_json,write_json
from screen_bank_environments import dump_rows,rows

RUN=ROOT/'runs/environment_screen_luna100_20261104_v1'
SOURCE=ROOT/'runs/bank3000_execution_20261104_v1'
PREVIOUS=ROOT/'runs/environment_screen_20261104_v2'
RELEASE=ROOT/'data/awm-dde80a0-repaired-v1'
PROMPT=ROOT/'configs/environment_screen/pilot100_v1/prompt_zh.txt'


def prepare():
    RUN.mkdir(exist_ok=False)
    (RUN/'private').mkdir();(RUN/'cases').mkdir()
    source_manifest=read_json(SOURCE/'manifest.json')
    candidates=[]
    for task in source_manifest['tasks']:
        path=SOURCE/'tasks'/task['id']/'result.json'
        if not path.exists():continue
        result=read_json(path);label=result['selection']['label']
        if label in {'valid_success','environment_failure','tool_failure','task_data_failure'}:
            candidates.append(dict(**task,label=label,result_ref=str(path),result_sha256=file_hash(path)))
    rng=random.Random(2026110401);rng.shuffle(candidates)
    chosen=[];used=set()
    # Maximize independent scenario coverage. Explicit stratification is frozen before verification.
    for label,wanted in [('environment_failure',10),('tool_failure',20),('task_data_failure',20),('valid_success',50)]:
        selected=0
        for row in candidates:
            if row['label']!=label or row['scenario'] in used:continue
            chosen.append(row);used.add(row['scenario']);selected+=1
            if selected==wanted:break
        if selected<wanted:raise ValueError(f'not enough distinct scenarios for {label}')
    rng.shuffle(chosen)
    tasks_by_scenario={r['scenario']:r['tasks'] for r in rows(RELEASE/'gen_tasks.jsonl')}
    specs={r['scenario']:r for r in rows(RELEASE/'gen_spec.jsonl')}
    schemas={r['scenario']:r for r in rows(RELEASE/'gen_db.jsonl')}
    static_manifest=read_json(PREVIOUS/'manifest.json')
    baseline=RUN/'baseline';baseline.mkdir();(baseline/'scenarios').mkdir()
    baseline_manifest={**static_manifest,'started_at':now(),'scenarios':sorted(used),'total_questions':100,
        'remaining':100,'scope':'100 selected scenarios only; no full bank sweep',
        'implementation_hashes':{p:file_hash(p) for p in static_manifest['implementation_hashes']}}
    write_json(baseline/'manifest.json',baseline_manifest)
    private=[];public=[];inventory=[]
    for i,row in enumerate(chosen):
        cid=f'C{i:03d}';folder=RUN/'cases'/cid;folder.mkdir()
        scenario=row['scenario'];world=SOURCE/'worlds'/scenario
        task=dict(case_id=cid,task_id=row['task_id'],scenario=scenario,index=row['index'],
                  task=tasks_by_scenario[scenario][row['index']],
                  runtime_clock_policy='actual_recorded',purpose='environment_verification_not_actor_score')
        write_json(folder/'task.json',task)
        shutil.copyfile(world/'initial.db',folder/'initial.db')
        env=read_json(world/'environment.json');write_json(folder/'environment.json',env)
        (folder/'environment.py').write_text(env['full_code'])
        write_json(folder/'schema.json',schemas[scenario]);write_json(folder/'contract.json',specs[scenario])
        static=read_json(PREVIOUS/'scenarios'/scenario/'static.json')
        if static['initial_sha256']!=file_hash(folder/'initial.db') or static['environment_sha256']!=digest(env):
            raise ValueError('historical/static asset version mismatch')
        dest=baseline/'scenarios'/scenario;dest.mkdir()
        for name in ['static.json','environment.json','contract.json']:
            shutil.copyfile(PREVIOUS/'scenarios'/scenario/name,dest/name)
        private.append(dict(case_id=cid,group='historical_pass' if row['label']=='valid_success' else 'historical_suspected_fault',**row))
        public.append(dict(case_id=cid,case_dir=str(folder)))
        inventory.append(dict(task_id=row['task_id'],scenario=scenario,index=row['index'],task=task['task'],remaining=True,
            rubric_accepted=True,rubric_status='passed',original_ref=str(folder/'task.json'),original_sha256=file_hash(folder/'task.json'),
            binding_errors=[],semantic_solvability='not_proven'))
        write_json(folder/'asset_hashes.json',{name:file_hash(folder/name) for name in ['task.json','initial.db','environment.json','environment.py','schema.json','contract.json']})
    dump_rows(RUN/'private/selection.jsonl',private)
    dump_rows(baseline/'task_inventory.jsonl',inventory)
    write_json(RUN/'cases.json',public)
    shutil.copyfile(PROMPT,RUN/'prompt_zh.txt')
    for n in range(3):write_json(RUN/f'queue_{n}.json',public[n::3])
    write_json(RUN/'manifest.json',dict(created_at=now(),sample_size=100,groups={'historical_pass':50,'historical_suspected_fault':50},
        suspected_strata={'environment_failure':10,'tool_failure':20,'task_data_failure':20},distinct_scenarios=len(used),
        seed=2026110401,model='gpt-6-luna',reasoning_effort='high',native_subagent_concurrency=3,
        prompt_ref=str(RUN/'prompt_zh.txt'),prompt_sha256=file_hash(RUN/'prompt_zh.txt'),
        selection_sha256=file_hash(RUN/'private/selection.jsonl'),source_manifest_sha256=file_hash(SOURCE/'manifest.json'),
        reference_label_limit='Historical model-assisted classifications, not human gold labels. No target pass rate.',
        direct_model_api_requests=0))
    print(json.dumps(dict(run=str(RUN),cases=100,scenarios=len(used),groups=dict(Counter(r['group'] for r in private)))))


def payload(response):
    if response.get('structuredContent') is not None:return response['structuredContent']
    texts=[c.get('text','') for c in response.get('content',[]) if c.get('type')=='text']
    if len(texts)==1:
        try:return json.loads(texts[0])
        except ValueError:return texts[0]
    return texts


def resolve(value,results):
    if isinstance(value,dict):
        if '$result' in value:
            result=results[value['$result']]
            for key in value.get('path',[]):result=result[key]
            return result
        return {k:resolve(v,results) for k,v in value.items()}
    if isinstance(value,list):return [resolve(v,results) for v in value]
    return value


async def probe(case,plan_path):
    from harness_awm.contracts import RunConfig
    from harness_awm.environment import AWMEnvAdapter
    case=case.resolve();plan=read_json(plan_path);task_data=read_json(case/'task.json')
    hashes=read_json(case/'asset_hashes.json')
    for name,sha in hashes.items():
        if file_hash(case/name)!=sha:raise ValueError('frozen input changed:'+name)
    calls=plan.get('calls',[])
    if not 0<len(calls)<=16:raise ValueError('one plan needs 1..16 calls')
    existing=[int(p.name[1:]) for p in (case/'probes').glob('p*') if re.fullmatch(r'p\d+',p.name)]
    folder=case/'probes'/f'p{max(existing,default=-1)+1:03d}'
    folder.mkdir(parents=True,exist_ok=False);write_json(folder/'plan.json',plan)
    task=SimpleNamespace(scenario=task_data['scenario'],initial_state_ref=str(case/'initial.db'),
        initial_state_hash=hashes['initial.db'],initial_state_fingerprint=db_fingerprint(case/'initial.db'),
        environment_ref=str(case/'environment.json'),environment_code_hash=digest(read_json(case/'environment.json')),
        tool_schema_hash=None,readiness_profile={'runtime_ready':True})
    adapter=AWMEnvAdapter(task,RunConfig(startup_timeout=45,tool_timeout=15,stop_timeout=3),folder)
    result=dict(case_id=task_data['case_id'],started_at=now(),calls=[],queries=[],setup_ok=False)
    resolved=[]
    try:
        await adapter.prepare();result['setup_ok']=True
        for i,call in enumerate(calls):
            args=resolve(call.get('arguments',{}),resolved)
            response=await adapter.call(call['tool'],args)
            data=payload(response);resolved.append(data)
            result['calls'].append(dict(index=i,tool=call['tool'],arguments=args,response=response,parsed=data))
            if response.get('isError') and not plan.get('continue_on_error',False):break
        # Auditor queries are read-only and never supply hidden values to an actor.
        with sqlite3.connect((adapter.runtime/'working.db').resolve().as_uri()+'?mode=ro',uri=True) as db:
            db.execute('PRAGMA query_only=ON')
            for query in plan.get('queries',[]):
                sql=query['sql'];params=resolve(query.get('params',[]),resolved)
                cur=db.execute(sql,params);records=cur.fetchmany(201)
                item=dict(sql=sql,params=params,columns=[c[0] for c in cur.description] if cur.description else [],rows=records[:200],truncated=len(records)>200)
                if 'expected' in query:item['assertion_passed']=records==[tuple(r) for r in query['expected']]
                result['queries'].append(item)
            result['integrity']=db.execute('PRAGMA integrity_check').fetchall()
            result['foreign_keys']=db.execute('PRAGMA foreign_key_check').fetchall()
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
    finally:
        result['stopped']=await adapter.close()
        if adapter.started:adapter.freeze()
        result['initial_unchanged']=file_hash(case/'initial.db')==hashes['initial.db']
    result['finished_at']=now();write_json(folder/'result.json',result)
    print(json.dumps(dict(result_ref=str(folder/'result.json'),setup_ok=result['setup_ok'],error=result.get('error'),
                         calls=[dict(tool=c['tool'],isError=c['response'].get('isError'),parsed=c['parsed']) for c in result['calls']],
                         queries=result['queries']),ensure_ascii=False)[:16000])


def publish_baseline():
    for row in read_json(RUN/'cases.json'):
        folder=Path(row['case_dir']);scenario=read_json(folder/'task.json')['scenario']
        source=RUN/'baseline/scenarios'/scenario
        # Copy rather than link, so the verifier only needs its own case directory.
        target=folder/'baseline'
        if target.exists():raise ValueError('baseline already published')
        shutil.copytree(source,target)
        # Rewrite copied evidence references to the local, blinded copy.
        for p in target.rglob('*.json'):
            text=p.read_text()
            text=text.replace(str(source),str(target))
            p.write_text(text)
    print('published baseline evidence for 100 cases')


def report():
    private=rows(RUN/'private/selection.jsonl');results=[];missing=[]
    cli_execution=read_json(RUN/'cli12/execution.json') if (RUN/'cli12/execution.json').exists() else None
    cli_cases=set(cli_execution['cases']) if cli_execution else set()
    cli_completed=[read_json(p) for p in (RUN/'cli12').glob('C*/completed.json')]
    cli_status=read_json(RUN/'cli12/status.json') if (RUN/'cli12/status.json').exists() else {}
    issue_groups=read_json(RUN/'issue_groups.json')['case_groups'] if (RUN/'issue_groups.json').exists() else {}
    baseline={r['task_id']:r for r in rows(RUN/'baseline/tasks.jsonl')}
    for row in private:
        folder=RUN/'cases'/row['case_id'];path=folder/'decision.json'
        if not path.exists():missing.append(row['case_id']);continue
        decision=read_json(path)
        first_path=folder/'decision.first_pass.json'
        if not first_path.exists():
            shutil.copy2(path,first_path)
        first=read_json(first_path)
        if decision['case_id']!=row['case_id'] or decision['decision'] not in ['pass','exclude','review']:raise ValueError('invalid decision:'+str(path))
        if not decision.get('requirements') or not decision.get('summary'):raise ValueError('missing requirements or summary:'+str(path))
        def check_refs(value):
            if isinstance(value,dict):
                for key,item in value.items():
                    if key in ('evidence_refs','probe_refs'):
                        for ref in item:
                            local=(folder/re.sub(r':\d+(?:-\d+)?$','',ref)).resolve()
                            if not local.is_relative_to(folder.resolve()) or not local.exists():
                                raise ValueError('invalid case evidence reference:'+str(ref))
                    else:check_refs(item)
            elif isinstance(value,list):
                for item in value:check_refs(item)
        check_refs(decision)
        if decision['decision']=='exclude':
            defect=decision.get('defect') or {}
            if defect.get('affects_required_outcome') is not True or not defect.get('evidence_refs') or not defect.get('alternatives_checked'):
                raise ValueError('exclusion lacks task impact/evidence/alternatives:'+str(path))
        for name,sha in read_json(folder/'asset_hashes.json').items():
            if file_hash(folder/name)!=sha:raise ValueError('frozen input changed:'+str(folder/name))
        results.append(dict(case_id=row['case_id'],task_id=row['task_id'],historical_group=row['group'],historical_label=row['label'],
            verifier_dispatch='cli12_independent_case' if row['case_id'] in cli_cases else 'native_three_queues',
            baseline_decision=baseline[row['task_id']]['environment_status'],
            first_decision=first['decision'],review_changed=first['decision']!=decision['decision'],
            issue_group=issue_groups.get(row['case_id'], '可采样' if decision['decision']=='pass' else '待确认' if decision['decision']=='review' else '尚未归类'),
            decision=decision['decision'],category=decision['category'],verification_level=decision['verification_level'],
            summary=decision['summary'],decision_ref=str(path),historical_result_ref=row['result_ref']))
    dump_rows(RUN/'comparison.jsonl',results)
    for outcome in ['pass','exclude','review']:
        dump_rows(RUN/f'{outcome}.jsonl',[r for r in results if r['decision']==outcome])
    if results:
        with (RUN/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    summary=dict(complete=len(results)==100,completed=len(results),missing=missing,model='gpt-6-luna',reasoning_effort='high',
        dispatch_counts=dict(Counter(r['verifier_dispatch'] for r in results)),
        cli_peak_concurrency=cli_status.get('peak_concurrency'),
        cli_verified_sessions=sum(bool(r.get('actual_config_verified')) for r in cli_completed),
        cli_usage_totals={k:sum((r.get('actual_usage') or {}).get(k,0) for r in cli_completed)
            for k in ['input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens','total_tokens']},
        decisions=dict(Counter(r['decision'] for r in results)),groups={g:dict(Counter(r['decision'] for r in results if r['historical_group']==g)) for g in ['historical_pass','historical_suspected_fault']},
        first_pass_decisions=dict(Counter(r['first_decision'] for r in results)),
        first_pass_groups={g:dict(Counter(r['first_decision'] for r in results if r['historical_group']==g)) for g in ['historical_pass','historical_suspected_fault']},
        review_changed_cases=[r['case_id'] for r in results if r['review_changed']],
        excluded_categories=dict(Counter(r['category'] for r in results if r['decision']=='exclude')),
        excluded_issue_groups=dict(Counter(r['issue_group'] for r in results if r['decision']=='exclude')),
        review_transitions=dict(Counter(r['first_decision']+' -> '+r['decision'] for r in results if r['review_changed'])),
        verification_levels=dict(Counter(r['verification_level'] for r in results)),
        baseline_groups={g:dict(Counter(r['baseline_decision'] for r in results if r['historical_group']==g)) for g in ['historical_pass','historical_suspected_fault']},
        targeted_probe_plans=len(list((RUN/'cases').glob('*/probes/*/result.json'))),
        reference_limit='Agreement with historical labels is not gold-standard accuracy; historical pass does not certify all tools; suspected faults require confirmation.')
    write_json(RUN/'summary.json',summary)
    body=['# 100 题环境盲测：程序基线与 Luna 子 agent', '',
          f"完成 {len(results)}/100；模型 gpt-6-luna，推理 high。先由 3 个原生子 agent 处理；用户要求提速后，余题切换为 12 并发 Codex CLI 独立单题会话。调度分布：{summary['dispatch_counts']}。" if cli_execution else f"完成 {len(results)}/100；模型 gpt-6-luna，推理 high，3 个原生子 agent 并行。",'',
          '固定选取历史执行通过 50 题，以及疑似环境启动/能力故障 10 题、工具故障 20 题、任务数据问题 20 题，共 100 个不同环境。选择在审查前冻结，验证员未获提供历史标签与轨迹。', '',
          '| 历史分组 | 方法 | 通过 | 排除 | 待确认 |', '|---|---|---:|---:|---:|']
    for group,label in [('historical_pass','历史通过 50'),('historical_suspected_fault','历史疑似故障 50')]:
        counts=summary['baseline_groups'][group]
        body.append(f"| {label} | 按场景保守隔离的程序基线 | {counts.get('screen_pass',0)} | {counts.get('exclude',0)} | {counts.get('review',0)} |")
        counts=summary['groups'][group]
        first_counts=summary['first_pass_groups'][group]
        body.append(f"| {label} | Luna 首次判定 | {first_counts.get('pass',0)} | {first_counts.get('exclude',0)} | {first_counts.get('review',0)} |")
        body.append(f"| {label} | Luna 加主 agent 证据复核 | {counts.get('pass',0)} | {counts.get('exclude',0)} | {counts.get('review',0)} |")
    body += ['', '历史标签由先前模型执行和归因产生，不是人工金标准。历史成功题被排除需复核是否为未经过的缺陷路径；历史疑似故障题被通过也不自动等于漏检，需要检查原标签或替代路径。这里统计标签对照，不宣称真实准确率。', '',
             '程序基线遇到共享环境任一明确缺陷会保守隔离；Luna 审查允许排除无关问题，并围绕本题义务验证。两者既有分析方法差异，也有判定粒度差异，不能把提升全部归因于 LLM。', '',
             '每个子 agent 独立上下文处理其队列；题目之间分别保存数据库副本与结果。属于操作盲测（提示词和分发输入隐藏标签），不是文件系统权限强制盲测。', '',
             '12 并发续跑采用每题独立 CLI 会话，补充说明保存在 cli12/supplement_zh.txt（若存在），明确初态目标记录、公开 ID 可发现性和模拟能力口径。首次判定包含试验中反馈后的结果，不能当成从始至终使用单一提示词的严格独立验证集。', '',
             f"首次判定保存在 decision.first_pass.json；主 agent 按题面和证据复核后的分类变化：{summary['review_changed_cases']}。最终结果包含复核，不能视为纯 Luna 首次输出的效果。", '',
             f"复核分类变化统计：{summary['review_transitions']}。排除原因分类：{summary['excluded_categories']}。这不是人工金标准误判率；主 agent 重点复核实际证据、必要条件与冲突项，没有另建独立金标准。", '',
             f"按主要可修复原因归类（详见 issue_groups.json 和 CSV 的 issue_group）：{summary['excluded_issue_groups']}。", '',
             '协议偏差：一名验证员曾查看同批次其他队列的题面摘要，已纠正；未发现其读取历史标签。详见 private/protocol_notes.json。', '',
             f"验证层级：{summary['verification_levels']}；针对性探测计划 {summary['targeted_probe_plans']} 个。程序基线详情见 baseline/summary.json。", '',
             '模型与费用：使用 gpt-6-luna high。原生子 agent 调度工具未提供完整 token/费用计数；CLI 续跑逐题记录实际模型配置和可获得用量。脚本未使用另行配置的 API key，不代表没有模型用量。', '',
             '后续采样建议：预检只对当前环境/数据库版本有效。采样时仍按实际故障证据处理：确认环境故障则丢弃轨迹，不记为模型负奖励；故障归因未明或写入结果未知则隔离并在新副本诊断；模型参数错误、正常业务拒绝和正常空结果不自动丢弃。API 配额/网络等基础设施故障单独记录，不直接把题目永久排除。', '',
             '推荐将正式入口启动与工具 schema 校验作为程序硬检查，再让 Luna 判断缺陷与本题的关系及可行替代路径。pass 表示存在经过验证的可用路径，不表示环境全部接口无 bug；如 C090，带 Z 的时间输入仍会失败，但已验证等价的公开输入可完成本题。', '',
             '如果用于 GRPO，应补足同题所需的有效轨迹数，不能将环境失败填成零分奖励。环境筛选也不证明 rubric/Judge 正确；评分有效性需要单独验证。审计员看到的私有数据库诊断信息不能注入 Actor 的做题上下文。', '',
             '文件：prompt_zh.txt（冻结提示词）、comparison.csv/jsonl（逐题分类）、cases/Cxxx/decision.json（理由与证据）、private/selection.jsonl（对照标签，仅汇总端使用）。', '',
             '改进版提示词：configs/environment_screen/pilot100_v2/prompt_zh.txt。已根据本次复核收紧必要义务与证据要求、放宽无关缺陷和模拟能力限制；完整 v2 尚未在新的独立保留集上验证。', '',
             '本次没有筛查剩余 7000 题，没有修改原题库，也没有部署正式采样/训练流程。']
    (RUN/'REPORT.md').write_text('\n'.join(body)+'\n')
    print(json.dumps(summary,ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','probe','publish-baseline','report'])
    p.add_argument('--case',type=Path);p.add_argument('--plan',type=Path);a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='probe':asyncio.run(probe(a.case,a.plan))
    elif a.mode=='publish-baseline':publish_baseline()
    else:report()


if __name__=='__main__':main()
