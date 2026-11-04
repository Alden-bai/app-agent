#!/usr/bin/env python3
"""Read-only evidence audit and transparent final report for the completed pilot."""
import collections
import json
import sqlite3
from pathlib import Path
import judge_accuracy100_openai as original
from judge_accuracy100_openai_v2 import strict_sql_evidence
from harness_awm.io import file_hash, now, read_json, verify_seal, write_json


def metrics(rows):
    n=len(rows);correct=sum(r['agreement'] for r in rows)
    return {'comparable':n,'agreement_count':correct,'agreement_rate':correct/n if n else None}


def main():
    root=original.RUN
    assert (root/'final_completion.json').exists()
    summary=read_json(root/'summary.json');manifest=read_json(root/'manifest.json')
    prior=read_json(original.PRIOR/'manifest.json')
    results=[read_json(root/'tasks'/t['id']/'result.json') for t in manifest['tasks']]
    assert len(results)==100
    assert [r['task_id'] for r in results]==[t['task_id'] for t in prior['tasks']]
    for path,expected in manifest['implementation_hashes'].items():assert file_hash(path)==expected,path
    for path,expected in manifest['input_hashes'].items():assert file_hash(root/path)==expected,path
    with sqlite3.connect(root/'budget.sqlite') as db:
        requests=db.execute("SELECT request_key,stage,status,started FROM requests").fetchall()
    judge=collections.Counter(key.split('/')[0] for key,stage,_,_ in requests if stage=='judge')
    assert max(judge.values())==1
    assert not any(status=='in_flight' for _,_,status,_ in requests)
    starts={key.split('/')[0]:started for key,stage,_,started in requests if stage=='judge'}
    audit={'at':now(),'fixed_sample_100':True,'frozen_inputs_and_implementations_intact':True,
           'max_physical_judge_requests_per_task':1,'sealed_episodes':0,'unchanged_official_prompts':0,
           'reproduced_valid_reference_sql_queries':0,'sql_mismatches':[],'initialization_error_types':{},
           'reference_timing':{},'current_usage_unknown_requests':sum(status!='response' and not stage.startswith('prior_') for _,stage,status,_ in requests)}
    init_errors=collections.Counter();unknown=collections.Counter();before=[];after=[]
    for r in results:
        out=root/'tasks'/r['id'];episode=out/'episode'
        if (episode/'evidence_seal.json').exists():verify_seal(episode);audit['sealed_episodes']+=1
        if r.get('error')=='ValueError: incomplete_initialization':
            initial=read_json(out/'initialize/output.json')
            for error in initial['schema_errors']+initial['sample_errors']:
                init_errors[error.split(' - ',1)[-1]]+=1
        refpath=Path(r.get('active_reference',out/'reference_frozen.json'))
        ref=read_json(refpath) if refpath.exists() else {}
        if r['reference_label']=='unknown':unknown[ref.get('error',r.get('error') or 'model_semantics_or_disagreement')]+=1
        if r.get('agreement') is not None:
            timing='before_first_judge_request' if ref['frozen_at']<starts[r['id']] else 'posthoc_blind_repair'
            audit['reference_timing'][r['id']]=timing
            (before if timing=='before_first_judge_request' else after).append(r)
            queryfile=refpath.parent/'queries.json'
            if queryfile.exists():
                latest={q['id']:q for q in read_json(queryfile)}
                for q in latest.values():
                    if not q.get('valid'):continue
                    actual=strict_sql_evidence(episode/'initial.db',episode/'final.db',q)
                    keys=['valid','columns','rows','truncated']
                    if any(actual.get(k)!=q.get(k) for k in keys):audit['sql_mismatches'].append({'task':r['id'],'query':q['id']})
                    else:audit['reproduced_valid_reference_sql_queries']+=1
        if r['judge_valid']:
            e=read_json(r.get('active_official_evaluation',out/'official_evaluation.json'))
            attempt=episode/e['attempt'];source=read_json(attempt/'official_request.json')
            wire=read_json(attempt/'transport/request.json')
            assert wire=={**source,'reasoning_effort':'none'},r['id']
            assert source['model']=='gpt-6-luna'
            audit['unchanged_official_prompts']+=1
    assert not audit['sql_mismatches'],audit['sql_mismatches']
    audit['initialization_error_types']=dict(init_errors)
    audit['strict_pre_judge_reference']=metrics(before);audit['posthoc_blind_reference']=metrics(after)
    audit['unknown_reference_reasons']=dict(unknown)
    write_json(root/'final_audit.json',audit)
    summary.update(strict_pre_judge_reference=metrics(before),posthoc_blind_reference=metrics(after),
        metric_interpretation='Conditional agreement with same-model rubric/evidence reference; not absolute accuracy or accuracy over all 100 tasks.',
        unknown_reference_reasons=dict(unknown),final_audit_sha256=file_hash(root/'final_audit.json'))
    summary['limitations'].extend([
        'Reference evidence-ID aliases were normalized mechanically and reviewed in fresh blind contexts; the original labels and every repair are archived.',
        'Pre-dispatch Judge length failures were dispatched once after increasing the conservative local byte guard; no physical Judge request was resampled.',
        'Five Actor API requests and one Judge request were interrupted during engineering fixes; unknown usage remains reserved. These are study artifacts, not model-service failures.',
        'Both strict pre-Judge and posthoc blind-reference subsets are reported; this exploratory pilot is not a pristine preregistered accuracy estimate.'
    ])
    write_json(root/'summary.json',summary)
    pct=lambda v:f'{v:.2%}'
    lines=['# GPT-6 Luna：固定100题真实轨迹与官方Judge试验','',
        f"固定抽样100题已全部处理；79题取得有效官方判分，49题同时取得可确认的参考标签，其中44题一致、5题分歧。条件一致率为 **{pct(summary['reference_based_accuracy'])}（44/49）**。",
        '这不是100题的绝对正确率。参考标签来自同一模型的独立上下文，并非人工金标准；无法确认的题未纳入一致率。','',
        '| 项目 | 数量/比例 |','|---|---:|',
        '| 固定样本 | 100 |','| 有效官方判分 | 79 |','| 可比较 | 49 |','| 一致 | 44 |',
        '| 参考失败、官方通过（误接受） | 4/29 = 13.79% |','| 参考通过、官方拒绝（误拒绝） | 1/20 = 5.00% |',
        '| 官方判分有效，但参考标签无法确认 | 30 |','| 没有有效官方判分 | 21 |',
        f"| 一致率的条件Wilson 95%区间 | {pct(summary['wilson95_conditional_on_reference_labels'][0])}–{pct(summary['wilson95_conditional_on_reference_labels'][1])} |",'',
        '21题缺少有效判分：17题原始数据库初始化失败，3题缺少或存在冲突的官方verifier，1题Judge调用在工程修复期间中断。没有替换抽样题，也没有因任务失败重跑Actor。',
        '79题有效判分中，30题仍不能确认参考标签：28题有语义/证据不确定或两次核查不一致，1题无法从理由恢复有效证据引用，1题整体失败标签与必需rubric项矛盾。','',
        '## 过程与偏差','',
        'Actor、参考核查、独立复核、官方Judge均调用OpenAI gpt-6-luna，reasoning_effort=none；上下文分开。参考核查看原始任务、公共契约、rubric、真实轨迹、只读初始/最终数据库查询，不看官方判分。官方判分保留官方提示词、温度与真实轨迹/验证输出。',
        '过程中修复了参考SQL初始/最终库歧义、失败查询ID复用、证据引用格式问题；所有旧记录保留，已有官方响应没有重新请求。最初本地长度检查按UTF-8字节作保守估算，11题请求在发送前被拦截，之后调高本地阈值才完成首次发送。',
        f"严格在首次Judge请求之前冻结的可比较标签：{before and sum(r['agreement'] for r in before)}/{len(before)} = {pct(metrics(before)['agreement_rate'])}。判分之后做盲态修复的可比较标签：{sum(r['agreement'] for r in after)}/{len(after)} = {pct(metrics(after)['agreement_rate'])}。二者分开报告，不能把合并结果包装成无偏的正式评测。",
        '工程修复期间中断了5次Actor请求和1次Judge请求，未知用量保守占额；已有实际执行保留，不重抽任务。这一轮属于探索性试跑。','',
        '## 分歧案例','',
        '| 任务 | 参考/官方 | 证据与问题 |','|---|---|---|',
        '| booking_reservations_32:1 | 通过/拒绝 | 官方verify用最新历史预约推断“本周”，得出2025-01-06至01-12；实际运行是2026-11-03，当前周为11-02至11-08。最终答案4次与本周查询一致，官方却按历史周要求2次。根因指向verify时间规则。 |',
        '| booking_platform_3:5 | 失败/通过 | 请求把confirmed预约处理成completed，但初始已是completed；官方只认可最终状态。属于状态前提/迁移要求与最终状态判分的冲突。 |',
        '| e_commerce_marketplace_10:3 | 失败/通过 | 请求创建listing，但匹配记录初始就存在，初始/最终相同；官方把最终存在视为完成。 |',
        '| code_hosting_2:9 | 失败/通过 | 请求新建release，工具返回HTTP 500，记录初始就存在且未变化；官方仍判完成。 |',
        '| it_service_management_2:8 | 失败/通过 | 明确请求新建SLA，初始已有相同记录，Actor未创建且说明无法创建；官方仍判完成。 |','',
        '新上下文分歧复核将后4题归为judge_error、首题归为environment_issue；结合实际verify源码，首题更具体地是时间推导规则问题。这些归因仍是模型辅助核查，不是独立人工金标准。对于预先存在的目标状态，应先明确是否允许幂等完成，尤其是明确要求“新建”或状态迁移的任务。分歧复核未改写上述主统计。','',
        '## Token','',
        f"本轮OpenAI已知用量 **{summary['current_run_known_tokens']:,} token**；含6次未知用量保守预留，占额 {summary['current_run_budget_occupied']:,}。",
        f"再计入此前DeepSeek失败试跑，占额总计 **{summary['budget']['tokens_accounted_or_reserved']:,} / 15,000,000**。这是预算占额，不是实际账单；累计16次请求的实际用量未知。",'',
        '| 本轮阶段 | 已知token |','|---|---:|']
    for stage in ['connectivity','actor','reference','reference_review','judge','dispute']:
        lines.append(f"| {stage} | {summary['by_stage'][stage].get('total_tokens',0):,} |")
    lines += ['', '## 审计与下一步','',
        f"已校验固定样本、输入与实现哈希；{audit['sealed_episodes']}条轨迹封存校验通过；{audit['unchanged_official_prompts']}次有效官方请求与原始请求一致（仅增加reasoning_effort）；{audit['reproduced_valid_reference_sql_queries']}个有效参考SQL查询重新读取封存数据库，结果一致。每题最多一次实际官方Judge请求。",
        '建议先人工确认上述5个分歧、明确“已存在是否算完成”和时间基准，再处理30题参考不确定与20题数据/verifier阻断。随后做单独的新评测；不要根据本次判分反向调整rubric后仍引用本次一致率作为独立验证结果。原有8909份rubric未修改。','',
        '## 全部100题','', '| 任务 | 运行状态 | 参考 | 官方 | 一致 | 证据 |','|---|---|---|---|---|---|']
    for r in results:
        path=root/'tasks'/r['id']/'result.json'
        official='无有效判分' if not r['judge_valid'] else ('通过' if r['official_pass'] else '拒绝')
        lines.append(f"| {r['task_id']} | {r['status']} | {r['reference_label']} | {official} | {r.get('agreement','—')} | [记录]({path}) |")
    (root/'report_zh.md').write_text('\n'.join(lines)+'\n')
    write_json(root/'final_report_manifest.json',{'at':now(),'summary_sha256':file_hash(root/'summary.json'),
        'report_sha256':file_hash(root/'report_zh.md'),'audit_sha256':file_hash(root/'final_audit.json'),
        'report_script_sha256':file_hash(__file__)})
    print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':main()
