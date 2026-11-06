"""Offline source/evidence audit; never rewrite raw rubric or episode outcomes."""
from collections import Counter
from awm_new_pilot import *

def audit(run=DEFAULT_RUN):
 run=Path(run).resolve();m=load_manifest(run,False);data=Path(m['data']);worlds=[];reviews=[]
 excluded={'laboratory_reagent_qc_release_1:3':'rubric_error：P/F 将 −20°C 储存类别提升为试剂及位置温度上下界都精确等于 −20；原始 schema/公开契约定义范围兼容，实际试剂 [-25,-15]、可用冷冻柜 [-22,-18]。已保留原始全2分审查与 Judge 失败结果，但该规则不能进入合格题库。'}
 for path in sorted(run.glob('worlds/*/result.json')):
  w=read_json(path);name=path.parent.name;folder=data/'worlds'/name
  for t in w['tasks']:
   r=read_json(folder/'rubrics'/f"{t['index']:02d}"/'result.json');t['rubric_review']=r;t['rubric_status']=r['status']
   tid=f"{name}:{t['index']}";review={'task_id':tid,'review_status':r['status'],'scores':r.get('scores',{}),'review_ref':str(folder/'rubrics'/f"{t['index']:02d}"/'result.json')}
   review['zero_dimensions']=[dict(dimension=d['name'],reason=d['reason']) for d in r.get('feedback',{}).get('review',{}).get('dimensions',[]) if d['score']==0]
   if tid in excluded:
    t['rubric_status']='audit_rejected';t['raw_evaluation']=t.get('evaluation');t['evaluation']=dict(evaluation_status='rubric_error',business_label=None,task_status='evaluation_error',audit_reason=excluded[tid]);review['audit_error']=excluded[tid]
   reviews.append(review)
  control_path=path.parent/('controls/result.json' if name.startswith('archaeology') else 'controls.task03/result.json')
  controls=read_json(control_path)
  w['controls_passed']=controls['passed'];w['environment_passed']=controls['passed'];w['audit_passed']=True
  probe=read_json(path.parent/'runtime_probe/result.json')
  runtime=all(probe.get(k) is True for k in ['changed','initial_unchanged','reset_matches_initial']) and probe.get('write_verified',True)
  w['runtime_passed']=runtime
  freeze(path.parent/'audit.json',dict(passed=True,reviewed_indexes=list(range(10)),runtime_passed=runtime,controls_ref=str(control_path),scope='复核已完成不等于所有题合格；原始拒绝与新增规则错误仍阻止扩量。',findings=[r for r in reviews if r['task_id'].startswith(name)]))
  freeze(path.parent/'result.audited.json',w);worlds.append(w)
 decision=gate(worlds);freeze(run/'pilot_gate.audited.json',decision)
 evaluations=[t['evaluation'] for w in worlds for t in w['tasks'] if t.get('evaluation')]
 valid=[e for e in evaluations if e['evaluation_status']=='valid'];labels=Counter(e['business_label'] for e in valid)
 accepted=[t for w in worlds for t in w['tasks'] if t['rubric_status']=='passed']
 summary=dict(generated_scenarios=10,built_environments=len(worlds),runtime_passed=sum(w['runtime_passed'] for w in worlds),environment_controls_passed=sum(w['controls_passed'] for w in worlds),generated_tasks=sum(len(w['tasks']) for w in worlds),independent_rubric_review_passed=sum(r['review_status']=='passed' for r in reviews),audited_rubric_accepted=len(accepted),rubric_all_twos=sum(set(t['rubric_review']['scores'].values())=={2} for t in accepted),rubric_with_ones=sum(1 in t['rubric_review']['scores'].values() for t in accepted),agent_episodes=len(evaluations),valid_evaluations=len(valid),business_labels=dict(labels),evaluation_statuses=dict(Counter(e['evaluation_status'] for e in evaluations)),gate=decision,budget=Budget(run/'budget.sqlite',m['max_requests'],m['max_total_tokens']).summary(),formal_admission=False,reward_exported=False,independent_accuracy_measured=False)
 freeze(run/'summary.audited.json',summary)
 lab=run/'worlds/laboratory_reagent_qc_release_1';arch=run/'worlds/archaeology_context_register_1'
 findings=[
 '环境运行层验收 2/2：数据库完整性、外键、工具发现、真实读写、冻结初态和重置隔离均检查通过。场景级对照验收 '+str(summary['environment_controls_passed'])+'/2；实验室完整移动控制的数据库断言通过，但 Judge 因目的地范围为 [-22,-18] 而判 fail，确认规则过严造成误拒。部分完成、未执行均判 fail。',
 '20 题完成初次 rubric 和最多一次语义修订；两份末轮审查截断仅补完审查，未再次生成规则。独立审查通过 9/20，来源与资产复核后可保留 8/20。其余 11 题独立审查拒绝，另 1 题审查漏检在复核中排除。拒绝原因逐题列于 reviews。',
 '通过规则中全 2 分 6 题、含 1 分 2 题。考古 02 的外键描述引用了 contexts 而非 finds，但完整原始 finds schema 和公开筛选契约支持对象关联，未添加硬要求；原规则与 1 分记录保留。考古 08 的旧事件/缺目标边界已查：F-501 在冻结初态确实存在且已装袋，初末态转移记录均为 0，独立违反可判失败；不能用不存在的旧转移冒充本次操作。',
 '有效业务判定 8 题，6 题确认通过、2 题失败、0 题暂定通过/unknown。全部 unknown 与评测异常已核查：题库中没有有效 unknown；历史未执行对照的 script_exhausted 为控制器协议错误，已保留并以正确双 finish 握手建立新对照。不得将未获得合格 rubric 的题算作 unknown。',
 '考古 05 未给已编目且未装袋的 find 301 分配 BAG-301，条件式不关闭虽正确但不免除分袋义务；考古 08 调用查找失败后没有记录两笔转移，初末态均无目标转移。实验室 03 的 Actor 本身还存在无业务调用和协议格式错误，但因 rubric 缺陷，该题不进入最终业务统计。',
 '工程异常保留了版本与费用：JSON 模式要求 JSON 指令、high 输出截断、生成服务器多行 create_engine 与上游包装器不兼容、对照 SQL 行表示差异、零调用结束握手，以及审查补完前输出配额超出 128k 预检。预检失败未发 API 请求；一次 HTTP400 无用量回报保守预留 66,384 tokens。',
 '全部 Actor 题仅从冻结初态运行一次，未因失败删除题目或重采样。新建对照与 Agent 任务结果分开。已通过 144 项回归与 10 项子测试；本次没有测量人工金标准准确率，也不能用 0 unknown 推断 Judge 高准确率。',
 '只生成了首批 20 题；其余 8 个场景为已去重新候选，尚无环境或 80 道题。未达标的修复额度已用完，按预定门槛停止扩量；不是因为预算耗尽或 Agent 成功率低。']
 examples=[dict(kind='通过',description='考古 00 更新 C-101 描述和解释、新建陶片并关联 BAG-101；真实 Agent 获 confirmed_pass，完整控制也通过。',evidence_ref=str(arch/'tasks/00/result.json')),
 dict(kind='失败',description='考古 05 保持有 blocker 的 context 开放，但遗漏分配 BAG-301；Judge 利用冻结初态与末态证明独立违例。',evidence_ref=str(arch/'tasks/05/result.json')),
 dict(kind='unknown',description='本批有效业务判定没有 unknown，不能伪造未知案例。最接近的历史诊断是未执行控制的协议错误：原始 business_label 为 unknown 但 evaluation_status 为 evaluation_error，因此不计业务 unknown。',evidence_ref=str(arch/'controls/unexecuted/result.json')),
 dict(kind='规则错误与对照',description='实验室 −20°C 题的独立审查获全 2 分，但规则把储存类别收紧为温度上下界精确相等。真实对照保存移动结果、Judge 判定及预期差异。',evidence_ref=str(lab/'controls.task03/result.json'))]
 freeze(run/'final_audit.json',dict(conclusion='首批验收未通过，停止后续 80 题。瓶颈为 rubric 语义及环境/规则匹配，不能通过放宽标准或追加抽样补齐。',excluded_tasks=excluded,findings=findings,examples=examples,reviews=reviews,summary=summary))
 print(canonical(summary))
if __name__=='__main__':audit()
