#!/usr/bin/env python3
"""Build an offline, source-preserving human rubric review packet. No API calls."""
import collections
import hashlib
import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/openai_gpt6_luna_judge_accuracy100_20261103'
DEST = ROOT / 'runs/judge_development20_20261103'
# Deliberate development coverage, not a random accuracy sample. Historical
# model labels determine strata only; human outcomes are left blank.
SELECTION = [
    ('t002','positive_agreement','创建','创建SaaS合同','新记录、模板绑定及字段完整性'),
    ('t008','positive_agreement','查询','查询表单答案','过滤条件、关联答案和返回集合完整性'),
    ('t036','positive_agreement','修改','修改竞拍额度','对象归属、数值和并发订单限制'),
    ('t058','positive_agreement','条件操作','按条件追加书签标签','先取最近20项再过滤；仅对缺失标签者追加'),
    ('t098','positive_agreement','修改','批量移动书签并标记已读','集合范围、路径和状态更新'),
    ('t009','negative_agreement','查询','生成志愿时数JSON报告','分组、时间范围和实际交付物'),
    ('t024','negative_agreement','创建','创建训练并记录三组动作','多项必需结果和精确参数'),
    ('t031','negative_agreement','条件操作','有舱位时升级行程','目标选择、可用性前提和条件分支'),
    ('t035','negative_agreement','修改','拆分餐厅交易','金额守恒、分类和原记录关联'),
    ('t059','negative_agreement','查询','检索需求页面','所有权和标签过滤、返回集合完整性'),
    ('t000','disagreement','查询','计算会员本周剩余次数','当前周、时区、预约状态和计算口径'),
    ('t028','disagreement','修改','完成预约并记录授课信息','初始状态前提、状态迁移及多项记录'),
    ('t041','disagreement','创建','创建商品listing','新建义务与已存在的目标状态'),
    ('t045','disagreement','创建','新建代码仓库release','新建义务、工具错误和数据库事实'),
    ('t076','disagreement','创建','新建SLA定义','新建义务、时间单位和对象绑定'),
    ('t006','unknown','创建','创建可用时间段和活动hold','日期展开、时间段冲突和证据是否充分'),
    ('t025','unknown','创建','申请志愿岗位','题目未写年份时的日期解释'),
    ('t072','unknown','条件操作','满足条件时缩短候位时间','当前班次、时区和条件是否触发'),
    ('t079','unknown','创建','记录托盘物流事件','事件顺序、时区偏移和时间值等价性'),
    ('t092','unknown','修改','设置服务区域并删除范围外区域','当前用户范围和删除义务'),
]


def read(path): return json.loads(Path(path).read_text())
def dump(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def block(value): return '```json\n'+json.dumps(value,ensure_ascii=False,indent=2)+'\n```'
def link(path, label): return f'[{label}]({path})'


def snapshot_rows(path):
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        tables=[r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        result={}
        for table in tables:
            cursor=db.execute('SELECT * FROM "'+table.replace('"','""')+'"')
            names=[c[0] for c in cursor.description]
            rows=[]
            for row in cursor:
                record={k:({'blob_hex':v.hex()} if isinstance(v,bytes) else v) for k,v in zip(names,row)}
                rows.append(json.dumps(record,ensure_ascii=False,sort_keys=True))
            result[table]=collections.Counter(rows)
        return result


def database_diff(initial,final):
    a,b=snapshot_rows(initial),snapshot_rows(final);result=[]
    for name in sorted(a.keys()|b.keys()):
        left,right=a.get(name,collections.Counter()),b.get(name,collections.Counter())
        result.append({'table':name,'initial_count':sum(left.values()),'final_count':sum(right.values()),
            'initial_table_exists':name in a,'final_table_exists':name in b,
            'removed_rows':[json.loads(x) for x in sorted((left-right).elements())],
            'added_rows':[json.loads(x) for x in sorted((right-left).elements())]})
    return result


def main():
    # Never regenerate over a human's edits.
    if DEST.exists():raise SystemExit('Destination already exists; preserve all human edits.')
    manifest=read(SOURCE/'manifest.json');entries={t['id']:t for t in manifest['tasks']}
    source_hashes={};copies=[];selected=[]
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        digest=sha(source);assert sha(target)==digest
        source_hashes[str(source)]=digest
        copies.append({'source':str(source),'copy':str(target.relative_to(DEST)),'sha256':digest})
    # Validate the selected historical strata before writing any material.
    for tid,stratum,*_ in SELECTION:
        r=read(SOURCE/'tasks'/tid/'result.json')
        actual=('unknown' if r['reference_label']=='unknown' else
                'disagreement' if r.get('agreement') is False else
                'positive_agreement' if r['reference_label']=='pass' else 'negative_agreement')
        assert actual==stratum and r['judge_valid'],(tid,actual)
    DEST.mkdir(parents=True)
    for index,(tid,stratum,operation,title,focus) in enumerate(SELECTION,1):
        entry=entries[tid];out=SOURCE/'tasks'/tid;episode=out/'episode';inputs=Path(entry['input'])
        case=DEST/'cases'/f'{index:02d}_{tid}';facts=case/'evidence';history=case/'history'
        task=read(out/'task_spec.json');rubric=read(inputs/'rubric.json');result=read(out/'result.json')
        for name in ['rubric.json','gen_spec.json','gen_db.json','gen_tasks.json']:
            copy(inputs/name,case/'sources'/name)
        for name in ['initial.db','final.db','trajectory.json','events.jsonl','evidence_seal.json']:
            copy(episode/name,facts/name)
        events=[json.loads(l) for l in (episode/'events.jsonl').read_text().splitlines()]
        raw_events=[e for e in events if e['event_type']=='tool_completed']
        for event in raw_events:
            rel=Path(event['raw_result_ref']['path']);copy(episode/rel,facts/rel)
        runtime=read(episode/'episode_result.json')
        factual_runtime={k:runtime[k] for k in ['termination_reason','termination_origin','execution_integrity','terminal_state_status','turns','tool_calls','error']}
        factual_runtime.update(task_id=entry['task_id'],instruction=task['instruction'],
            timezone=task['timezone'],clock_policy=task['clock_policy'],world_datetime=task['world_datetime'],
            relative_time_resolved=task['relative_time_resolved'],start_utc=events[0]['time'],end_utc=events[-1]['time'])
        dump(facts/'runtime_facts.json',factual_runtime)
        differences=database_diff(facts/'initial.db',facts/'final.db');dump(facts/'database_diff.json',differences)
        diff_lines=['# 初始/最终数据库逐行差异','',
            '离线读取两份封存快照后按整行内容比较；更新表现为旧行移除、新行加入，不能据此推断实际调用了哪条SQL。保留重复行计数，不按时间戳猜测是否新建。未修改数据库。','',
            '| 表 | 初始行数 | 最终行数 | 移除行数 | 新增行数 |','|---|---:|---:|---:|---:|']
        for d in differences:diff_lines.append(f"| {d['table']} | {d['initial_count']} | {d['final_count']} | {len(d['removed_rows'])} | {len(d['added_rows'])} |")
        for d in differences:
            if d['removed_rows'] or d['added_rows']:
                diff_lines += ['',f"## {d['table']}",'',block({'removed_rows':d['removed_rows'],'added_rows':d['added_rows']})]
        (facts/'database_changes.md').write_text('\n'.join(diff_lines)+'\n')
        spec=read(inputs/'gen_spec.json')['api_spec'];schema=read(inputs/'gen_db.json')['db_schema']
        (case/'sources/contract.md').write_text('# 原始公共接口契约\n\n'+block(spec)+'\n')
        (case/'sources/schema.md').write_text('# 原始数据库schema\n\n'+(schema if isinstance(schema,str) else block(schema))+'\n')
        trace=read(episode/'trajectory.json')
        tl=['# 执行事实','', '未包含参考标签或官方判分。助手自述不是状态证明，请结合原始工具结果与数据库。','',block(factual_runtime),'','## 最终答复','',trace.get('final_answer') or '（没有最终答复）','','## 动作序列','']
        for step in trace['trajectory']:
            tl += [f"### 第 {step.get('iteration','?')} 步",'',str(step.get('content','')),'']
            if step.get('tool_calls'):tl += [block(step['tool_calls']),'']
            for event in raw_events:
                if event['turn_id']==step.get('iteration'):
                    rel=event['raw_result_ref']['path'];tl += [link(facts/rel,'完整原始工具返回'),'']
        tl += [link(facts/'trajectory.json','完整轨迹（含工具发现和分页观测）')]
        (facts/'trajectory_readable.md').write_text('\n'.join(tl)+'\n')
        # Keep all prior model decisions out of the review form and fact digest.
        copy(out/'result.json',history/'result.json')
        refpath=Path(result.get('active_reference',out/'reference_frozen.json'))
        copy(refpath,history/'reference_frozen.json')
        if (refpath.parent/'queries.json').exists():copy(refpath.parent/'queries.json',history/'reference_queries.json')
        evpath=Path(result.get('active_official_evaluation',out/'official_evaluation.json'))
        ev=read(evpath);copy(evpath,history/'official_evaluation.json')
        copy(episode/ev['attempt']/'official_request.json',history/'official_request.json')
        copy(inputs/'gen_verifier.json',history/'gen_verifier.json')
        if (out/'final_dispute/outcome.json').exists():copy(out/'final_dispute/outcome.json',history/'dispute.json')
        (history/'README.md').write_text('# 历史模型结果（建议人工结论冻结后再看）\n\n'+
            f"历史分层：{stratum}；参考：{result['reference_label']}；官方通过：{result['official_pass']}。这些不是人工金标准。\n\n"+
            '\n\n'.join(link(history/name,label) for name,label in [('reference_frozen.json','历史参考标签及理由'),('official_evaluation.json','官方判分及理由'),('gen_verifier.json','官方verify源码'),('official_request.json','官方实际判分输入')])+'\n')
        lines=[f'# {index:02d} · {title}', '',f"任务ID：`{entry['task_id']}`；原运行ID：`{tid}`；操作类型：{operation}。",'',
            '本题是开发样本，不用于最终独立测试。下方原始rubric未经修改；人工结论尚未建立。','',
            '## 原始任务（完整原文）','',task['instruction'],'','## 原始来源和执行事实','',
            '\n\n'.join([link(case/'sources/rubric.json','原始rubric JSON（包括coverage和unresolved）'),
                link(case/'sources/contract.md','完整公共接口契约'),link(case/'sources/schema.md','原始数据库schema'),
                link(facts/'trajectory_readable.md','真实轨迹、最终答复及运行时钟'),
                link(facts/'database_changes.md','初始/最终数据库逐行差异'),
                link(facts/'initial.db','初始数据库'),link(facts/'final.db','最终数据库')]),'',
            '## 人工复核记录（请填写）','',
            '- 复核人：\n- 复核日期：\n- 状态：待复核\n- 填写前是否已看历史模型判分：\n- 题目本身是否可判定（是／需补充约定／数据有问题）：\n- 原始任务的必需结果，用自己的话列出：\n- 时间、时区、对象范围等需要确定的约定及原始依据：','',
            '逐项先判断规则是否忠实于原任务，再判断这条真实轨迹是否满足规则。可判定的必要违例与其他未知项分开记录；不要只因模型同意就确认。','']
        for item in rubric['rubric']:
            lines += [f"## {item['id']} · 原始rubric",'',block(item),'',
                f"### {item['id']} · 人工填写",'',
                '- 规则处置（保留／修改／删除／暂不能确定）：\n- 是否增加了原题没有的要求，或遗漏必要要求：\n- 原始依据（原文引用、source_id或文件位置）：\n- 建议的新规则全文（无修改可写“不改”）：\n- 本次轨迹结论（满足／违反／证据不足／样本无效）：\n- 支持结论的证据（具体工具步骤、表/行/字段、SQL或最终答复）：\n- 仍缺少的证据：','']
        lines += ['## 原始覆盖关系','',block(rubric.get('coverage',[])),'','## 原始未决问题','',block(rubric.get('unresolved',[])),'',
            '## 整题结论（请填写）','',
            '- 是否需新增rubric项及内容：\n- 修订后的rubric是否已覆盖全部任务义务：\n- 对本条轨迹的人工标签（通过／失败／无法确认／样本无效）：\n- 标签理由和决定性证据：\n- 尚待解决的问题：\n- 可否冻结为可信样板（是／否）：\n- 冻结版本与日期：','',
            '## 历史模型结果（可在填写后再打开）','',link(history/'README.md','历史判分附录')]
        (case/'review.md').write_text('\n'.join(lines)+'\n')
        selected.append({'number':index,'id':tid,'task_id':entry['task_id'],'scenario':entry['scenario'],
            'stratum':stratum,'operation':operation,'title':title,'selection_focus':focus,
            'review_file':str(case/'review.md'),'human_review_status':'pending',
            'human_label':None,'trusted_reference':False,'exclude_from_final_independent_test':True})
    counts=collections.Counter(r['stratum'] for r in selected);ops=collections.Counter(r['operation'] for r in selected)
    assert counts=={'positive_agreement':5,'negative_agreement':5,'disagreement':5,'unknown':5}
    exclusion=ROOT/'configs/evaluation/development_exclusions.json'
    old=read(exclusion) if exclusion.exists() else {'schema_version':1,'task_ids':[],'groups':[]}
    old['task_ids']=sorted(set(old['task_ids'])|{r['task_id'] for r in selected})
    old['groups'].append({'name':DEST.name,'manifest':str(DEST/'selection_manifest.json'),'task_ids':[r['task_id'] for r in selected]})
    old['usage']='Future independent-test samplers must exclude task_ids. This registry does not rewrite frozen historical sampling code or results.'
    dump(exclusion,old)
    selection={'schema_version':1,'created_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'purpose':'human_rubric_development_not_independent_evaluation','source_run':str(SOURCE),
        'selection_method':'purposeful coverage based on historical outcomes and operation types; not representative sampling',
        'strata':dict(counts),'operation_counts':dict(ops),'tasks':selected,
        'api_calls':0,'agent_reruns':0,'rubric_mutations':0,'human_labels_assigned':0,
        'exclusion_registry':str(exclusion),'prior_exposed_task_ids':[t['task_id'] for t in manifest['tasks']]}
    dump(DEST/'selection_manifest.json',selection)
    dump(DEST/'provenance.json',{'sources':source_hashes,'copies':copies,'builder_sha256':sha(__file__)})
    readme=['# 20题rubric人工复核开发样本','',
        '**从下面的20份复核表开始填写。** 每题原始rubric和人工填写栏都在同一份 `review.md`；无需修改其他文件。原始英文任务保留完整内容，中文名称仅便于导航。','',
        '已按历史模型结果选取：通过且一致5题、失败且一致5题、unknown 5题、判分分歧5题。历史分层用于覆盖，不代表人工真值。操作类型：'+ '、'.join(f'{k}{v}题' for k,v in ops.items())+'。','',
        '这些题只用于开发与人工样板建设，不用于最终独立测试。当前20题全部待复核，尚无人工通过/失败标签。使用已有轨迹、数据库和规则；本次没有API调用、Agent重跑或rubric改写。','',
        '## 填写方式','',
        '1. 打开复核表，先读原始任务、公共契约和原始rubric，检查每项规则是否有依据。\n2. 查看真实轨迹和数据库差异；需要时以只读方式查询数据库，区分初始和最终状态。\n3. 在每条rubric下填写保留/修改/删除意见及证据，再填写整题人工标签。规则正确性和本条执行是否成功分开判断。\n4. 缺少事实或任务确有歧义时可以保留“无法确认”；列明缺少什么。不要为了凑满20份金标准而强行确定。\n5. 结论形成后再看历史模型附录；若已提前看过，直接在表中记录。\n6. 只有规则和证据均经你确认、并填写冻结信息的题，才可成为可信样板。','',
        '## 复核入口','', '| 序号 | 任务（中文导航名称） | 操作类型 | 任务ID |','|---:|---|---|---|']
    for r in selected:readme.append(f"| {r['number']:02d} | {link(r['review_file'],r['title'])} | {r['operation']} | {r['task_id']} |")
    readme += ['', '## 数据与判分隔离','',
        '`sources/` 保存完整原始rubric、接口契约、schema和任务来源。`evidence/` 保存复制的初始/最终DB、完整轨迹及真实工具返回，并提供逐行差异和易读轨迹。`history/` 单独保存模型参考、官方判分和verify源码，不写入人工答案栏。',
        '数据库差异只说明两份快照间的状态变化，不独立证明某条操作成功；工具返回和最终答复同样需要与真实状态互相核对。时间判断必须用运行时记录的日期与时区，不能使用人工复核当天日期。','',
        '## 排除与复用','',link(exclusion,'独立测试排除清单'),'、'+link(DEST/'selection_manifest.json','样本清单及选择依据'),'、'+link(DEST/'provenance.json','原始文件和副本哈希记录'),' ',
        '排除清单已登记20个任务ID；以后编写或运行独立抽样器时必须读取它。未修改原100题的冻结抽样器和历史结果。原100题中其余80题也已被本轮评测使用，不能自动视为全新的独立测试集；正式评测优先从此前未用于调试或查看判分的任务中另抽样。','',
        '## 只读查询示例','',
        '请在对应题目的 evidence 目录操作，用真实表名和条件替换示例；两份库路径分别指向初始和最终快照。','',
        '```python\nimport sqlite3\nfrom pathlib import Path\nwith sqlite3.connect(Path("final.db").resolve().as_uri() + "?mode=ro", uri=True) as db:\n    print(db.execute("SELECT name FROM sqlite_master WHERE type=\'table\'").fetchall())\n```']
    (DEST/'README.md').write_text('\n'.join(readme)+'\n')
    # Check every preserved source and every copy after rendering, including DBs.
    for record in copies:
        assert sha(record['source'])==record['sha256']
        assert sha(DEST/record['copy'])==record['sha256']
    assert len(selected)==len({r['task_id'] for r in selected})==20
    assert all(r['task_id'] in old['task_ids'] for r in selected)
    dump(DEST/'validation.json',{'selected':20,'copied_files_verified':len(copies),
        'all_original_sources_unchanged':True,'all_human_labels_blank':all(r['human_label'] is None for r in selected),
        'all_selected_ids_registered_for_exclusion':True,'api_calls':0,'agent_reruns':0})
    print(json.dumps({'packet':str(DEST/'README.md'),'strata':dict(counts),'operations':dict(ops),'copies_verified':len(copies)},ensure_ascii=False))


if __name__=='__main__':main()
