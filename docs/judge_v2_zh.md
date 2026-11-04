# Judge V2

V2 只评估真实执行，输出逐项硬要求、质量项及可追溯证据。奖励固定为 `null`，`reward_valid=false`；没有质量加权分数，也不会导出训练样本。旧官方协议仍是默认值，D1 离线接口保持原行为。

## 判定契约

模型每项提交 `satisfied`、`violated` 两份证明，包含 established、evidence_ids 和 reason。两个布尔值不能同时为真；全部必要条件成立才 pass，一个独立必要违例成立就 fail，均未成立为 unknown。硬软标签由冻结 rubric 读取，模型不能输出总标签或奖励。

| 条件 | business_label | task_status |
|---|---|---|
| 至少一个硬项明确失败 | fail | failed |
| 全部硬项通过 | pass | confirmed_pass |
| 无硬项失败，正常取证后仍存在真实歧义、缺证或冲突 | unknown | provisional_pass |

后三值只在 `evaluation_status=valid` 时可作为任务结果。关键评测故障、未完成取证或预算耗尽产生 `evaluation_error`；有证据证明任务不可评价为 `invalid_sample`；影响硬项的规则缺陷为 `rubric_error`。这些情况下 task_status=undetermined，保留已有 business_label 作诊断。质量项的问题不阻断独立的硬项结论。没有硬要求不自动通过。

unknown 必须包含 kind、missing_fact、impact、checked_evidence_ids、next_action、can_resolve_with_tools。still-failed 查询若被 unknown 引用，程序将其归为 evidence_error，不能包装成可放行歧义。schema/引用校验不能证明任意自然语言推理为真，仍需语义复核。

unresolved 逐项映射到 U001 等固定 ID，记录 not_triggered/resolved/material，并保留原 affected_items。未触发和已解决不扣分、不增加要求。

## 来源适配与冻结输入

`JudgeInput` 保存 original、原 rubric、normalized_rubric、episode 绝对路径及来源、规范化结果、轨迹、事件、初末数据库、任务绑定和运行信息的哈希。读取执行前先检查来源；评测前后检查快照和 seal。模型只看到当前证据，不接收历史参考标签和官方结果。

适配器支持现有 `要求/G/P/F` 和旧 `要求/G/P` 格式。原 P、原文、来源引用和软硬标签不变；旧规则的独立失败条件定义为“原 P 中有来源的必要义务的独立反例”，不再将 G 当失败总开关。适配结果独立保存，并明确标记它是逻辑规范化，**不是已完成的语义修复**。来源引用不成立会拒绝输入；有来源但语义附加要求、遗漏义务等问题由 Judge 标为 rubric_issues，修订后使用新的输入版本。

## 取证与边界

- inspect_schema 从实际快照读取表结构、主键、外键；query_snapshot 查询指定 initial/final 库，SQL 中使用本库表名，不加 initial./final. 前缀。
- compare_records 仅接受完整主键或非部分唯一键，不用行号猜配对；无可靠键时改用显式查询。
- read_evidence 按字符 offset 读取大证据，返回 next_offset。完整原始记录保留在证据目录；提示词中的 view_truncated 标记只描述预览，不能将预览缺失当作记录不存在。
- SQLite 只读连接、query_only、authorizer 同时限制写入、ATTACH、PRAGMA 和文件/扩展调用；单条 SQL 默认 3 秒、100 行、24 KB。
- 空查询结果、查询错误和截断分别记录。截断只证明返回的事实，不能证明全集或不存在。程序检查引用合法性，查询范围和证明是否充分仍由语义判断负责。
- 主取证默认最多 6 轮，每轮 12 项；同一 objective 查询失败最多修正 2 次，保留每次结果。硬项 unknown 增加一次新上下文复核，最多一组补查。格式修正全程最多一次。
- 默认 `review_all_decisions=true`，同一复核阶段也检查候选 pass/fail 的引用值与证明理由；原候选与复核结果分别保存。关闭时仍保留关键 unknown 复核。复核不是独立人工裁决，不能保证自然语言推理完全正确。
- 主轮次耗尽或最后一组复核查询已用完后，硬项仍缺事实时保守归为 budget_exhausted/evaluation_error，不能仅凭模型把 can_resolve_with_tools 改为 false 就放行；已查明的真实题意歧义不受此影响。
- API 网络错误不自动重试。必要补查尚未完成时不能给 provisional_pass。模型调用、复核和格式修正共用请求/token 预算。

## 使用

下面命令使用仓库运行环境。示例配置的预算为零；运行真实模型前在新配置中设置请求和 token 上限，并配置 OPENAI_API_KEY。代码不读取密钥到结果文件。

```bash
agent-world-model/.venv/bin/python -m harness_awm.cli judge-prepare \
  --episode /absolute/path/to/sealed/episode \
  --original /absolute/path/to/original.json \
  --rubric /absolute/path/to/rubric.json \
  --output /absolute/path/to/new/judge_input.json

agent-world-model/.venv/bin/python -m harness_awm.cli judge-replay \
  --input /absolute/path/to/new/judge_input.json \
  --config /absolute/path/to/judge_config.json \
  --output /absolute/path/to/new/evaluations

agent-world-model/.venv/bin/python -m harness_awm.cli judge-report \
  --results /absolute/path/to/new/evaluations \
  --output /absolute/path/to/new/report.json
```

judge-replay 不改原 episode。结果目录按输入、配置、提示词和实现哈希分组；每次请求、原始响应、证据、格式错误和复核结果均保存。相同版本复用首个有效结论，包括暂定通过；明确的样本/规则缺陷也冻结。只有评测错误可产生新尝试，旧尝试不覆盖。进程中断后的未完成尝试保留，并消耗原调用预算。

自动执行入口使用 `judge_protocol="rubric_v2"` 和 `judge_inputs_manifest`。清单的 tasks 按 task_instance_id 关联原始来源和 rubric；相对文件路径相对于清单所在目录解析：

```json
{
  "schema_version": "awm-judge-input-manifest-v2",
  "tasks": {
    "actual-task-instance-id": {
      "original": {"path": "original.json", "sha256": "实际文件SHA256"},
      "rubric": {"path": "rubric.json", "sha256": "实际文件SHA256"}
    }
  }
}
```

V2 分支不运行官方 SQL verifier，也不依赖它构建模型请求。旧结果可在冻结 V2 结论后作对照，不送入 Judge。配置仅支持 development；formal 训练准入等待未来奖励协议。episode 中新增 judge_protocol、evaluation_valid、business_label、task_status；V2 的 metrics 根据这些字段统计，不通过 reward==1 推断成功。

## 回归材料

`scripts/judge_v2_regression.py prepare/run/report` 使用原 100 个冻结样本：先运行 `--phase development` 的20题，再运行 `--phase all`。已有初始化失败且没有轨迹的样本按原始失败记录列为 invalid_sample，不重跑 Agent。所有来源哈希、输入及模型结果另存；原数据保持不变。

报告分别列出确认通过、暂定通过、失败、规则/样本/评测问题，记录查询目标恢复情况、用量和旧 unknown 转确定标签的复核清单。source_evidence_checks 是机械完整性检查，不能替代对证明内容的核验。已暴露的20/100题都不是独立测试集，历史模型判分不是人工金标准，不据此声称误放率或独立准确率。
