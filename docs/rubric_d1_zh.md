# D1 rubric：减少无依据的 unknown

后续开发使用 `scripts/rubric_d1.py` 和 `configs/rubric/d1/`。本版依据20题复核和已验收题库的 unknown 抽查修正模板、生成/修订提示词、独立审查提示词及运行结果校验。历史 Luna 批次的程序、提示词、题库与官方 Judge 结果仍按原 manifest 冻结；旧批次入口不会自动切换协议。

## 改了什么

| 原问题 | D1处理 |
|---|---|
| G要求全部证据齐备，一个未知挡住已知失败 | P写全部成功条件；新增F独立失败条件。决策模板不再用G作为失败总开关 |
| 契约已写月数，仍要求确认单位 | 无冲突的公开字段语义可以直接解释任务；仅“接口支持某单位”不够 |
| 假设会重名、并列、混币种 | unresolved必须写trigger和resolution，运行时确认是否实际影响结论 |
| NULL、空记录、无交付被当缺证 | 已知空值与没有证据分开；独立必要违例直接失败 |
| 强制某API、最终读取、固定JSON键或OR顺序 | 接受题目允许的等价路径与表示；精确文本及真正交付要求仍严格检查 |
| 历史状态代替本次动作、只数新增漏末态总量 | 同时约束动作身份、对象绑定、必要时点、完整末态集合和总量 |
| 条件假/条件未知混淆 | 写清真/假分支及保留义务，不用整条not_applicable跳过其他要求 |
| Judge看不到软硬标签、自己决定总分 | 新请求显式带原题、公开来源及完整rubric；代码按原rubric标签汇总 |

日期上的“提前30天”默认自然日，是本版显式解释政策；题面或契约有其他定义时优先遵循。缺失年份、时区、真实发生时间不能靠这项政策猜出来。

## 固定判定

模型每条分别提交 `satisfied`、`violated` 两份证据声明。全部必要义务已证明才允许 satisfied；任一独立必要违例已证明即可 violated。两者都未证明才是 unknown，两者同时声称证明则拒绝输出。其余属性未知或歧义不能盖住独立反证。

整题由程序计算：required有fail则fail；全部required为pass则pass；否则unknown。quality的失败和未知不阻断通过，不抵消失败。没有任何required不自动通过。条件未触发时，检查正确分支及其他义务后正常给pass/fail/unknown。

样本有效性单列。样本invalid、未确认的样本前提、rubric缺陷或结构不合法不产生有效训练奖励；业务标签保留便于诊断。只有可结算的pass/fail映射1/0。此处reward_valid只说明D1结果可结算，正式导出仍须通过执行完整性、审批/数据版本等既有流程；该离线工具不会写训练集。

校验覆盖：每个条目恰好一次，禁止模型更改软硬标签/提交总标签，严格布尔类型、有效证据ID、不能以失败查询证明结论、unknown须说明具体缺口。截断证据可以证明已返回事实，但不能证明全集或不存在；后者的语义边界由独立审查及运行Judge核实，代码不能验证任意自然语言推理。

## 使用

所有命令离线运行，不调用模型。输出使用新文件路径，已有文件会报错。输入JSON重复键也会报错。

```bash
# 从包含task_id/task/sources的原始来源输入准备生成请求（含动态JSON schema）
agent-world-model/.venv/bin/python scripts/rubric_d1.py prepare \
  --input runs/rubric_d1_development20_20261103/01_t002/original.json \
  --output /tmp/d1_generate_request.json

# 将生成器返回的业务JSON编译为五字段rubric，并校验原始来源和双向映射
agent-world-model/.venv/bin/python scripts/rubric_d1.py compile \
  --input runs/rubric_d1_development20_20261103/01_t002/original.json \
  --selection runs/rubric_d1_development20_20261103/01_t002/selection.example.json \
  --output /tmp/d1_rubric.json

# 生成独立质量审查请求；不会自动发送
agent-world-model/.venv/bin/python scripts/rubric_d1.py review-request \
  --input runs/rubric_d1_development20_20261103/01_t002/original.json \
  --selection runs/rubric_d1_development20_20261103/01_t002/selection.example.json \
  --output /tmp/d1_review_request.json
```

生成修订请求时在 `prepare` 同时提供 `--previous` 与 `--feedback`；反馈仍须按来源核实。`review-validate --input ... --selection ... --response ... --output ...` 沿用有效五维0/1/2且无零分的质量审查汇总，拒绝/非法结果会返回非零退出码；结构通过不代表语义已独立审查。

运行Judge使用单独bundle，根字段为 `original`（上述完整来源输入）、`rubric`（D1编译结果）、`trajectory`、`final_answer`、`evidence`。evidence为ID到记录的映射，每条必须有显式布尔 `valid` 和 `content`，可带 `error`、`truncated` 等取证信息。这些标记由可信取证程序提供，不从Agent答复或Judge输出反向生成。

```bash
agent-world-model/.venv/bin/python scripts/rubric_d1.py judge-request \
  --bundle /tmp/d1_bundle.json --output /tmp/d1_judge_request.json
agent-world-model/.venv/bin/python scripts/rubric_d1.py judge-validate \
  --bundle /tmp/d1_bundle.json --response /tmp/d1_judge_response.json \
  --output /tmp/d1_result.json
```

生成请求中的 `messages` 供模型调用层发送，顶层protocol/prompt_sha256用于审计；它们不是API参数。响应是业务JSON，不是API响应信封。运行Judge的新协议独立于旧官方二分类适配器，**尚未接入旧 `evaluate_episode` 自动执行/导出链路**。若继续使用旧评测入口，它仍运行原官方协议。历史rubric需依据来源通过新生成/修订流程补充F，不能批量替换三个decision字符串就视为完成修复。

## 本次交付与验证范围

- `runs/rubric_d1_development20_20261103/manifest.json`：20题来源哈希核对后的新生成请求，未混入轨迹、历史评分或AI建议标签。
- `01_t002/selection.example.json` 与 `rubric.example.json`：真实合同题的来源修订示例，消除了已被公开月数定义解释的1条unresolved，同时补独立失败分支。属于待独立审查的开发草稿，不是已验收题库新版本。
- `tests/test_rubric_d1.py`：三值真值表、硬软汇总、同条独立反证、坏样本/错误证据、来源完整性和CLI往返等回归；同时验证历史编译器仍保留旧输出。

20题准备可以用 `scripts/prepare_rubric_d1_development.py --output 新目录` 重现。测试是离线程序回归，不等于LLM语义准确率。没有重跑Agent、付费模型或全量题库，因此暂不能报告实际unknown下降率；材料包中4个无效样本和缺年份/时钟冲突等问题仍需要各自处理。
