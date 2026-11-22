# Judge V2

V2 评估真实执行，输出逐项硬要求、软要求及可追溯证据。Judge 本身及独立 replay 的奖励保持 `null` / `reward_valid=false`；自动评测入口由版本化奖励模块结算。旧版 `judge-v2-terminal-discount-v1` 使用 1 / 0.3 / 0；新版 `judge-v2-cst-v1` 按下述 C/S/T 公式计算。评测错误仍为 null。旧官方协议仍是默认值，D1 离线接口保持原行为。训练接入见[训练说明](training_setup_zh.md#judge-v2-奖励与逐轮-grpo)。

## C/S/T 奖励（2026-11-10）

按用户提供的 `AWM-reward-design.md`，并经用户确认 unknown 是打三折：

`R = C × [1 − 0.05 × (1 − S) − 0.15 × (1 − T)]`

- C：全部硬项确认通过为 1；无明确失败、复核后仍有真实 unknown 为 0.3；至少一个硬项明确失败为 0。
- S：软项等权平均；pass=1、fail=0、复核后的真实 unknown=0.3。附件没有另定软项聚合细节，这是本实现的明确约定。无软项时 S=1，避免无偏好要求的任务被扣分。未完成取证、工具故障、预算耗尽和软规则缺陷不冒充软项失败或真实 unknown，完整奖励暂不结算。
- T：独立的第二次 LLM 评估；信息使用、反馈处理、探索与终止三个维度各取 d∈{0,0.5,1}，程序计算 `T=1−sum(d)/3`。使用同一 Judge API 配置，但独立提示词、上下文、输出和缓存。

第二个 Judge 接收硬项结论及封存的逐轮 Actor 输入和输出，不接收私有 rubric、隐藏数据库或第一个 Judge 的取证理由。输入消息去重后仍保留每一轮实际可见消息 ID，并直接展开每轮动作前的最新观察；非零缺陷必须逐字引用动作和同一轮可见的先前依据。不能按长度直接扣分、奖励先制造错误再修复、把一个根本缺陷跨维度重复扣分。第二个 Judge 对成功、失败、提前终止的轨迹都评价已发生的动作；没有适用行为或没有确认缺陷的维度给 0 并说明检查范围，不因最终结果缺失而返回 unknown。评分服务失败或记录不完整仍标记技术错误，不能自动填满分。初评后尝试复核，独立于第一个 Judge 的 `review_all_decisions` 设置；仅复核 JSON 语法持续失败时保留已经校验的初评，并明确记录 `reviewed=false`，不冒充复核成功。保留全部请求、回复和证据。自然语言证据的因果关系仍须靠校准和抽查，结构校验不等同于语义准确率。

确认成功奖励范围为 [0.8,1]；真实 unknown 为 [0.24,0.3]；明确硬失败恒为 0。硬失败无需依赖 S/T 的可用性。`reward_components` 保存 C/S/T、软项分数、三个缺陷等级及固定参数；`trajectory_quality` 保存第二个 Judge 结果。训练入口重算公式、拒绝篡改或混用版本，接受连续奖励，不再只允许 0/0.3/1。

新本机配置 `configs/training/grpo-landlock-local.json` 选择 CST，旧冻结计划仍属于旧奖励版本。新奖励须重新校准和生成准入；Landlock 校准除原四类业务控制外，增加正常轨迹与重复获取已知 schema 的对照。后者仍完成硬目标，用于确认轨迹 Judge 能识别冗余。校准轨迹保持 test_only，不能进入策略更新。

如果一组的 C/S/T 都相同，优势仍然为零，不人为制造差异。成功组只有质量差异时，组内标准差归一化可能放大小差异，因此小 λ 不等于小梯度；须同时观察成功率、缺陷证据与混合组占比。

## 第二个 Judge 的轨迹回放与引用修复（2026-11-11）

`trajectory_quality` 的 `awm-trajectory-quality-v4-local-repair` 版本显式列出每轮可引用的非 assistant 消息 ID，并给出消息实际可见的轮号。Actor 自述使用 `A` 编号，与观察的 `M` 编号区分。校验仍以封存的 Actor 输入为准；未来反馈、被压缩移除的旧观察、assistant 自述均不能作为当时可见的证据。JSON 嵌套转义可确定性还原成原 content 中的连续片段后再接受，持久化的是原文片段；不做模糊匹配，不自动更换引用轮次或消息。

重复执行账本按记录无损去重，保留每轮账本的顺序及全部字段。上下文预算包含后续复核空间；仍然过长时按动作窗口拆分，每个动作保留完整实际输入。所有窗口都需要初评并尝试复核；有缺陷时再统一复核三个维度，不累加窗口分数。单轮完整输入或汇总仍超限时显式报错，不截断证据或默认满分。同值重复字段及完整 JSON 代码围栏可本地规范化，记录 normalization.json；冲突值不得任选其一，来源证据仍使用严格解析。每个初评/复核阶段最多两次局部修复，锁定其他已通过校验的维度；语法修复不要求重新判分，引用错误只复核相关缺陷。有效初评和完成窗口分别缓存，恢复复核不会重做初评。语义或引用错误耗尽恢复额度后仍保留 error/null。

引用纠错聚焦被指控动作的完整输入，不再反复发送无关轮次；初评纠错后仍回到完整上下文复核。仅凭系统规则和账本引用不能对 schema/观察重读扣分，因为它可能是在恢复被压缩移除的内容；需要具体的当时可见证据。

只重评第二个 Judge，可复用历史轨迹与第一 Judge 的有效结果，不启动环境、Actor 或 GPU。先做本地检查，再用 `--execute` 调用配置中的远程 Judge API：

```bash
python scripts/replay_trajectory_judge.py \
  --run runs/grpo_floor_20261110_v1 \
  --output runs/trajectory_judge_repair_20261111

python scripts/replay_trajectory_judge.py \
  --run runs/grpo_floor_20261110_v1 \
  --output runs/trajectory_judge_repair_20261111 \
  --execute --controls 2 --concurrency 3
```

回放默认选取第二 Judge 的全部 error 样本与两条 valid 对照。`report.json` 保存重算奖励、调用用量、来源哈希和评审路径；`recomputed_reward` 是评分结果，不是新训练准入。原始 `episode_result.json`、第一 Judge 结果和历史训练更新不改写。Judge 实现身份已更新，旧缓存不会被误用；恢复正式训练前须用新实现重新校准并生成匹配的奖励准入，不能直接改旧准入哈希。

## 执行反馈与单条异常（2026-11-11）

V2 初态仍要求物理与外键检查均通过。执行后的审计和末态快照保留可读取的外键非法状态，不替模型修数据；工具响应后立即将外键错误反馈给 Actor，允许查询、纠正并继续。数据库物理损坏仍拒绝可信快照。Judge 1 按相关必要义务判断最终非法关联是否构成任务失败；已修复则正常判断，未修复且确认违反要求则给失败奖励 0。Judge 2 仅根据实际可见反馈评估纠错过程。

GRPO / verl 在验证来源和准入后按候选过滤评分故障：确认失败的 0 分参与训练；未评分样本既不填 0，也不参与均值、标准差或损失。有效候选至少 2 条时继续组内训练，并按有效候选数归一化损失；否则仅暂缓该组。verl 的 reward_selection.json 保存原候选索引及排除原因。一次 API 故障不再退出全程，多数 Judge 故障也不再自动隔离任务；多数真实环境故障仍可隔离任务。延后样本不自动混入其他策略版本的组。

这些改动更新 Judge 配置/实现身份。历史奖励和证据不改写；真实训练恢复前重新运行校准并生成准入和计划，不直接替换旧准入哈希。离线传输与环境测试验证恢复行为，不代替真实 Judge 语义校准。

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
- Judge API 的连接错误、超时、临时服务端错误和可恢复限流默认最多发送 3 次（`judge_api.judge_retry_attempts`），采用退避并尊重 Retry-After；等待超过 `judge_retry_max_delay`（默认 60 秒）时保留待恢复状态。鉴权、额度耗尽和取消不盲目重试。限速覆盖每次实际发送，重试不会叠加成多层次数，每次请求都占预算并保存脱敏异常链。Actor 和业务工具调用不随 Judge 重试。必要补查尚未完成时不能给 provisional_pass。

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

使用 OpenAI `gpt-6-luna` 时，可将命令中的 `-m harness_awm.cli` 换成 `-m scripts.judge_rate_limit`。启动器从 `.env` 或环境变量读取 `JUDGE_OPENAI_RPM` 和 `JUDGE_OPENAI_TPM`，同一仓库中通过它启动的进程共用 60 秒限速窗口。按 OpenAI 项目的实际限额留出余量，例如实测为 5,000 RPM / 2,000,000 TPM 时可配置 `4500` / `1800000`。

启动器在发送前预留估算输入与最大输出 token，遇到临时 429 时按 `Retry-After` 或退避等待，最多发送三次。每次发送计入原有总预算。这个窗口覆盖由该启动器发出的 Judge 请求，Actor 和独立运行的补评脚本有各自的调度。

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

V2 分支不运行官方 SQL verifier，也不依赖它构建模型请求。旧结果可在冻结 V2 结论后作对照，不送入 Judge。formal 模式需要 strict 隔离、真实 Actor/Judge 和绑定当前版本的 reward_admissions。episode 中包含 judge_protocol、evaluation_valid、business_label、task_status；V2 的 metrics 根据这些字段统计，暂定通过即使取得 0.3 奖励，也不计入确认成功率。

## 回归材料

`scripts/judge_v2_regression.py prepare/run/report` 使用原 100 个冻结样本：先运行 `--phase development` 的20题，再运行 `--phase all`。已有初始化失败且没有轨迹的样本按原始失败记录列为 invalid_sample，不重跑 Agent。所有来源哈希、输入及模型结果另存；原数据保持不变。

报告分别列出确认通过、暂定通过、失败、规则/样本/评测问题，记录查询目标恢复情况、用量和旧 unknown 转确定标签的复核清单。source_evidence_checks 是机械完整性检查，不能替代对证明内容的核验。已暴露的20/100题都不是独立测试集，历史模型判分不是人工金标准，不据此声称误放率或独立准确率。
