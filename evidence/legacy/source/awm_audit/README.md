# AWM 任务与验证器抽检

本程序检查任务是否能通过官方工具完成，并用独立规则标注的正反例检验官方纯代码验证器及 SQL＋LLM Judge。它不评测解题模型，不用大模型生成标准答案，也不修复官方资产。

## 运行

在 `/home/bjx/app-agent` 执行：

```bash
# 安装到独立环境；不改变 agent-world-model 的项目文件
uv venv --python python3 .audit-venv
uv pip install --python .audit-venv/bin/python -r awm_audit/requirements.lock.txt

.audit-venv/bin/python -m awm_audit preflight
.audit-venv/bin/python -m awm_audit prepare
.audit-venv/bin/python -m awm_audit run --limit 10
.audit-venv/bin/python -m awm_audit run
.audit-venv/bin/python -m awm_audit report
```

已有虚拟环境时跳过安装。`preflight --offline` 不请求模型。`run --mode code` 只测纯代码，`run --mode sql --extract-only` 提取 SQL 证据并把最终 Judge 标为未测。主评测使用 `run` 默认的 `both`。

默认输出：`/home/bjx/app-agent/audit_runs/20261021_seed20261021`。`--run-dir` 可指定新目录。更换目录会生成独立的随机样本运行记录；不要通过反复换种子挑选结果。

```bash
# 只重跑指定的冻结任务；完全匹配的历史结果会直接复用
.audit-venv/bin/python -m awm_audit run --task inventory_management_6:1
# 用本地完整发布文件准备新运行，必须提供官方40位commit
.audit-venv/bin/python -m awm_audit prepare --run-dir audit_runs/offline \
  --source /absolute/path/to/official/files --revision <40-character-commit>
```

本机如遇 Hugging Face TLS 连接重置，可对下载命令设置 `AWM_AUDIT_TLS_FQDN=1`。该兼容传输使用等价的完整域名 SNI，仍验证 CA 链和规范主机名，不关闭 TLS 验证。该选项不改变 LLM 的连接方式。

## 配置

读取工作区 `.env` 中的 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`，也接受 AWM 原生的 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`AWM_SYN_OVERRIDE_MODEL`。优先级为命令行 `--base-url/--model`、进程环境、`.env`；同一层原生字段优先于 DeepSeek 别名。密钥只从环境或文件读取，不提供命令行密钥参数。

判定提示词、温度和输出解析沿用当前检出的官方 `awm.core.verify`。只包装 HTTP 传输来记录请求/响应、usage 和首次判定；超时120秒，仅连接/超时/429/5xx最多重试两次。格式错误、截断、语义不满意都不重试。`llm_requests.json` 不含认证头。

## 方法与文件

- `manifest.json` 冻结官方commit、完整1000环境抽样框哈希、种子、10环境100题清单和源文件/提取资产哈希。对环境名称排序后使用 `random.Random(20261021).sample`，不因缺陷替换样本。
- `initials/` 根据官方原始DDL与INSERT构造一次数据库，保存插入错误、FK检查及完整性检查。初始化失败的环境不会拿部分数据库继续判分。初态内容哈希固定，案例只使用SQLite backup副本。
- `rules/<scenario>.py` 是逐题独立规则与确定性调用配方。编写依据是原始任务、数据库和工具实现；构建阶段不读取官方验证器。冻结后的规则及独立查询写入每题的 `rules/<hash>.json`。
- 每个案例启动独立的本地MCP服务，先真实发现工具，再按发现的参数结构调用。HTTP body的展开是由实际MCP schema决定的。保存工具schema、原始返回、服务日志、完整轨迹、最终回答和数据库快照。
- 先用独立规则核对案例标签，再运行两种官方判定。参考脚本不满足规则、负例实际仍正确、无独立规则、工具失败，都不能作为已知正确/错误案例进入准确率分母。
- 重复或错配的验证器不随意挑一个；该模式记为资产错误，另一模式仍可测试。验证代码在限时子进程里读数据库副本，检查它是否修改副本。
- `attempts/<fingerprint>/` 保留不同实现/规则/资产/初态/依赖/模型配置版本的结果；中断目录保留。运行器开发期间的pilot尝试也保留，报告通过 `task_states.json` 明确引用当前版本，不能把多次判定挑最好的一次合并。

程序预置了全部100题的审计入口。部分任务的指标口径、主体、截止时间或前置条件无法唯一确定，规则会明确记为待确认或已证实的静态矛盾；不以缺失规则冒充“已经验证”。环境初始化错误会覆盖运行状态，但保留能够编写的独立规则。

SQL原始取证输出与最终Judge准确率分别保存。`sql_evidence_review=needs_review` 表示其内容尚未独立核对，不能把它描述为SQL证据已证明正确。官方Judge只接收官方取证及真实调用轨迹，不接收独立标签、独立规则或另一模式的判定。

可在运行后额外核对 SQL 证据并刷新报告：

```bash
.audit-venv/bin/python -m awm_audit.rules.review_evidence --run-dir audit_runs/20261021_seed20261021
.audit-venv/bin/python -m awm_audit report
```

该复核器将结构化行事实与初态/终态独立查询比较，保存每次表绑定和字段证据；启发式绑定发现的矛盾先记为待核对，不冒充确定错误。车辆筛选/排序和库存活动日期另有显式任务规则。`partially_checked`只表示列出的事实已核对，其余聚合、布尔摘要、遗漏及语义解释仍未证明。

## 报告解释

`report.md` 是中文报告；`summary.json` 包含全局及各环境统计，`tasks.jsonl`保留固定100题，`cases.jsonl/csv`提供逐案例结果。每种模式分别报告正例通过率、负例误放率和案例准确率；只有至少1个正例及3个有效不同负例的任务才能进入完整任务覆盖分母。

错误和未测试不当作正常拒绝，分母为0显示未测得。查询答案核对必要字段及完整对象集合；未要求排序时支持无序等价，浮点允许微小计算误差。必须通过API查询的任务也核对真实调用证据。初态已满足时，无操作案例可能是正确的，不会被强行标为负例。

数据按环境成组抽样，100题不是独立同分布样本。本次构造案例的准确率也不等同于整个10000题的总体准确率。已通过工具完成只是存在一个正确的参考执行，不代表所有合法执行路径都通过，或验证器不存在其他未覆盖漏洞。

## 验证

```bash
.audit-venv/bin/python -m pytest awm_audit/tests -q
```

测试包含配置优先级、固定抽样、资产哈希与重复、SQLite/WAL快照、独立规则正反例、查询答案、必要API证据、验证器异常/超时和固定分母统计。额外耳机回归仅为数据库诊断，不是MCP或SQL＋LLM成功，也不混入随机100题统计。

## 充值或网络恢复后续跑

```bash
.audit-venv/bin/python -m awm_audit run --resume-judge
```

该操作重新预检接口，只处理 `llm_call_failure`、`llm_unavailable`、`llm_pending`、`not_tested`，使用已保存的真实轨迹与原始 SQL 取证。首次有效判定以及输出格式错误均不重采样；旧失败不覆盖，追加到每案例 `judge_recovery/`，通过 `previous_evaluation` 追溯。模型和接口配置必须与原运行一致；401/402/403 后停止本批后续请求。

修正独立规则后，可对明确指定的任务复核原轨迹，再补齐首次 Judge：

```bash
.audit-venv/bin/python -m awm_audit run --recheck --task it_service_management_itsm_1:3
.audit-venv/bin/python -m awm_audit run --resume-judge --task it_service_management_itsm_1:3
```

规则修订记录保存在 `oracle_recheck/`，已有有效官方判定保留；新确认的正确案例才补运行缺失的官方验证。仅对已人工审核修订的任务使用该命令，涉及“今天”等条件时需保留原案例时点，不能用重新计算的日期重标旧轨迹。恢复、复核、运行命令均须串行执行，避免同时修改 `task_states.json`。

配方修订时可加 `--reuse-recorded-cases`（兼容旧参数名 `--reuse-no-action`）。只有冻结配方、初态内容、环境工具源码、验证器资产及模型配置一致，且新独立规则重新确认同一标签时才复用原轨迹和首次判定；包括首次非法输出，不通过重采样提高结果。SQLite backup 的文件头可能变化，因此额外比较完整 schema/数据的规范化内容，保留原始文件哈希用于完整性检查。

本次首轮规则有版本化修订，依据用户要求和实际工具行为；修订前后的规则、执行和官方判定均保留。`stages/initial100_summary.json`保留首轮汇总。`postdiagnostics/metadata_persistence.json`记录真实MCP调用后未保存JSON元数据的问题，这类诊断不能代替通过工具的正确案例。
