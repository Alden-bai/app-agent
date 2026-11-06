# 新场景开发试跑

入口：`scripts/awm_new_pilot.py`。新数据使用独立 manifest，不通过绑定旧题库数量或哈希的清洗入口。默认模型为 gpt-6-luna，生成/独立审查 high，Actor none，Judge V2 low；总预算 30,000,000 tokens。所有 API 使用同一 SQLite 账本，SDK 隐式重试关闭。

## 本次结果

运行目录：`runs/awm_new_pilot100_20261104_v1`；数据目录：`data/awm-new-pilot100-20261104-v1`。

最终报告为运行目录中的 `results_zh.md`，最终统计为 `summary.audited.json`，复核为 `final_audit.json`。首批门槛未通过：10 个新候选、2 个环境、20 道题；独立 rubric 审查通过 9 题，来源/资产与真实对照复核后保留 8 题。9 个 Agent episode 中，8 个有效业务判定为 6 通过、2 失败；另 1 个规则错误排除。后续 80 题未生成。

两个环境都完成真实读写与重置隔离。场景级对照通过 1/2：实验室完整移动满足公开温度范围契约，但 rubric 将“−20°C 储存”误收紧为上下界都精确 −20，Judge 按错误规则判失败。审查全 2 分不等于业务规则正确。原始审查、执行、判分及修复前结果均保留。

## 使用

从仓库根目录执行，使用 `agent-world-model/.venv/bin/python`：

```bash
# 离线准备；两个目录都必须是新目录。
agent-world-model/.venv/bin/python scripts/awm_new_pilot.py prepare --run runs/my_pilot --data data/my_pilot
# 分阶段：scenarios / assets / rubrics / pilot / all。
agent-world-model/.venv/bin/python scripts/awm_new_pilot.py run --run runs/my_pilot --phase pilot
# 全流程仍先检查 20 题门槛；不通过不会生成后续 80 题。
agent-world-model/.venv/bin/python scripts/awm_new_pilot.py run --run runs/my_pilot --phase all
# 以下为离线汇总及兼容 JSONL 导出。
agent-world-model/.venv/bin/python scripts/awm_new_pilot.py report --run runs/my_pilot
agent-world-model/.venv/bin/python scripts/report_awm_new_pilot.py --run runs/my_pilot
```

API key 从环境变量或仓库 `.env` 加载，不写入资产。`run` 使用进程锁；已完成请求按输入身份复用，中断的付费请求或未封存 episode 不自动重放。已完成且未通过的审计门槛会阻止再次调度。未知 API 用量保守预留，缓存用量包含在实际用量中，不另行累加。

## 资产及证据

每个场景包含原始题面、schema、初态、工具契约、MCP 代码和工具清单、verifier、registry，以及逐题 D1 输入/选择/编译结果/独立审查。verifier 仅兼容旧资产格式，Judge V2 不依赖其 SQL 结论。

执行目录保存 Actor 全轨迹、初末数据库、证据封存和 Judge 只读补查/全部结论复核。`result.audited.json` 与 `pilot_gate.audited.json` 是后续审计层，原始 `result.json` 和旧门槛不覆盖。导出 `accepted_rubrics.jsonl` 只包含复核保留规则；`question_bank.jsonl` 保留全部生成题及拒绝/异常状态。

`repair_awm_pilot_*.py`、`continue_awm_pilot.py`、`finish_awm_pilot.py`、`check_awm_pilot_lab_control.py` 和 `audit_awm_new_pilot.py` 是本批具名修复/复核记录，其中部分为一次性操作，不应当成通用重试循环。执行代码变更通过 `execution_revisions` 哈希链记录；不能直接改冻结资产后复用旧结果。

本批只用于开发，不进入正式训练或最终测试集；没有运行 GRPO、导出正式奖励或测量人工金标准准确率。
