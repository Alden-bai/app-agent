# Qwen3.5-2B 基模验证／测试集评测

## AutoDL 并发续跑（2026-11-09）

最新收尾运行目录：`runs/qwen35_base_tail_queue_20261109_v1`。
用户要求空闲进程帮助慢分片后，已在安全边界保留全部 1155 道既有结果，将剩余 22 道未开始题目
交给 `scripts/tail_baseline.py` 的 SQLite 共享队列，最多 20 个 GPU 进程。
完成一题的进程可继续领取下一道待运行题；事务领取保证任务唯一，已领取任务和失败结果均不自动重试。
`status.json` 分别记录 `queue_pending`（排队）、`queue_running`（已领取执行中）和 `queue_finished`。
当队列为空时，空闲进程正常退出；所有剩余题均已在途后，显存随任务完成而下降是正常现象。
验证／测试报告仍以固定 589／588 题汇总，Judge 用量包含前序全部运行及队列中的逐题账本。

以下保留固定分片阶段的实现和运行说明，旧目录状态指向最新续跑目录。

用户授权使用 Landlock 替代 namespace 隔离，并要求利用 96GB GPU 提高吞吐。
用户随后要求显存使用接近 90GB，固定分片运行目录为 `runs/qwen35_base_parallel20_20261109_v1`，由
`scripts/parallel_baseline.py` 启动 20 个独立 GPU 工作进程；每进程 1 个 CPU 线程，加载相同 BF16 基模，
使用原 greedy、种子、上下文、轮数、工具调用及 Judge 参数。累计复用原串行、8 路及 19 路运行的 135 道结果，
包括失败和评分异常，剩余 454 道验证题、588 道测试题按任务 ID 分到互不重叠的队列。
每题完成后释放未使用的 CUDA 缓存块以减少峰值，模型权重常驻 GPU。显存随当前上下文变化。
各工作进程完成自己的验证题后继续自己的测试题，验证／测试报告分别汇总。

切换前确认没有在途 Judge 请求、已推理题目均完成评分，并核验封存轨迹。
仅启动环境但尚未发出首次模型请求的下一题已记录在 `parallel_handoff.json`，交由新进程执行。
原始目录与结果不覆盖。新目录中的 `plan.json` 固定分片和结果路径；汇总按完整题目清单逐题合并，
不平均各分片成功率、不挑选最佳尝试。并发改变调度时序，报告注明 20 个工作进程及 Landlock；
后续模型阶段应使用相同执行条件。

运行状态见新目录的 `status.json`；各进程日志为 `worker_00.log` 至 `worker_19.log`；
`validation/` 和 `test/` 中的 `metrics.json`、`per_task.csv`、`report_zh.md` 每约 15 秒刷新。
`result_records` 包括尚未完成判分的执行记录；以 `scored` 和 `score_coverage` 判断有效评分进度。
既有评分异常不会因并发切换而自动重评。累计 Judge 请求/token 不限额，逐项记账，
汇总包含所有前序运行的预算账本且不重复计数。每次扩容保留前序驱动脚本快照和协议哈希。

不要重复运行旧的串行启动脚本。恢复中断的并发工作进程前，应核实任务分配及已生成结果；
当前控制器拒绝覆盖已启动的工作目录。

按用户最新要求，本轮直接运行 Base 的全量 validation 589 题和 test 588 题，分别出报告。
不运行训练集、不更新模型、不签发训练奖励准入。数据划分保持 `training_split_20261106_v1`。
取消此前 20 题试跑和预算推算的执行要求；旧模式作为可选能力保留。

## 已冻结的评测协议

- 本地原始 Qwen3.5-2B，按模型 lock 校验；单 GPU BF16、greedy、关闭 thinking。
- 每题一次 rollout，种子 20261106；32,768 上下文，2,048 单轮输出，32 模型轮次、32 业务调用上限。
- 沿用 single-JSON-v2 提示词、工具发现、上下文压缩及超时规则。
- 真实 Judge V2，`gpt-6-luna`，逐项复核；默认 strict 隔离，本机经用户明确要求改用 `landlock`。密钥只来自环境，服务地址及模型配置冻结进运行清单。
- 当前累计 Judge 请求/token 上限均为 JSON `null`，表示显式无限额；仍完整记账。
  每题轮数、上下文和超时限制保持固定。API 仅用于 Judge，Actor 使用本地权重。
- 可选旧模式的 20 题按 `SHA256("20261106:scenario:native_task_id")` 升序选择，优先不同 scenario。
  旧模式试跑预算 1,000 次请求、1,000 万 token；全量预算分别为 `ceil(试跑累计用量 / 20 × 589 × 1.5)`。
  未知用量使用原预算预留值，重试计入请求/token；耗尽后停止继续生成，保留全部计划任务分母。
- 启用可选试跑时，必须全部得到有效评测得分才能自动扩大规模；模型失败的 0 分不阻止扩大规模。
  无效样本或评测故障会暂停到 `pilot_needs_review`，不会自动删题、换题或挑最好结果。

## 首次启动

使用 Python 3.12 与 `requirements/training-cu128.lock.txt`，可从 `wheelhouse/cu128` 离线安装。
本机已经在 `runs/qwen35_base_setup/venv` 创建独立环境，保留原迁移环境不变。
所有命令在仓库根目录运行。

```bash
export PATH="$PWD/runs/qwen35_base_setup/bubblewrap/usr/bin:$PATH"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

# 清单审计：不生成环境，不调用 API。输出必须使用新文件名。
runs/qwen35_base_setup/venv/bin/python -m harness_awm.training.tasks \
  --split data/training_split_20261106_v1 --audit \
  --output runs/runtime_sources_audit.json

# 原题、rubric、筛查结论之外，还必须有冻结数据库、环境及工具目录。
runs/qwen35_base_setup/venv/bin/python -m harness_awm.training.tasks \
  --split data/training_split_20261106_v1 \
  --output data/training_runtime_20261109_v1

# 冻结两份评测集、模型/源码哈希、Judge 配置；实测严格隔离，不发起模型或 API 调用。
runs/qwen35_base_setup/venv/bin/python -m harness_awm.training.baseline prepare \
  --tasks data/training_runtime_20261109_v1 \
  --model models/Qwen3.5-2B \
  --runtime-python "$PWD/runs/qwen35_base_setup/venv/bin/python" \
  --output runs/qwen35_base_heldout_v1 \
  --include-test --skip-pilot --unlimited-judge-budget

# 重新检查版本与隔离，直接运行完整验证集和测试集。
runs/qwen35_base_setup/venv/bin/python -m harness_awm.training.baseline execute \
  --output runs/qwen35_base_heldout_v1
```

`prepare` 写出 `plan.json`、`pilot_tasks.json`、`pilot_config.json` 和隔离检查结果。
当前模式的 `execute` 产生独立的 `validation/`、`test/` 报告及状态文件；预算文件记录无限额模式，不推算上限。
去掉三个模式选项可恢复原先的 20 题试跑后仅验证集评测。
修改代码、权重、Judge 配置或题包后，必须在新目录重新准备协议。
SwanLab 沿用仓库现有配置；只需本地报告时可设 `SWANLAB_MODE=offline`。

## 指标

原有成功率字段保持兼容。新增 `metrics.json.evaluation` 提供受有效执行条件约束的得分与统计：

- `success_over_planned`：确认成功 / 全部计划题数；`success_over_scored`：确认成功 / 有效得分题数。
- `score_coverage`：有效得分 / 全部计划题数，必须与成功率同时展示。
- `eval_reward`：确认成功 1、正常复核后暂定通过 0.3、明确失败 0；异常与缺失为 null。
  必须是已结算、真实 Judge V2、有效评测、可信执行和可信终态。它是评测统计，训练 `reward=null` 不变。
- `steps`：有执行记录、有效判分、确认成功三个子集；分别统计 `turns` 和 `tool_calls` 的样本数、均值、中位数、P90。
  P90 用最近秩法；缺失记录不算零，真实零调用保留。
- `turns` 含发现工具、纠错和最终回答；`tool_calls` 只计通过协议校验后实际发起的业务调用，包含返回工具错误的调用。
- `wall_seconds` 包含串行任务和 Judge 时间；逐题表保留执行耗时及 actor token，Judge 用量来自预算账本。

每次评测输出 `per_task.csv`、`metrics.json` 和 `report_zh.md`。逐题 CSV 包含原任务 ID、判定、得分、步数、终止原因及完整轨迹路径。
不改变 episode 原始结果或训练奖励字段；达到轮数上限的模型失败仍参与有效评分。

## 中断与补跑

输出目录不可覆盖。完整有效的 pilot 可以在同一冻结计划下复用，以继续尚未创建的 validation。
部分 pilot 或已经创建的 validation 不会自动重跑。先检查每题证据；仅缺失记录或确认的基础设施故障允许补跑。
使用现有 `harness_awm.training.evaluate --task scenario:index` 在新目录执行指定任务，沿用原冻结配置、模型及采样方式。
原始尝试必须保留，补跑须记录原任务 ID、旧轨迹、新轨迹和基础设施原因；模型失败和合法 0.3 不得为提升分数而重采样。
当前入口不自动合并补跑报告，不能把不同批次均值直接平均或挑最好尝试宣称全量结果。

## 本机采用 Landlock（2026-11-09）

用户明确要求更换隔离方案以跑通评测。本机 namespace 被外层容器禁止，但内核支持 Landlock ABI 1。
新增显式 `--isolation landlock`：限制 worker 打开文件读写，Python/依赖和指定源码只读，
每题独立数据库、工作目录与临时目录，子进程继承限制；保留干净环境变量和进程组清理。
不会自动从 strict 降级；Landlock 不可用或探测失败时停止。formal/训练奖励仍要求 strict。

这不是与 strict 等价的沙箱：不隔离 PID 和网络，ABI 1 不限制 truncate，chmod/stat 等元数据操作也未全面限制。
只适用于当前受控评测环境；运行清单和报告明确标记 `landlock`，不冒充 strict 结果。
技术依据：https://docs.kernel.org/userspace-api/landlock.html

冻结题包已迁移至 `data/training_runtime_20261106_v1`，validation 589 题和 test 588 题已通过静态检查。
原模型 CUDA BF16 greedy 短生成检查见 `runs/qwen35_base_setup/gpu_smoke.json`。
Landlock 实测见 `runs/qwen35_base_setup/landlock_probe/result.json`；真实环境启动和工具目录检查见
`runs/qwen35_base_setup/landlock_environment_smoke_v2/smoke.json`。

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 SWANLAB_MODE=offline \
  runs/qwen35_base_setup/venv/bin/python -m harness_awm.training.baseline prepare \
  --tasks data/training_runtime_20261106_v1 --model models/Qwen3.5-2B \
  --runtime-python "$PWD/runs/qwen35_base_setup/venv/bin/python" \
  --output runs/qwen35_base_heldout_landlock_20261109_v1 \
  --include-test --skip-pilot --unlimited-judge-budget --isolation landlock

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 SWANLAB_MODE=offline \
  runs/qwen35_base_setup/venv/bin/python -m harness_awm.training.baseline execute \
  --output runs/qwen35_base_heldout_landlock_20261109_v1
```

使用已批准的命令执行通道启动以上命令。无需修改宿主内核参数或启用容器特权模式。
每次修改源码后需用新目录 prepare；旧 strict 阻塞记录和旧实验保持原样。
