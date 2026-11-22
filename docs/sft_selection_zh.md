# 96GB GPU：SFT、验证选模与最终测试

## 2026-11-10：补回两个验证 worker

原 worker 3、13 被 SIGKILL 终止，原因尚未确认。它们领取的两项任务均未创建输出目录，
审计后退回 pending，记录为控制目录下 `unstarted_worker_recovery_20261110.json`。
按用户要求恢复约 85 GiB 显存目标，`scripts/restore_sft_validation_workers.py` 错开 30 秒
启动两个替代进程，保留其他 16 个 worker。新的 PID 写入原 `validation/launch.json`，
显存监控动态读取该文件，超出 86 GiB 时仍在任务边界退役进程。
原 supervisor 的进程对象仍指向旧 PID；实际总并发与补位进程退出状态应查看
`runs/sft_validation_resume_20261109_v1/replacement_workers/status.json`。
补位管理器在所有进程结束后检查队列和归档；仅当全部任务 done 时，修正原 supervisor
因旧 PID 退出而产生的过期失败状态。Judge 仍等待注册进程退出及原 supervisor 锁释放。

## 2026-11-09：停止重复训练，恢复旧 checkpoint 验证选模

用户要求停止重复的 veRL SFT，并复用已完成的四个旧 checkpoint。
`runs/qwen35_verl_sft_20261109_v4` 已在第 7 步后停止，日志和停止请求保留。

接续入口为 `scripts/resume_sft_validation.py`，控制记录目录
`runs/sft_validation_resume_20261109_v1`。继续使用原始本地验证队列和统一补评目录，
四个 SFT 加基模对照各 589 题，已有完整结果保留；本次仅完成验证和选模，不自动开始测试。

旧任务中断原因是网络存储盘 inode 耗尽。18 个遗留 claimed 中，2 个具有完整封存结果，
直接置为 done；16 个确实因 supervisor 退出被终止，保留原输出和预算，使用新的
`*_storage_retry1` 目录补跑。3 个 Judge 补评已经完成，直接根据缓存恢复 recovery.json，
没有追加 API 调用。恢复后队列 614 done、2066 pending，另有此前复用的 265 个原始结果。

所有旧评测实现文件的 SHA256 均未变化；源代码集合只新增了 `training/verl/` 模块。
计划增加这些训练模块的源哈希，完整原计划、队列快照、claim 决策和文件哈希均已保存到控制目录。
补评代码、rubric、权重、greedy、EOS、固定分母、90% 覆盖率和胜者不确定性检查保持原规则。

后续 episode 归档到 `/root/autodl-tmp/awm-sft-validation-resume-20261109`，逐文件校验后保留原路径链接。
SQLite 仍在本地。初始 18 个模型 worker；显存目标约 85 GiB，连续采样超过 86 GiB 时，
按任务边界逐个减少 worker，最多减至 14，等待当前任务/Judge 完成才释放对应模型。
因此这是软目标，允许短时峰值；不通过中断正常答题或裁剪上下文来压低显存。
`primary/gpu.jsonl` 保存采样，`worker_control.json` 保存并发调整记录。

验证进度：https://swanlab.cn/@Alden-bai/awm-agent/runs/y82cyk2g 。
补评与选模：https://swanlab.cn/@Alden-bai/awm-agent/runs/iaso50aq 。
本次恢复和既有选模相关 23 项测试通过。


## 2026-11-09：无有效评分补评与选模接续

最终选模控制已接到 `runs/sft_judge_recovery_20261109_v1`，控制器为
`scripts/sft_judge_recovery.py`。原始验证仍由 `qwen35_sft_eval_local_20261109_v1`
及速度对照结束后的续跑控制器产生；补评不修改原始轨迹、评分或封存哈希。
SwanLab：https://swanlab.cn/@Alden-bai/awm-agent/runs/cmqrknkv 。

12:07:21 UTC 状态中的 116 次无效评分已逐条冻结到 `initial_audit.json`：
查询轮数耗尽 47、必要证据未查完 33、其他评审错误 20、动作后快照外键检查中断 14、rubric_error 2。
这些是评分不可用的次数，不能直接视作答错，也不能算成功。

冻结补评策略见 `policy.json`。所有候选（包括基模）采用同样规则：
已有有效成功/失败评分保留；执行完整且末态可信的无效评分只追加一次正式 Judge 补评。
补评使用相同原题、固定 rubric 和封存证据，主查询上限 12 轮、复核 4 轮、格式纠错 2 次，
最多 32 次 API 请求；4 路并发，不加载 GPU 模型。统一修复查询目标覆盖范围扩展的误报，
继续核查可补查的必要事实，并给自相矛盾的输出提供具体条目提示。真实歧义和真实评分规则问题仍保留。
`judge_jobs/` 中四条调试试跑仅用于修复诊断，不参与正式选模；正式补评位于 `judge_jobs_v1/`。

动作后外键违例属于逻辑状态，应作为证据交由评审判断，而不应使物理 SQLite 快照失败。
外部适配器保留严格的初态检查和每次快照的物理完整性检查，记录全部外键违例，不修改数据库行。
原始验证和速度对照释放 GPU 后，仅对明确被旧快照检查提前中断的题追加一次模型补跑。
权重、greedy 解码、EOS、任务划分与固定分母保持原计划；其他模型失败不重采样。
补跑证据同样封存，随后适用同一套 Judge 补评策略。

原始目录 `test/JUDGE_RECOVERY_PENDING.json` 阻止旧控制器提前启动测试集。
最终有效结论以补评目录中的 `selection.json`、`summary.json` 为准，不采用原始目录的中途选模结果。
控制器等待全部验证及补评完成，仍按确认成功数/589、覆盖率、奖励和稳定名称顺序选 SFT；
覆盖率门槛仍为 90%。另记录 `selection_uncertainty.json`：若剩余未评分题全部成功时可能改变第一名，
状态变为 `needs_attention` 并暂缓测试，不宣称已可靠选出最佳者。
通过后仅评测基模和所选 SFT 的测试集（各 588），再按相同规则补评；测试成绩不参与选模。

新记录归档到 `/autodl-fs/data/awm-sft-judge-recovery-20261109`，逐文件校验哈希后才替换为目录链接；
SQLite 队列与预算账本保留本地。`status.json` 记录已补评、恢复有效、仍未评分和待模型补跑数量，
SwanLab 展示对应数量及每个候选的补评奖励与后续正式验证/测试指标。


## 当前有效评测：结束标记修复

两组训练与全部四个 checkpoint 已完成，权重继续使用 v2 目录。旧验证出现输出
达到 2048 token 后仍续写角色的问题：监督目标以 `<|im_end|>`（248046）结束，
而当时保留的生成配置只接受 `<|endoftext|>`（248044）。这属于推理配置错误，不能用于评价 SFT 效果。

当前评测目录是 `runs/qwen35_sft_eval_eosfix_20261109_v1`，对应数据盘
`/autodl-fs/data/awm-sft-eval-eosfix-20261109`。统一的停止集合为 `[248044, 248046]`，
版本为 `generation-and-tokenizer-eos-v1`，写入每题 manifest 和策略记录。
不修改既有权重，不重训；后续训练入口也不再恢复过期 EOS 配置。

旧协议未开始任务转为 `held_protocol`，已领取任务自然结束，旧记录完整保留，
不混入新比较。四个 SFT 模型和基模在同一修正协议下重跑验证（5×589），
只在四个 SFT 候选中选择最佳者；随后评测该模型和基模的测试集（2×588）。
旧基模指标也不直接混作新协议下的对照。

真实 checkpoint 停止检查中，四个 SFT 模型对同一条首轮输入生成 18–22 token，
输出合法的 `search_tools` 动作并在 248046 停止。该检查仅为推理诊断，不计入正式成功率。
修复相关 79 项测试通过，包括两类 EOS 停止、策略 token 记录、完整选模顺序和基模不参与 SFT 选优。
新运行的 `tracking.json` 保存实际 SwanLab 链接，`status.json` 显示排空旧 worker、启动及验证进度。

### 长上下文显存调整与 OOM 补跑

正式验证中，20 个模型副本随 SFT 上下文变长挤满 96GB 显存，出现了明确的 CUDA OOM。
已通过 SQLite 领取事务触发器让 worker 18、19 完成当前任务后退出，保留该次结果并将下一道题留在队列中。
剩余 18 个 worker 继续处理所有候选；实测显存降至约 85GiB。
这两个预期退出在旧 supervisor 中可能表现为失败退出码，调整依据记录在 `memory_adjustment.json`。

`scripts/sft_eval_memory_recovery.py` 已安排自动接续，目录为
`runs/qwen35_sft_eval_memory18_20261109_v1`（数据盘 `awm-sft-eval-memory18-20261109`）。
它等待当前验证全部完成、原 supervisor 退出，然后只重试原始 `episode_result.json` 中
明确记录 CUDA OOM 且执行状态为 runtime_error 的任务。正常成功、模型失败以及其他评分异常均保留原尝试；
每次补跑绑定原轨迹路径、原结果与 evidence seal 哈希，补跑结果不会按高低分挑选。
汇总沿用每个模型 589 题的固定分母，Judge 用量包含原运行和补跑。
补跑完成后，只在四个 SFT 候选中选择最佳者，再以 18 个 worker 评测最佳模型和基模的测试集。
触发器不丢任务、只补跑显式 OOM、其余结果保留等 11 项相关测试已通过。

以下为训练设置和原 v2 流程说明；其中旧 EOS 保留说明已被上述修复取代。

本轮正式目录为 `runs/qwen35_sft_selection_20261109_v2`，链接到数据盘
`/autodl-fs/data/awm-sft-20261109-v2`。系统盘剩余空间不足以保存全参数训练状态，
checkpoint 和评测证据均写数据盘。既有基模评测保留原样。

## 已冻结的实验

- 原始 Qwen3.5-2B 的全部 1,881,825,088 个文本参数参与训练；FP32 参数与优化器、BF16 混合精度、梯度检查点。
- 使用原划分的 758 条轨迹、6,714 条当前响应监督样本；没有把验证或测试任务放入 SFT。
- 学习率 `1e-5`、`2e-5`，分别从原始基模开始训练 2 epoch，cosine 调度、3% warmup、固定种子 20261106。
- 每次更新有效 batch=64，梯度累积=1；一次前后向最多 122,880 个 padded input tokens。
  长样本按行拆分，短样本合批；所有子批次使用同一个目标 token 总数归一化。
  只移除右侧 padding，不截断样本、不拼接不同轨迹、不改变监督 mask。
- 每个 epoch 保存模型、优化器、调度器和 RNG 状态，共 4 个候选。
  不再额外复制 final 权重；选定模型的位置写入 `selection.json`。
- 优化的 FLA 依赖仅通过训练子进程的 `PYTHONPATH` 引入。
  评测使用原基模环境和内核；推理终止 token 配置也保留基模设置，避免 Trainer 的自动对齐悄悄改变比较协议。

初版 131,072-token 配置完成了最长样本单步检查，但正式训练第二步因显存碎片 OOM。
原始运行、SwanLab 失败实验和日志保留在 v1 目录；当时第一步学习率为零，尚无有效参数更新。
v2 使用 `PYTORCH_ALLOC_CONF=expandable_segments:True`，降低单次 token 上限，
并在启动前额外检查优化器状态常驻后的连续多步训练。
修正后的三步检查通过，累计处理约 138 万个有效输入 token，约 222 秒；
实际数值以 v1 目录下 `probes/adaptive64_expand/progress.json` 为准。
预检包含初次 kernel 编译，不能直接当作正式训练的稳定吞吐。
显存随长度、编译和保存阶段变化；不通过无用张量填充剩余显存。

## 验证与测试的顺序

两个设置训练完成后，四个候选分别评测完整的 589 道验证题。
20 个 GPU worker 使用 SQLite 共享队列，优先领取已加载模型的题目；
该候选没有待领取任务后，worker 可切换到其他候选帮忙。
任务只领取一次，已有失败或暂定通过不会为提高分数自动重采样。

沿用基模的 Landlock、greedy、关闭 thinking、上下文和调用限制、Judge V2 配置及逐项复核。
Judge 使用量继续逐题记账；累计请求/token 上限沿用原基模计划的 null。

选模规则提前写入 `plan.json`：

1. 每个候选必须完成全部计划题，且有效评分覆盖率至少 90%；当前基模有效覆盖率约 93.9%。
2. 首先最大化确认成功数 / 589，平局优先有效评分更多者，再比较评测奖励总和，最后按候选名称稳定破平。
3. 将结果及 checkpoint 哈希写入 `selection.json` 并计算哈希后，才创建测试任务。
4. 只对选定的 SFT 模型运行 588 道测试题。测试结果不进入选模函数。

异常评分保留为 null，并展示覆盖率，不能解释为模型业务失败或偷偷删除。
评测中断、worker 失败、没有合格候选时，状态记录为 failed 并保留原始尝试；不会伪造完整结论。

## SwanLab 和运行状态

项目：[awm-agent](https://swanlab.cn/@Alden-bai/awm-agent)。密钥从仓库 `.env` 读取，
不写入运行配置或文档。

- `sft-lr1e-5-*`、`sft-lr2e-5-*`：loss、梯度范数、学习率、entropy、token 准确率、输入 token 数与吞吐、GPU allocated/reserved 及峰值。
- 主实验 `awm-sft-20261109-v2`：基模指标、各候选验证奖励/成功率/覆盖率/步数、队列进度、Judge 用量、最终模型的测试指标。
- `eval_reward` 使用既有统计映射：确认通过 1、正常复核后的暂定通过 0.3、明确失败 0，异常为 null。
  SFT 优化目标是监督交叉熵，不产生 RL 训练奖励。

本地文件：

- `status.json`：阶段或失败原因；训练详细进度见 `training/lr*/progress.json`。
- `tracking.json` 与 `training/lr*/tracking.json`：实际 SwanLab 实验链接。
- `supervisor.log`、`train_lr*.log`：控制器和训练日志。
- `validation/<候选>/metrics.json`、`per_task.csv`、`report_zh.md`：每个候选的完整验证报告。
- `selection.json`：唯一选定模型及选择依据。
- `test/<最佳候选>/`：最终测试报告。
- `summary.json`：完成后的基模与最佳 SFT 比较。

入口为 `python -m harness_awm.training.selection`。本轮已启动时不要重复 prepare/run；
控制器用锁和启动记录拒绝重复执行。训练 checkpoint 保留恢复所需状态，但控制器不会自动覆盖已启动的实验。

验证包括 56 项相关测试（自适应梯度与真实小模型更新一致性、并发队列唯一领取、
验证选模与测试顺序、SwanLab 回调、已有本地评测等）及实际 2B 模型 GPU 长样本更新。

## 2026-11-09 评测协议与运行修复（以下覆盖前文初始运行配置）

两组全参数 SFT 均已完成，各保留 epoch 1 / epoch 2，共四个 checkpoint。
原始评测停止设置只包含 248044，没有包含 SFT 响应末尾的 248046。
已将停止策略统一为 `generation-and-tokenizer-eos-v1`，实际 token 集合为
`[248044, 248046]`；四个模型的真实生成检查通过，无需重新训练。
修复前的评测证据独立保留，不与修复后的结果混合。
基模也按相同停止策略重新评测：验证为基模加四个 SFT，共 2,945 次；
选模后测试为基模加最佳 SFT，共 1,176 次。测试仍不参与选模。

修复后运行 `qwen35_sft_eval_eosfix_20261109_v1` 初始采用 20 workers，
长上下文并发出现 38 条 CUDA OOM 结果。随后在任务边界退役两个 worker，
18 workers 时显存约 84–85 GiB；未中断已发出的 Judge 请求。
网络数据盘 `fuse.autofs` 上的 SQLite 锁等待随后造成 GPU 空转及
`OperationalError: database is locked`。将待领取任务置为 `held_storage`，
等待所有已领取任务结束后，冻结该轮证据。
最终队列为 302 done、1 error、2,642 held_storage；303 条结果中有效评分 224 条。
其中队列 error 的一题已经具有完整成功结果，直接保留，不重跑。

本地续跑入口：`scripts/sft_eval_local_recovery.py`。
运行目录 `runs/qwen35_sft_eval_local_20261109_v1` 是系统盘真实目录，
任务队列和每题预算数据库均在本地；checkpoint 继续留在数据盘。
265 条完整结果复用，38 条明确 CUDA OOM 重试，其余 2,642 条开始首次评测。
普通错误答案、Judge 无有效评分、token limit 不自动重试。
原始结果/封存哈希、旧队列状态及处理理由记录在 `recovery_audit.json`。
旧预算数据库在所有 worker 退出后复制到本地，未知用量仍保留原状态。
之前等待中的 memory18 恢复控制器已停止并标记被本地恢复替代。

为避免系统盘耗尽，每个完成任务的完整 episode 复制到
`/autodl-fs/data/awm-sft-eval-local-archive-20261109`，逐文件 SHA256 验证后，
原 episode 路径改为指向归档目录的链接。整目录链接保留原封存校验的路径边界。
活跃任务不移动，队列和预算 SQLite 不归档到网络盘。
本地剩余空间低于 1 GiB 时停止领取新任务并保留收尾结果。
续跑继续向 SwanLab 记录奖励、成功率、覆盖率、队列和 Judge 用量；
实际实验链接以该运行目录 `tracking.json` 为准。
恢复、队列和选模相关测试共 11 项通过，归档后原始 `verify_seal` 校验通过。

当前续跑 SwanLab：https://swanlab.cn/@Alden-bai/awm-agent/runs/9msp0m8c 。
使用真实完成 episode 的副本，归档前后全部 46 个封存文件校验一致。

## 快速内核实测：已测试，未切换正式评测

用户要求先启用快速内核查看收益后，使用独立进程、5 GiB allocator 上限、
不少于 8 GiB 空闲显存的启动保护，进行了真实输入的交替快慢对照。
探针没有调用 Judge，正式 18 workers 持续运行。
`causal-conv1d==1.7.0` 安装在 `runs/sft_eval_kernel_deps`，
FLA 0.6.0 复用 `runs/sft_kernel_deps`；正式进程的依赖路径未改变。

- `lr1e-5/epoch2`，FLA + causal-conv1d，3 个上下文、两轮交替测量：
  6/6 组生成 token 一致，单次生成 wall-time 加速比中位数 1.195。
- `lr1e-5/epoch1`，同样组合，3/3 组 token 一致，观察到中位加速比约 1.31。
- 基模同样组合：3 个输入中 2 个生成 token 不一致；其中一个参考输出 19 token，
  快速输出触及探针 96-token 上限，故不能把该组耗时比解释为等工作量加速。
- 基模只开 FLA：同样 2/3 不一致；只开快速卷积：1/3 不一致，匹配输出的耗时收益很小。
- 同一内核重复运行输出一致；差异在快慢内核之间可复现。
- 未完成另外两组 SFT 的额外探针：显存余量保护触发且基模兼容性门槛已失败。

这里的短探针不是完整任务吞吐测试，使用共享 GPU，输出上限为 96 token，
预热和编译从正式计时中排除，不能据此承诺整轮耗时缩短 20%。
当前决定为 `do_not_switch_active_evaluation`：保留正在进行的同口径正式评测。
预备的滚动切换程序没有启动，正式运行没有 `kernel_switch.json`。
完整记录在 `runs/sft_fast_kernel_report_20261109/summary.json`，
测速 SwanLab 链接见同目录 `tracking.json`。若以后统一使用快内核，
应明确另行冻结运行配置，不能将此次不一致样例当作无影响的直接替换。

## 共享模型批处理探针（2026-11-09）

用户要求试验共享模型、批量推理的收益。入口为
`scripts/benchmark_shared_batch.py`，保持现有 Transformers 参考内核、
BF16、SDPA、greedy、关闭 thinking 和 EOS 集合，独立使用一份模型。
固定工作量回放用于吞吐比较，自然生成另外检查逐 token 一致性；
batch=1 自定义解码与现有 actor 的输出一致。3 项停止、padding 与回放测试通过。
探针最多分配 5 GiB PyTorch 显存，未使用 Judge，正式任务继续运行。

SFT `lr1e-5/epoch2` 的测量结果（相对同一共享模型 batch=1 串行）：

| 输入集合 | batch | 固定工作量吞吐倍数 | PyTorch 峰值 GiB | 自然生成一致请求 |
|---|---:|---:|---:|---:|
| 8 个混合长度请求，679–4062 token | 2 | 1.509 | 4.632 | 7/8 |
| 8 个短请求，679–835 token | 2 | 1.611 | 3.778 | 7/8 |
| 同上 | 4 | 2.740 | 4.044 | 8/8 |
| 同上 | 8 | 4.092 | 4.574 | 7/8 |

混合长度集合的 batch 4/8 超过本次独立探针的 5 GiB 上限，未进入计时。
这不是完整 96GB GPU 的 batch 上限。短请求全部批次通过内存预检。
随后对基模进行 4 个短请求的 batch=4 自然生成兼容检查，2/4 完全一致，
另两个输出长度分别从 87/11 token 变为 95/19 token；基模这组没有给出
等工作量加速结论，也没有测任务正确率。

这些数字来自静态批处理原型，**不是 vLLM/连续批处理服务部署，也不是
与当前 18 个独立模型进程的受控吞吐对比**。GPU 与正式任务共享，
两轮计时顺序交替，预热排除，探针输出上限为 96 token。
没有直接修改正式评测运行配置或替换已完成结果。后续若采用批处理服务，
需要完整队列吞吐比较，并明确统一的新运行口径及输出变化。

完整报告：`runs/shared_batch_report_20261109/summary.json`。
SwanLab：https://swanlab.cn/@Alden-bai/awm-agent/runs/1rujv0gb 。
