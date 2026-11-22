# 接收方执行说明

请先读根目录的 `任务_全量GRPO.md` 和 `README.md`。接收方 Codex 负责校验交付包、准备共享盘上的项目文件，核对模型与三个解释器路径，并把资源配置和以下一行命令发给平台操作人。

将示例共享盘路径替换为实际部署目录，模型、解释器和输出目录按交付说明配置。

```bash
bash /shared/awm-deliveries/awm-nioflow-cu128-grpo2000/run.sh all
```

平台操作人负责在 NIOFlow 手动选择单机 8×A100 80GB、一个启动进程，沿用 已验证的单卡诊断任务 的 CUDA 12.8 环境、共享盘挂载和 Judge Secret，粘贴启动命令并提交。提交后由操作人提供 Job ID 和日志，平台在分配的八卡节点上执行训练。

正式 GRPO 只执行 `delivery/nioflow/grpo-2000.json` 中按顺序冻结的 2,000 题，每题 4 条在线轨迹。四条全失败时直接跳过，四条全成功时每条均评质量，比较四个最终 reward 后决定是否更新。第 500、1,000、1,500、2,000 题各保存一次，第四份即最终 checkpoint。回传四个题数和各自的实际优化器更新数。

收到 Job ID 和日志后，核对后端选择、`SFT_SMOKE_PASSED`、GRPO 真实更新、`coverage.json` 和 `ONLINE_COMPLETE`，按实际产物整理成绩。`CPU_CHECK_PASSED`、`SFT_COMPLETE` 和 `ONLINE_BLOCKED` 各自只代表该阶段的状态。保留首次报错和 checkpoint，处理故障时使用 README 中的分阶段入口。
