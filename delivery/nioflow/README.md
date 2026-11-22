# NIOFlow 八卡实验交付

本包在新的解压目录运行数据处理、Qwen3.5-27B 全参数 SFT、2,000 题在线 GRPO 和 SkillRL 配对评测。完整实验要求见根目录的 `任务_全量GRPO.md`。代码、冻结题库、来源轨迹、监督标注、SQLite 初始库、工具服务、Judge 规则、分词器、技能库和 veRL 源码均在包内。三套 Python 环境、CUDA 工具链和基模权重使用接收方已准备的挂载。Judge 凭据由 Job 注入。

## 接收方 Codex 准备文件

接收方 Codex 在可访问共享盘的终端校验并解压交付包，按下方路径表核对三个解释器、模型和输出目录，然后把资源配置和启动命令交给平台操作人。在存放压缩包和校验文件的目录执行以下命令。将示例中的 `/shared/awm-deliveries` 替换为实际共享盘目录。若沿用站点已有环境、模型和输出卷，在 Job 中分别设置 `AWM_ENV_ROOT`、`MODEL_DIR` 和 `AWM_RUN_ROOT`。

```bash
sha256sum -c awm-nioflow-cu128-grpo2000.tar.gz.sha256 &&
  mkdir -p /shared/awm-deliveries &&
  test ! -e /shared/awm-deliveries/awm-nioflow-cu128-grpo2000 &&
  tar -xzf awm-nioflow-cu128-grpo2000.tar.gz -C /shared/awm-deliveries
```

## 人工在 NIOFlow 平台提交 Job

平台操作人手动选择 **1 个节点、8 张 A100 80GB、1 个启动进程**。项目入口会启动各个训练 rank。环境沿用已验证的单卡诊断环境 的 CUDA 12.8 配置、共享盘挂载和 Judge Secret，保留已验证的 `CUDA_HOME`、`PATH` 和 `LD_LIBRARY_PATH`。为 CPU offload 分配至少 384 GiB 可用内存，建议 512 GiB 或以上。实验输出使用有数 TiB 配额的共享卷，正式训练会保留四份 SFT 和四份 GRPO checkpoint。

将下面一行粘贴到平台的 Job 启动命令栏，然后由操作人点击提交。Job 获得资源后，平台会执行这条命令。

```bash
bash /shared/awm-deliveries/awm-nioflow-cu128-grpo2000/run.sh all
```

提交后，操作人将平台生成的 Job ID 和可读取的日志路径或日志内容交给 Codex。Codex 根据日志核对首个实际训练 step、所选 worker 后端及四库短测结果。结束后根据共享盘产物整理 `online/experiment-results.json`、两份验证选模表及 GRPO 覆盖表。若作业提前退出，记录退出码、阶段名、首个异常和对应日志。

## 本机路径与配置

| 项目 | 默认值或配置方式 |
|---|---|
| `AWM_ENV_ROOT` | 默认使用解压目录下的 `.venv-delivery-env`、`.venv-delivery-sft`、`.venv-delivery-grpo`。这里只取已安装的解释器及第三方依赖。项目模块从新解压目录载入。也可以分别指定 `ENV_PYTHON`、`SFT_PYTHON`、`GRPO_PYTHON` 的绝对路径。 |
| `MODEL_DIR` | 默认使用解压目录下的 `models/Qwen3.5-27B`，官方非量化权重及原 tokenizer。 |
| `AWM_RUN_ROOT` | 默认使用解压目录同级的 `awm-runs/nioflow-cu128-grpo2000-v2`。`SFT_WORK` 和 `ONLINE_WORK` 默认是其下的 `sft`、`online`。 |
| Judge | 通过 Job Secret 注入 `OPENAI_API_KEY`。优先沿用接收方实验使用的 `AWM_JUDGE_BASE_URL`、`AWM_JUDGE_MODEL`、`AWM_JUDGE_KEY_ENV`，也识别 `OPENAI_BASE_URL`、`OPENAI_MODEL`。启动器依次查找新项目和 `AWM_ENV_ROOT` 的 `.env`，仅读取 Judge 字段，也可用 `AWM_JUDGE_ENV_FILE` 指定文件。训练、验证与测试使用同一 Judge 配置。包内原配置为 `https://api.openai.com/v1` 和 `gpt-6-luna`，使用前核实账号具有该模型或指定已开通的评判服务。 |

历史单卡诊断记录是 Ubuntu 20.04.3、glibc 2.31、Linux `5.4.0-216-generic`、驱动 `535.216.03`、Python `3.12.16`。SFT 实测为 torch `2.10.0` / CUDA 12.8，GRPO 为 torch `2.11.0+cu128` / vLLM `0.20.2`，两者的 Transformers 均为 `5.22.1`。新入口选择 `local-sdpa-cu128`，使用站点现有的 12.8 编译器和 cu12 NCCL。八卡、vLLM 权重同步、一次有效更新及 Judge 评分须以本次 Job 实际日志验收。

## 运行与验收

| 入口 | 工作与验收产物 |
|---|---|
| `bash run.sh cpu-check` | 在有 CPU Python 环境的节点校验整个包、冻结证据、监督掩码和技能库。实测同一道训练题的四份独立 SQLite 副本，各写一条不同角色，核对四次工具清单与冻结哈希，产出 `worker-smoke.json`。 |
| `bash run.sh rebuild-data` | 用审核决定和来源轨迹重建各划分，再用包内 27B tokenizer 重编码。逐文件比较划分 JSONL 和用于 SFT 的 `train.jsonl`。产物在 `data-rebuild/`。需要 SFT 解释器，全程使用 CPU。 |
| `bash run.sh check` | 校验权重、两套训练环境、实际八卡 NCCL、工具读写与 Judge 配置。真实 API 调用从正式在线校准开始。 |
| `bash run.sh all` | 校验数据并重建训练输入，在线预检，SFT 最长样本单步与 HF 重载，两组 SFT，校准和全量评测，GRPO 单步及正式全量训练，最后选模和 SkillRL 测试。正式流程以前台退出码和 `ONLINE_COMPLETE` 为终态。 |
| `bash run.sh sft` / `bash run.sh online` | 单独运行对应阶段。在线预检失败时，`all` 会记录 `ONLINE_BLOCKED`，完成可独立运行的 SFT 后返回失败状态。修正后用 `online` 接续同一 `SFT_WORK`。 |

题目进程在 `init` 时实际测试 Landlock、bubblewrap，随后可选 `private-root`。第三种路径用包内官方生成器预先渲染的 980 个冻结环境，在轨迹目录内运行原 FastAPI/MCP 工具和 SQLite。它使用本进程已有的 `CAP_SYS_CHROOT`，或在平台准许的用户命名空间内切换根目录，随后清空 capabilities、启用 `no_new_privs`，通过系统的 `libseccomp.so.2` 加载系统调用白名单。进程只继承该轨迹的回环监听 socket。平台须允许该子进程执行相应操作。可用 `AWM_WORKER_ISOLATION=private-root bash run.sh cpu-check` 单测这条路径。所有允许的后端均需通过实测。

`private-root` 会拒绝题目主动外连。冻结工具中 `billing_and_payments_1` 的 `send_test_webhook_event` 在 URL 非空时尝试发送 HTTP。该环境入选的五道训练、验证和测试题均未要求发送 webhook。模型自行选择此操作时按实际工具错误记录。实验报告保留实际使用的隔离后端。

## 实验口径

- 数据审核从 966 条候选保留 758 条轨迹。7,071 个轮次中屏蔽 357 个，得到 6,714 条只监督当前助手回复的 SFT 样本。冻结审核、来源输入和掩码可在解压目录重放。`data/awm-dde80a0-repaired-v1` 附修复后的题库、数据库和变更记录。
- 两组 SFT 均从相同基模开始，学习率分别为 `1e-5`、`2e-5`，各跑 2 个 epoch。每个 epoch 保存一次，共 4 个 checkpoint。各自跑 589 题验证集，选最佳权重作为 GRPO 起点。
- GRPO 从冻结的 2,746 题训练池按项目种子 `20261106` 无放回选取 2,000 题，顺序见 `delivery/nioflow/grpo-2000.json`。每题在线采样 4 条，共 8,000 条。四条全失败直接跳过质量评分和更新。四条全成功逐条评质量，最终 reward 全相同则跳过，有差异则更新一次。评分异常、未知和缺失单列。按已处理题数在第 500、1,000、1,500、2,000 题保存四份 checkpoint，跳过更新的题也计数。第四份即最终权重。每份记录实际更新数。
- 基模单独跑 588 道测试题。所有保留的 GRPO checkpoint 用 589 题验证。固定最佳 GRPO 权重，用同一批 588 题做技能关／开对照。冻结技能库仅用训练来源，最多检索 3 条、5,000 UTF-8 字节。报告成功数、以全部 588 题为分母的成功率、评分覆盖、token、耗时和百分点差。

主要结果位于 `$ONLINE_WORK` 下的 `sft-selection.json`、`grpo-selection.json`、`grpo/coverage.json`、`grpo/summary.json`、`best-grpo-test/` 和 `experiment-results.json`。已验收阶段有完成记录。中断后保留原目录与日志。未完成的训练目录须核对进度和 checkpoint 后再作恢复决定，启动器会阻止覆盖或混合两次实验。更改模型、数据、代码或 Judge 后请使用新的实验目录。
