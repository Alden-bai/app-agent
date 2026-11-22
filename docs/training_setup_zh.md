# Qwen3.5-2B 全参数训练与离线预检

SFT / GRPO 的新 verl 后端及独立安装/启动流程见 [verl 迁移说明](verl_training_zh.md)。本页的 TRL SFT 和单进程 GRPO 用法保留为原实现与离线语义参照。

本项目使用 **PyTorch + Transformers + TRL + Accelerate/FSDP2**。AWM 提供真实环境和执行证据，训练框架负责梯度更新。当前数据为纯文本：通过 `Qwen3_5ForCausalLM` 加载官方 checkpoint 的全部文本参数，不加载视觉塔，不使用 LoRA、量化或冻结文本层。

## 当前已验证的范围

- 独立 Python 3.12 环境 `.venv-training`；原 `agent-world-model/.venv` 不需要迁移。
- 已在另一个空虚拟环境中，用锁文件和本机缓存完全离线重装，并重新运行 SFT/GRPO smoke；避免仅验证当前环境恰好能导入。
- CUDA 12.8 的完整安装包约4.1 GB，161个依赖已逐文件校验；另一个空环境已仅从这些 wheels 离线安装，依赖检查和 Qwen3.5/TRL 导入通过。编译架构含 `sm_86`（兼容4090）和 `sm_120`（5090）；这不是实际GPU执行测试。
- PyTorch 2.10.0、Transformers 5.22.1、TRL 1.17.1、Accelerate 1.15.0、Datasets 5.1.0；完整传递依赖见 `requirements/` 的 lock 文件。
- 真实 Qwen3.5-2B 权重、tokenizer、模板已从 Qwen 官方 ModelScope 仓库下载；每个文件绑定 revision、大小和 SHA256，见 `configs/training/qwen3.5-2b-model.lock.json`。
- 缩小尺寸的 Qwen3.5（同时包含线性注意力和完整注意力）在 CPU 上完成 SFT、GRPO 各一次真实优化器更新，并完成 SFT checkpoint 保存/加载。
- **实际预训练模型**的 1,881,825,088 个文本参数全部可训练，使用 25-token 合成输入在 CPU 上完成一次 SFT/AdamW 更新。loss 约 2.21034，权重发生变化。诊断更新未写回原始权重。
- 当前机器 CUDA 不可用，**GPU 显存、FSDP2/NCCL、多卡吞吐和 GPU 内核尚未实测**。准备了 4×4090 与 3×5090 配置，但不把 CPU 通过当成多卡验收通过。

本地证据：

- `runs/training_preflight_20261106_v2/report.json`：小模型 SFT/GRPO。
- `runs/training_qwen35_2b_model_check_v2/report.json`：真实 2B checkpoint 的 CPU 更新。
- `data/training_qwen35_2b_v1/report.json`：全量数据和 token 统计。
- `runs/training_cuda_offline_install.log`、`runs/training_cuda_binary_check.json`：CUDA 完整离线安装及二进制架构检查。
- `runs/training_setup_20261106_summary.json`：本次准备的状态汇总。

## 数据已转换为可训练格式

保持已有划分：758 条 SFT 轨迹，GRPO 训练 2,746 题，验证 589 题，测试 588 题。SFT 的 7,071 轮中，357 轮不直接监督，产出 6,714 个样本；被屏蔽动作仍保留在后续轮次的原始输入中。

`train.jsonl` 的 `labels` 只监督当前 assistant 响应及结束标记；系统、用户、历史动作、环境观察和 padding 均为 `-100`。监督全被屏蔽的轮次不单独进入 batch。冻结 manifest 和逐轮源文件都校验哈希；Judge 私有 rubric、审查意见不进入模型输入。模板固定 `enable_thinking=False`，与现有单 JSON 动作目标相配。

| 统计 | token 数 |
|---|---:|
| 总输入（包含重复历史） | 20,814,467 |
| 实际监督目标 | 235,195 |
| 长度中位数 / P95 | 2,684 / 6,867 |
| 最长样本 | 17,743 |

4K/8K/16K 分别会超限 1,697 / 150 / 4 个样本。默认长度上限 32K，不进行全长 padding，不静默截断或丢弃。数据准备可在 CPU 上完成：

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv-training/bin/python -m harness_awm.training.data \
  --split data/training_split_20261106_v1 \
  --tokenizer models/Qwen3.5-2B \
  --output data/training_qwen35_2b_v2
```

输出必须是新目录。迁移仓库后，原始 manifest 的绝对路径可用 `--root 新仓库路径` 重定位，不需要修改被冻结的文件。训练直接使用已经准备的 232 MB 数据包，不要求搬运全部原始历史轨迹；重新做源文件验证时则需要这些原始材料。

验证集/测试集是环境任务，没有参考动作轨迹，不能直接当作 SFT `eval_loss` 数据集。SFT 入口不自动挪用测试集；模型选择应在验证任务上执行 harness，测试集留到最终评测。

## 当前机器：安装和重跑离线检查

需要 `uv==0.11.23`、Python 3.12。安装脚本只管理 `.venv-training`。

```bash
bash scripts/setup_training.sh cpu
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv-training/bin/python -m harness_awm.training.smoke \
  --output runs/my_training_smoke
.venv-training/bin/python -m harness_awm.training.doctor --model models/Qwen3.5-2B
```

CPU smoke 自建随机小模型和 tokenizer，不联网、不调用 Judge，不依赖 2B 权重。真实模型检查会占用数十 GB CPU 内存：

```bash
.venv-training/bin/python -m harness_awm.training.model_check \
  --model models/Qwen3.5-2B --output runs/my_2b_model_check
```

重新下载固定模型资产（支持断点续传，完成后核验 SHA256）：

```bash
python scripts/download_training_model.py --weights
```

## 搬到 GPU 机器

准备 Linux x86_64、Python 3.12、足够新且能支持 CUDA 12.8 的 NVIDIA 驱动。5090 需要 Blackwell 支持，不能把 CPU wheel 或旧 CUDA wheel 直接复制过去使用。Python 虚拟环境包含机器路径，不直接搬 `.venv-training`。

提前传好代码（包括 `agent-world-model` 子模块）、`models/Qwen3.5-2B/`、`data/training_qwen35_2b_v1/`、`requirements/` 和 `wheelhouse/cu128/`。训练本身不需要完整原始 `runs/`；真实 GRPO/harness 另外需要其冻结环境及证据资产。不要把 `.env` 密钥文件打入公开包。

若租赁平台允许提前使用数据盘或 CPU 容器，先完成这些文件的传输和安装，再启动 GPU 计费；否则即使包已在本机缓存，上传等待仍可能占用租机时间。

GPU 锁分两种：`training-cu128.lock.txt` 是核心框架；`training-cu128-kernels.lock.txt` 另含 `flash-linear-attention==0.6.0` / `fla-core==0.6.0`，用于 Qwen3.5 线性注意力。GPU 内核在目标卡第一次运行时可能需要 Triton JIT；这一耗时和兼容性只能在实际 GPU 上确认。未安装独立 `causal-conv1d` 时，短卷积会使用 PyTorch 参考实现。

wheelhouse 附带与锁文件同名的离线锁，以及逐文件 SHA256 清单。`pwinput` 上游仅提供源码包，已预先构建纯 Python wheel，其新哈希单独冻结。更改/补充缓存后重新验证：

```bash
.venv-training/bin/python scripts/verify_training_wheelhouse.py \
  --lock requirements/training-cu128-kernels.lock.txt --wheelhouse wheelhouse/cu128
```

```bash
# 有完整 wheelhouse 时，无须连接包索引。
bash scripts/setup_training.sh cu128-kernels wheelhouse/cu128

# 未缓存时也可在线按相同锁安装。
# bash scripts/setup_training.sh cu128-kernels
```

如果机器没有 uv，可先用 Python 的 pip 从已准备的 wheelhouse 安装 `uv==0.11.23`。`bubblewrap` 是真实 harness 严格隔离的系统依赖，需要宿主允许用户/PID namespace；Python 包安装不能替代这个要求。

先跑一次不训练的 GPU 检查，再测最长样本：

```bash
.venv-training/bin/python -m harness_awm.training.doctor --require-cuda --model models/Qwen3.5-2B
.venv-training/bin/torchrun --standalone --nproc_per_node=4 \
  -m harness_awm.training.doctor --require-cuda --distributed

SWANLAB_MODE=disabled HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  .venv-training/bin/accelerate launch --config_file configs/training/fsdp2-4x4090.yaml \
  -m harness_awm.training.sft \
  --model models/Qwen3.5-2B --data data/training_qwen35_2b_v1 \
  --output runs/qwen35_sft_gpu_preflight --max-steps 3 \
  --gradient-accumulation 1 --preflight-longest
```

3×5090 把 `--nproc_per_node` 改成 3、配置换成 `configs/training/fsdp2-3x5090.yaml`。最长样本检查会更新该次独立模型，并写到独立输出目录；正式训练仍从原始模型开始，不能把调试产物混为正式结果。若最长样本 OOM，先调整激活内存/分片方案，不能隐式裁掉这些数据。

短测通过后，正式 SFT 示例：

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  .venv-training/bin/accelerate launch --config_file configs/training/fsdp2-4x4090.yaml \
  -m harness_awm.training.sft \
  --model models/Qwen3.5-2B --data data/training_qwen35_2b_v1 \
  --output runs/qwen35_sft_full_v1 --epochs 1 --gradient-accumulation 8
```

这是起始超参数，不是效果调优结论。每卡 batch=1，4卡/3卡累积8次的全局 batch 分别为32/24；如要两种硬件保持全局 batch=24，4卡设累积6次，3卡设8次。默认 FP32 参数和 AdamW 状态、BF16 计算、FSDP2 分片、激活检查点；宿主建议准备至少64 GB内存。最终模型位于输出目录 `final/`；可用 `--resume 输出目录/checkpoint-N` 恢复保存的训练状态。SwanLab 回调沿用工作区配置，离线检查显式设 `SWANLAB_MODE=disabled`。

## Judge V2 奖励与逐轮 GRPO

`harness_awm.reward_policy` 实现 `judge-v2-terminal-discount-v1`：有效且已准入的 `confirmed_pass/pass` 得 1，复核后仍有真实不确定性的 `provisional_pass/unknown` 得 0.3，`failed/fail` 得 0。评测故障、预算耗尽、必要取证未完成、无效样本、不可信终态、测试结果或未准入配置仍得到 null。quality 不参与奖励。独立 required 反证可以决定失败，不被其他条目的 unknown 抵消。

0.3 固定在 `harness_awm.reward_values`，不是正确概率，也不是梯度权重。暂定通过仍保留 unknown 标签，不计入确认成功率。奖励策略已升级版本：旧 `judge-v2-terminal-binary-v1` 或缺少 `policy_version` 的准入记录不会自动升级，需要按新策略和当前 Judge 配置重新确认版本绑定；历史结果不覆盖。

Judge 缓存继续保存纯判分。`evaluate_episode` 在缓存返回后重新检查当前准入，生成独立 `rewards_v2` 记录并应用到 episode；撤销准入不会因旧缓存而继续产生可用奖励。默认没有准入文件，默认仍不发放奖励。本次实现不自动批准任何实际任务。

RunConfig 新增 `reward_admissions`，指向如下结构的 JSON（占位符不能直接使用）：

```json
{
  "schema_version": "awm-reward-admissions-v1",
  "tasks": {
    "任务实例ID": {
      "status": "admitted",
      "calibration_ref": "独立正反例校准记录的位置或版本",
      "policy_version": "judge-v2-terminal-discount-v1",
      "task_instance_id": "任务实例ID",
      "original_hash": "digest(original)",
      "rubric_hash": "digest(rubric)",
      "initial_state_hash": "TaskSpec.initial_state_hash",
      "environment_code_hash": "TaskSpec.environment_code_hash",
      "tool_schema_hash": "TaskSpec.tool_schema_hash",
      "judge_config_hash": "judge_runtime.config_hash(config)"
    }
  }
}
```

准入是操作方完成独立校准后的声明，程序只校验版本绑定，不把文件存在当成校准实验已经做过。撤销时把 `status` 改为 `revoked`。Judge 实现或配置变动后需重新确认准入。运行时另外检查工具目录和 strict 隔离记录。

`TrainingActorBackend` 使用当前本地模型采样，保存每轮实际 prompt IDs、completion IDs、采样 logprob、策略版本及 action mask 到封存的 `model_turns/*/policy_turn.json`。采样固定温度 1、无 top-k/top-p 截断或额外 logits penalty，确保采样分布与 loss 的策略概率一致。上下文压缩后仍使用当轮保存的 prompt，不能用完整历史替代。

`harness_awm.training.grpo.GRPOEngine` 接通独立候选环境 → Judge V2 → 奖励 → 总体标准差优势 → clipped GRPO loss → optimizer step。每组共享任务、初态、策略版本和预算，各候选有独立目录及数据库。接受 1/0.3/0，任一 null 暂缓整组，不补采、不把 null 转成零；全组奖励相同（包括全为 0.3）且 beta=0 时不执行 optimizer.step，避免 AdamW 权重衰减造成无信号更新。loss 先在每条轨迹全部生成 token 上平均，再对 K 条轨迹平均；环境、提示和 padding token 不参与 loss。每组只更新一次，拒绝复用旧策略组。beta>0 时要求独立冻结的 reference model。

此实现采用自有逐轮 loss，避免直接套用 TRL 默认样本标准差及单 completion 上下文假设。**目前支持单进程；SFT 的 FSDP2 配置不能直接用于该 GRPO 入口。** CUDA 大模型、分布式 GRPO 和实际环境训练效果仍需另行验收。通用 `training.smoke` 是 TRL 框架检查，新链路由 `tests/test_training_grpo.py` 离线验证。

只做准入与配置检查，不加载模型、不调用 Judge、不启动训练：

```bash
.venv-training/bin/python -m harness_awm.training.grpo \
  --config /path/to/grpo_run_config.json \
  --tasks /path/to/grpo_train.jsonl --group-size 4
```

配置需要 `actor="training"`、`judge_protocol="rubric_v2"`、真实 Judge、strict 隔离、源材料 manifest、奖励准入文件及明确的正数请求/token 上限；只接受 train split。该命令没有启动训练的副作用。

现在可在同一命令加 `--execute --model ... --output ... --groups ...` 正式启动，或调用异步 `training.grpo.train(...)` / `GRPOEngine.collect/update`。`groups` 限制采集组数量，包含 held 和零优势组，不等于 optimizer 更新次数。Judge 共享累计预算；输出保留 checkpoint 文件哈希、任务包版本、启动参数、每组证据和更新报告、最终模型及 optimizer 状态。当前没有断点续训入口，应从新的输出目录开始。完整用法见下节。

离线验证命令（随机小模型、内存环境和模拟 Judge transport，不访问付费服务）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-training/bin/python -m pytest -q \
  tests/test_training_grpo.py tests/test_training_runtime.py \
  tests/test_judge_v2.py tests/test_harness_unit.py
```

测试需要本机 socket 通信：Judge 通过 `asyncio.to_thread` 执行只读查询，工作线程完成后靠事件循环的 socket 唤醒主线程。若沙箱禁止 `socketpair().send()`，会出现查询已经完成、测试却卡在事件循环等待的现象；开启 asyncio debug 可见 `Fail to write a null byte into the self-pipe socket` / `PermissionError`。应在允许本机 socket 的测试环境重跑，不应删除异步取证或跳过这些测试。排查时可加 `-o faulthandler_timeout=20` 和外层 `timeout 120s`。这些离线测试不需要外部模型 API。

正式 reward 路径与 test-only 诊断路径分别测试。test-only episode 始终保留 `reward=null`；诊断更新必须显式提供独立测试信号，报告标记 `test_only/diagnostic_rewards`，不能导出为正式奖励。本次未启动正式训练，未调用付费 Actor/Judge。

## 从已划分数据启动 GRPO 和本地 checkpoint 评测

### 运行任务包：绑定资产，不重新划分

`training_split_20261106_v1` 记录题目属于哪个 split；`training.tasks` 将它们绑定为执行器接受的 `TaskSpec`，复制已筛查的冻结初始数据库、环境代码、工具目录和 Judge 原题/rubric。相同世界去重存储，每次 rollout 仍获得独立数据库副本。不会重新初始化数据库、改变题目/划分、加入历史答案，或签发奖励准入。

```bash
.venv-training/bin/python -m harness_awm.training.tasks \
  --split data/training_split_20261106_v1 \
  --output data/training_runtime_20261106_v1
```

本地已生成该运行包：GRPO 2,746 / validation 589 / test 588，980 个冻结世界。`application_management_4:0` 的原筛查目录缺少 MCP 目录，转换器使用同一冻结数据库和环境代码的 `application_management_4:1` 公共目录；版本匹配和来源记录在包 manifest 中。所有 split 的任务 ID 保持互斥；同一环境跨 split 是本次明确的评测范围，不套用旧 registry 的“世界必须隔离”规则。

包可整体搬迁，内部运行路径在加载时解析，来源哈希会重新验证。GRPO/本地评测的 `--tasks` 接受包目录，也兼容已有绝对路径的 `TaskSpec` JSONL；原始划分清单不能直接传入。新包使用 Judge V2，无需旧官方 SQL verifier；若误用 official Judge，会明确拒绝缺少 verifier 的任务。数据包包含 Judge 私有材料，执行器只向 Actor 提供题目、公开工具和运行观察。

### GRPO 启动

以 `configs/training/grpo-local.example.json` 为模板准备运行配置。示例中的奖励准入路径是待填写路径；请求/token 上限是示例上限，不代表足以完成全量训练。配置中的 `purpose=development` 不会绕过训练奖励准入：每个训练任务仍必须通过当前策略版本的独立校准绑定。配置和 `.env` 中的 Judge/API 信息沿用已有机制。

```bash
# 默认只预检，不加载模型或调用 Judge。
.venv-training/bin/python -m harness_awm.training.grpo \
  --config configs/training/grpo-local.example.json \
  --tasks data/training_runtime_20261106_v1 --group-size 4

# 到正式环境、完成奖励准入后显式执行。
.venv-training/bin/python -m harness_awm.training.grpo \
  --config configs/training/grpo-local.example.json \
  --tasks data/training_runtime_20261106_v1 \
  --model runs/qwen35_sft_full_v1/final \
  --output runs/qwen35_grpo_v1 --groups 100 --group-size 4 --execute
```

可重复传 `--task scenario:index` 选择已完成准入的小批任务。默认 CUDA、FP32 参数、AdamW 学习率 `1e-6`、beta=0、种子 20261106；`--dtype bfloat16` 是整个模型/优化器参数 dtype，不是自动混合精度。`--beta` 为正时必须传 `--reference-model`，该独立模型冻结。`--device cpu` 仅用于显式 CPU 执行。仍是单进程逐个候选，不支持 `torchrun/accelerate` 多进程；目标 GPU 的显存、数值一致性和吞吐留待实测。

### 本地 checkpoint 验证和最终测试

`harness_awm.training.evaluate` 直接加载 Base/SFT/GRPO checkpoint，复用 GRPO 的 `EpisodeRunner`、MCP 环境和 Judge V2。使用独立 `purpose=evaluation`：真实 Judge 判分、严格环境隔离，不需要训练奖励准入，不创建优化器，不输出训练奖励。只能选择 validation 或 test；每题一次 rollout，默认 greedy。`--sampling policy` 使用与 GRPO 相同的温度 1 原始策略采样。所有 checkpoint 比较必须固定同一种采样方式、任务范围、预算和 Judge 配置。

```bash
# 预检模型文件、验证集及来源绑定；不加载权重、不调用 Judge。
.venv-training/bin/python -m harness_awm.training.evaluate \
  --config configs/training/evaluate-local.example.json \
  --tasks data/training_runtime_20261106_v1 --split validation \
  --model runs/qwen35_sft_full_v1/final

# 在正式环境执行验证集。
.venv-training/bin/python -m harness_awm.training.evaluate \
  --config configs/training/evaluate-local.example.json \
  --tasks data/training_runtime_20261106_v1 --split validation \
  --model runs/qwen35_sft_full_v1/final \
  --output runs/qwen35_sft_validation_v1 --execute

# 通过验证集确定 checkpoint 后，使用选定模型执行测试集。
.venv-training/bin/python -m harness_awm.training.evaluate \
  --config configs/training/evaluate-local.example.json \
  --tasks data/training_runtime_20261106_v1 --split test \
  --model runs/qwen35_grpo_v1/final \
  --output runs/qwen35_grpo_test_v1 --execute
```

评测按题顺序执行，配置中的 worker 数不会让同一本地模型并发生成。可用 `--task` 做小批预检，但其指标只代表选中的子集。示例请求/token 上限需按实际评测预算调整；耗尽产生评测故障，不算业务确认失败，也不将未完成评测从总题数中删除。

`metrics.json` 同时报告确认通过/计划题数、确认通过/有效判分数、暂定通过/失败/故障计数和终止原因；`manifest.json` 冻结 checkpoint 哈希、split、任务、采样方式和 Judge 配置。选择 checkpoint 时仅比较相同验证集协议的结果，检查评测覆盖率，避免把更低覆盖率的“有效样本成功率”当成提升。当前提供逐 checkpoint 评测命令，选模仍由操作方根据验证报告完成；测试集不用于调参或选模。

Judge V2 会结合执行轨迹、最终答复、初末数据库证据判断 required/quality 条目。它不是已实现的逐步过程奖励模型：`ProcessEvaluator` 仍未实现，quality 不参与终局训练奖励。0.3 的暂定通过单列，不计入确认成功率。这里衡量的是 Judge 判定的任务成功率；衡量 Judge 自身准确率需要独立人工真值。

参考：[TRL SFT](https://huggingface.co/docs/trl/sft_trainer)、[TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer)、[Qwen3.5-2B 官方模型](https://huggingface.co/Qwen/Qwen3.5-2B)、[PyTorch Blackwell/CUDA 12.8 支持](https://pytorch.org/blog/pytorch-2-7/)、[NVIDIA 二进制架构兼容规则](https://docs.nvidia.com/cuda/ada-compatibility-guide/)。
