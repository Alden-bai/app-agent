# 使用 verl 运行 AWM 的 SFT 和多轮 GRPO

## 2026-11-13：全量 2746 道训练题

本次按用户明确授权使用全部 `grpo_train`（2746 题、980 个场景），不再把小规模控制题当作训练白名单。
`awm-reward-admissions-v3` 明确区分共享 Judge 控制校准和逐题来源核验：复用 6 道真实通过的控制，
绑定每题的原题、rubric、初始状态、环境、工具及 Judge 配置哈希；`per_task_calibrated=false`。
这不是 2746 题分别完成了正负控制校准。原 v1/v2 逐题准入保持支持，隔离和证据封存检查继续执行。

`prepare --shuffle-tasks` 按配置 seed 冻结全量任务顺序。全部题目进入任务池，目标仍为 800 次有效更新、
最多 3200 组，达到更新目标即停止，因此不保证 2746 题每题都实际采样一遍。采样/KL/clip/null 规则沿用
`single-gpu-800`。入口为 `scripts/run_full_grpo.py`，先调用 `scripts/prepare_full_grpo_admissions.py`，
再执行 prepare/check/train 和 checkpoint 独立验收。运行记录位于 `runs/grpo_full_2746_20261113_v1`；
checkpoint 写挂载盘，完成且逐字节校验归档的旧运行缓存可回收，只保留最近两个活跃组。

## 2026-11-12：单卡 4 候选、800 次更新预设

`configs/training/grpo-single-gpu-800.json` 是本轮训练设置，使用 `prepare --preset single-gpu-800` 加载。
temperature=1、top-p=1；KL reward/loss 均关闭；PPO clip 下限和上限均为 0.2（概率比率区间 0.8–1.2）。
学习率为 1e-6、warmup=0；一次更新使用一个任务的 4 条候选轨迹，最多 4 个环境并发。
训练微批为 1 个实际对话轮次；累积整组全部轮次的梯度后仅执行一次 optimizer step。
因此 prompt batch=1、轨迹 batch=4，不是每条轨迹各更新一次，也不是 4 个任务。

目标是 800 次有效参数更新；全同分组（包括全 0、全 1、其他相同分数）不更新、不增加步数。
每 200 次更新保存 checkpoint，即 200/400/600/800；正常提前结束时另存最终状态。
最多采样 3200 组，耗尽时明确报告目标未完成；可用 `--groups` 修改这一采样上限。
采样组数与有效更新数分别记录。800 次更新是训练预算，不是当天完成的时间承诺。

本预设显式启用 `null-as-zero-v1` 训练奖励规则：最终 reward=null 的候选训练奖励为 0，
与明确失败相同；例如 `[1,null,0,1]` 使用 `[1,0,0,1]` 计算优势。四条都为 null 则全为 0，跳过。
原始 Judge 评分、有效性、错误和封存证据保持原值，映射原因记录到 `reward_selection.json` 的
`null_penalties`，训练指标记录 `group/null_penalties`。不会将原始评测 null 改写为有效业务失败。
只有具有已验证的实际采样 token、logprob 和上下文的轨迹才能产生梯度；生成任何 token 前失败的
候选没有可优化动作。用户取消、证据篡改、失效任务准入不会被此规则转成可训练样本。
默认不选此预设时仍采用原有排除 null 的语义。

同时启用 `recover_tool_errors`：工具返回的错误或抛出的普通异常进入模型下一轮上下文，
模型可检查当前状态、纠正参数或换用工具，在原有轮数/调用预算内尝试修复。
写入超时可能已经提交，反馈会明确说明，harness 不自动重放写入；取消信号仍终止执行。
Judge 在轨迹结束后运行，其最终 null 使用上述 0 奖励，不重新打开已结束的环境。

在训练机器上，用实际配置（包括奖励准入）、数据和模型目录准备一个新运行：

```bash
python -m harness_awm.training.verl prepare \
  --preset single-gpu-800 \
  --config /absolute/path/to/training-config.json \
  --tasks /root/awm-agent-train/data/training_runtime_20261106_v1 \
  --model /absolute/path/to/selected-sft-checkpoint \
  --output /absolute/path/to/new-grpo-run \
  --checkpoint-dir /absolute/path/to/checkpoints
python -m harness_awm.training.verl check --run /absolute/path/to/new-grpo-run
python -m harness_awm.training.verl train --run /absolute/path/to/new-grpo-run
```

使用已安装 verl 本机依赖的 Python；`prepare` 只生成计划，不调用 Judge 或启动训练。
上述命令没有指定显存比例，沿用入口的保守默认值 0.35；GPU 资源参数需结合实际训练机器设置。
本次仅落地配置和离线测试，当前主机会话无可见 GPU，未启动 800 步训练。
离线回归结果：196 项测试、22 项子测试通过。包含 null→0 与证据保留、工具异常反馈和恢复、
固定采样/KL/clip/batch 配置，以及使用模拟 rollout/optimizer 的 1200 组控制循环：
剔除同分和无可训练动作的组后恰好更新 800 次，仅在 200/400/600/800 保存。
另验证采样上限耗尽时报告目标未完成；这些测试没有运行真实 GPU 训练或调用真实 Judge。

## 多步骤任务试跑与无限累计预算

`RunConfig.episode_token_limit=null` 关闭单条轨迹累计 token 截止；
`max_physical_requests=null` 与 `max_total_tokens=null` 同时关闭 Judge 累计请求/token 预算。
这些设置不改变单次模型上下文容量。轮次、工具次数、超时与重复动作保护仍独立生效。

`scripts/calibrate_verl_grpo.py` 支持重复 `--task` 与 `--controls`，只从 `grpo_train`
选题。控制文件指定真实工具动作、最终交付和独立 SQL 断言；新增对象须核对初末状态差异。
每题需要两条正确执行、部分完成、未执行，以及正常/冗余轨迹质量校准全部通过，才能签发准入。
配置或 Judge 实现哈希变化时必须重新校准，不直接搬用旧准入。

`prepare --rollout-correction token-is-v1` 显式启用 verl 的 token 级重要性采样修正。
保留真实采样 `old_log_probs`，另存冻结训练端的 `learner_old_log_probs`，用后者作为 PPO
分母，再乘 detached 的 `exp(learner_old - sampled_old)`。该模式是有偏、低方差的 token
修正，不是精确的整轨迹重要性采样。概率比超出 `[0.5, 2]` 或出现非有限值仍中止；
逐组报告原始差异及修正权重。默认 `strict` 继续执行原来的严格 allclose 检查。

`scripts/run_expanded_grpo.py` 将通过校准的多步骤任务组成固定池，未通过的题保留证据并排除。
运行目录保存 `status.json`、`gpu.jsonl`、逐任务校准日志，以及训练目录中的 `progress.json`
和逐组 `update.json`。预算无限不等于无限训练：采样组数与有效更新目标由启动参数分别指定。


新本机运行配置已选择 `judge-v2-cst-v1`：`R=C×[1−0.05(1−S)−0.15(1−T)]`，
复核后的真实 unknown 使用 `C=0.3`。完整定义见 [C/S/T 奖励](judge_v2_zh.md#cst-奖励2026-11-10)。
第二个 Judge 使用封存的 Actor 可见上下文评价轨迹；新运行必须重新执行校准和准备计划。
下文 96 个候选全为 1 的历史结果来自旧终局奖励，不包含 S/T，也不代表 CST 已完成真实训练验证。

## C/S/T 接入与本次短跑

`runs/cst_reward_20261110_v1/calibration-v5` 完成三个任务的业务与轨迹质量控制校准。
正常成功控制的诊断合成分为 1；冗余成功控制分别为 0.95、0.95、0.975。控制保持 test_only。
前四次校准的停止、漏读证据和引用错误记录保留；最终实现使用逐轮动作前观察、短证据编号、
短原文引用、明确动作时间线以及每阶段一次格式修正。只补评无效输出，不重评已有效分数。

真实短跑 `runs/cst_reward_20261110_v1/training-001` 使用完整 CST、每组 4 候选、3 组，
按用户要求 Judge 请求数和 token 都为 null（不限）。12 个候选均为 C=1、S=1、T=1，
没有软项（S 默认 1）且没有确认的过程缺陷，因此全部 R=1、0 次有效更新，首次更新目标未达到。
这不能算模型提升；后续应扩大任务池，不能给正确轨迹强造扣分。

为减少网络盘加载等待，将选定 SFT 文件复制到 `/dev/shm/awm-cst-sft-checkpoint-210-20261110`，
逐文件核对选模记录的 SHA-256。计划在完整哈希前后核对文件集合、inode、size、mtime、ctime；
启动时元数据完全不变才复用该次哈希，变化或旧计划仍做完整哈希。相关修改覆盖 192 项离线检查。
本次 GPU 观察到 87,754 MiB（约 85.7 GiB）占用。结果与用量见运行根目录 `report.json`。

## 本机 Landlock 开发试跑

本机新增显式 `local-sdpa` 配置，用于 Qwen3.5 文本 SFT checkpoint 的单卡 GRPO。
正式 `strict` 隔离要求保持原样；Landlock 仅用于 `purpose=development`，并要求
重新校准、绑定隔离和证据哈希的 `awm-reward-admissions-v2`。旧 v1 准入不能用于 Landlock。

2026-11-10 本次试跑已经执行完 24 组（96 个真实候选）。所有奖励均为 1，24 组优势均为 0，
优化器按规则没有更新，因此 **3 次有效更新验收未完成**。12 条 Judge 校准控制、GPU 探针、
完整状态保存和独立 HF 重载生成均通过；重载输出为 `{"ok":true}`，参数样本哈希与保存前一致。
按用户追加要求，最后三组采用 vLLM 比例 0.78，实测 GPU 峰值 87,828 MiB（85.77 GiB）。
完整记录在 `runs/verl_grpo_smoke_20261110_v1/report.json`，最终权重在
`/autodl-fs/data/awm-verl-grpo-smoke-20261110-v1/training-002/global_step_0/actor/huggingface`。
这是 step 0 的保存检查，不是训练提升。试跑结束后工作区新增的奖励逻辑不属于本次验证范围；
本次代码按计划哈希保存在各运行目录的 `source-snapshot`，后续运行需重新准备计划和校准准入。

独立环境位于 `/root/autodl-tmp/awm-grpo-runtime-20261110/venv`。安装入口：

```bash
PIP_INDEX_URL=https://pypi.org/simple python scripts/setup_verl_grpo_local.py \
  --venv /root/autodl-tmp/awm-grpo-runtime-20261110/venv \
  --source /tmp/awm-verl-v080 \
  --work /root/autodl-tmp/awm-grpo-runtime-20261110/repair
```

`--source` 是原固定提交的完整 verl 源码目录。安装器检查原训练接口源码哈希，
在独立构建目录仅修改 NumPy 依赖声明和包版本，产出 `0.8.0.dev0+awm.numpy2` wheel。
它解决原 verl 的 NumPy 1.x 约束与 vLLM/OpenCV 的 NumPy 2.x 依赖冲突，
保留 torch 2.11 / CUDA 13 / vLLM 0.20.2。构建补丁、wheel 哈希和最终依赖清单
保存在 `--work`，安装后必须通过 `pip check`。

`requirements/verl-local-sdpa-runtime.json` 固定本机配置的版本和上游源码哈希。
训练使用 SDPA、上游等价 padding/indexing 和 Liger FP32 累加 CE；不安装 SFT 的
微批或 loss 补丁，也不要求独立 `flash-attn`。CUDA 13 nvcc/CRT/NVVM 随独立环境安装，
避免 Blackwell 错用系统 CUDA 12.8。纯文本 vLLM 插件补齐混合缓存及 M-RoPE 文本位置协议。
准备计划同时读取 tokenizer.json 中的特殊 token，当前选定模型的停止集合为 `[248044,248046]`。

一键执行新的完整试跑（输出目录必须不存在，checkpoint 放有足够容量的数据盘）：

```bash
/root/autodl-tmp/awm-grpo-runtime-20261110/venv/bin/python scripts/run_verl_grpo_smoke.py \
  --output runs/verl_grpo_smoke_fresh \
  --checkpoint-dir /autodl-fs/data/awm-verl-grpo-smoke-fresh
```

入口先核对选模记录和 checkpoint 身份，再执行 GPU 无更新探针、12 条独立控制的真实
Judge 校准、prepare/check/train 和独立重载验收。默认使用已选定的 lr1e-5-epoch2，
3 个固定训练任务，每组 4 个候选、2 个并发环境、最多 24 组，达到 3 次有效更新立即停止。
按当前 96 GB 主机的约 85 GiB 总显存目标，一键入口默认 `--gpu-memory-utilization 0.78`；
这是 vLLM 的显存比例，另需给训练模型和 CUDA 上下文留空间。实际占用应以 GPU 监控为准，
进入训练阶段时 vLLM 会释放缓存，显存占用会随阶段变化。
原试跑采用排除 null 的规则：有效样本同分或不足两条时不计作更新，其余有效样本按实际数量计算优势和损失。2026-11-12 的 `single-gpu-800` 预设改用上文的 null→0 规则。未达到更新目标不能通过最终验收。

Judge 配置在 `configs/training/grpo-landlock-local.json`：累计请求/token 上限均为 null，
按用户指定的 5000 RPM / 200 万 TPM 限流，所有实际请求、429 重试和未知用量继续记账。
校准和训练的 API 凭据从现有 `.env` 读取。SwanLab 沿用已有配置；`SWANLAB_MODE=disabled`
可关闭远程日志。SQLite、活跃证据和日志留本地，完整 checkpoint 与 HF 导出写数据盘。

也可分阶段使用 `scripts/calibrate_verl_grpo.py` 和原 `python -m harness_awm.training.verl`
入口。`prepare` 新增 `--runtime-profile local-sdpa`、`--target-updates 3`、`--checkpoint-dir`；
`probe --run ...` 只验证真实 GPU 加载、权重同步、EOS 和采样 logprob 一致性，不调用 Judge，
不产生参数更新。`scripts/verify_verl_grpo_checkpoint.py --run ...` 核对真实优化器步数、
参数变化及完整训练状态，并比较 HF 重载权重和训练端固定参数样本哈希，然后独立生成。
调整已冻结的运行配置时使用新目录。`prepare --group-offset N` 可保留原采样组编号和任务轮转，
它不恢复优化器状态；仅在确认此前没有参数更新时才可用原 SFT 权重接续采样。

本地保留 `progress.json/jsonl`、各组 `update.json`、预算账本、`summary.json` 和
`acceptance.json`。训练 loss/梯度、logprob 偏差和参数变化是闭环证据，不是效果提升结论；
本流程不使用验证或测试任务训练。

SFT 新入口为 `python -m harness_awm.training.verl.sft`，GRPO 新入口为 `python -m harness_awm.training.verl`。AWM 继续负责冻结数据、任务、独立环境、上下文压缩、Judge V2 和证据；verl 负责 SFT/FSDP2 参数更新、vLLM 连续批处理推理、训练与推理之间的权重同步及 checkpoint。

这是基于 verl 的自定义逐轮 GRPO recipe，不能直接用上游 `main_ppo` 加数学题 reward 函数替代。普通“一条对话一条训练序列”的表示不能复现本项目压缩后的实际上下文。

## 版本与支持范围

- 固定 verl v0.8.0 标签对应提交 `7aed6b230776f963fa09509c10d9c3a767d1102c`；该提交的包版本字符串实际为 `0.8.0.dev0`。
- 独立 `.venv-verl`：Python 3.12、PyTorch 2.11.0、Transformers 5.22.1、vLLM 0.20.2、Ray 2.55.1、TensorDict 0.10.0。关键接口源码哈希保存在 `requirements/verl-runtime.json`。
- SFT 接受原始 Qwen3.5 基模目录或纯文本 checkpoint，另存文本配置并只加载全部文本参数；GRPO 接受 SFT 导出的纯文本 `Qwen3_5ForCausalLM` checkpoint，其 `model_type` 必须为 `qwen3_5_text`。
- 全参数文本训练，FP32 参数与优化器状态、BF16 计算、FSDP2、激活检查点；不启用 LoRA、量化或跨样本 attention packing。
- 本版支持单机，`group_size` 必须可被 GPU 数整除。默认单卡、每组 4 个候选、最多 4 个并发环境，一次有效组对应一次 optimizer update。
- 本版固定 `beta=0`、每组仅一个 PPO epoch、温度 1、无 top-k/top-p 截断。暂不提供续训入口、KL reference、跨节点运行或完全异步旧策略训练。
- SFT 和 GRPO 均有 verl 入口；原 TRL SFT 和单进程 GRPO 保留作语义参照。验证/测试仍走现有评测入口，不在训练中使用测试集。

vLLM 0.20.2 包含 Qwen3.5 的文本解码器，但没有注册 SFT 的纯文本架构名称。本项目通过 `vllm.general_plugins` 显式注册已有解码器，并复用上游混合注意力缓存接口。插件仅在 `AWM_VERL_TEXT_MODEL=1` 时生效；训练入口负责设置该变量。未加载视觉塔，也不重写 checkpoint 权重名。

上游依据：[verl 固定源码](https://github.com/verl-project/verl/tree/7aed6b230776f963fa09509c10d9c3a767d1102c)、[vLLM 0.20.2 Qwen3.5](https://github.com/vllm-project/vllm/blob/v0.20.2/vllm/model_executor/models/qwen3_5.py)。

## 本机已验证的 SFT CUDA 环境（2026-11-09）

SFT 单独使用 `.venv-verl-sft`，复用本机已有的 CUDA 依赖目录，避免系统盘重复安装。
`python scripts/setup_verl_sft_local.py` 可重建该轻量环境。依赖版本和上游源码校验见
`requirements/verl-sft-runtime.json`：PyTorch 2.10.0+cu128、verl 0.8.0.dev0、
Transformers 5.22.1、FLA 0.6.0、Liger 0.8.5。这里不需要 vLLM；GRPO 的独立依赖锁保持原样。
这些被复用的依赖目录在训练期间必须保留。

SFT 在 SDPA 路径使用 verl 自带的纯 PyTorch padding/indexing 实现，
跳过不适用于原生 SFT 的旧 TRL ValueHead 补丁。另用 FP32 累加的 Liger CE
替换未安装 FlashAttention 时逐 token 的 BF16 log-softmax；不启用 Liger RMSNorm、
SwiGLU 或 fused linear head，不改变响应 mask、token 归一化、样本或 attention。
真实 248320 词表维度下，CE 与 FP32 参考最大差约 9.54e-7；
BF16 输入梯度相对差约 9.72e-4。完整结果在 `runs/verl_sft_calibration_20261109/ce_validation.json`。

`prepare --max-tokens-per-gpu N` 启用单卡严格动态微批，按每批行数乘最大长度限制 padded tokens；
上游按估计计算量平衡的拆批不保证该上限，可能导致 OOM。默认 0 保留原生单样本微批。
它只改变每次前后向的拆批，不改变有效 batch；预算小于真实最长样本时拒绝启动。
每个完成更新追加 `progress.jsonl` 并更新 `progress.json`，记录 loss、梯度范数、
实际 tokens/s、步骤耗时和 PyTorch 峰值显存；同步传给原生日志和 SwanLab。
显存目标约 85 GiB，随长度、优化器初始化和 allocator 缓存变化，不是固定占用承诺。

### 本次正式运行（已按用户要求停止）

第 7 步后停止，转回旧四个 checkpoint 的验证选模，详见 `docs/sft_selection_zh.md`。

运行目录：`runs/qwen35_verl_sft_20261109_v4`。原始 Qwen3.5-2B 的全部文本参数，
6,714 条冻结样本，学习率 1e-5，有效 batch 64，2 epoch / 210 optimizer steps。
微批最多 20,000 padded tokens；`--gpu-memory-limit-gib 84` 限制 PyTorch allocator，
连同 CUDA 上下文目标约 85 GiB。显存仍随样本和缓存波动。

后台入口为 `scripts/run_verl_sft_monitored.py`；`train.log`、`progress.json`、
`progress.jsonl`、`status.json` 和 `gpu.jsonl` 保留本地。
每轮保存完整模型、优化器、调度/RNG 状态和 HF 导出，checkpoint 目录链接到
`/autodl-fs/data/awm-verl-sft-20261109-v4/checkpoints`，预估两份约 56 GiB。
训练期间不要重复启动；v1/v2 为准备记录，v3 在 torchrun 参数解析阶段退出，没有训练更新。

存储盘曾耗尽 200,000 个文件名额。已将旧 EOS 修复评测中基模前 10 个完成的 episode
目录迁至本地盘，逐文件 SHA256 和全部 evidence seal 复核通过，原路径保留目录链接。
迁移审计在 `runs/verl_sft_calibration_20261109/storage_migration.json`。
三步零学习率 GPU 校准处理 1,593,781 tokens，约 193.6 秒，平均约 8,231 tokens/s；
这是校准吞吐，不是相对旧 TRL 的受控加速结论。报告为该目录的 `summary.json`。

## 安装（完整 GRPO 环境）

在提供 CUDA GPU、足够新驱动和 CUDA 编译工具的机器上执行：

```bash
bash scripts/setup_verl.sh
```

该脚本只安装到 `.venv-verl`，不更新原 `.venv-training`。需要至少 15 GiB 可用空间安装 CUDA 依赖和构建内核，另外预留训练 checkpoint 空间。FlashAttention 在 PyTorch 安装后单独构建；即使训练使用 SDPA，所固定的 verl 版本仍引用其 padding helper。可用 `VERL_PYTHON` 指定 Python 3.12，`MAX_JOBS` 控制编译并行度。

`requirements/verl.in` 固定主要运行组件，尚不是所有传递依赖的离线 wheel 锁。安装完成会运行 `pip check` 和核心版本/源码哈希检查。

真实 MCP 环境还需要 `RunConfig.runtime_python` 指向可执行、已安装 AWM 依赖的解释器，以及允许 bubblewrap 用户命名空间的宿主。只有 GPU 可见不足以证明环境能执行。

## SFT：准备、检查、启动

直接使用已有 `train.jsonl` 和 `report.json`，不重新套聊天模板，也不将整段历史变成监督目标：

```bash
.venv-verl/bin/python -m harness_awm.training.verl.sft prepare \
  --model models/Qwen3.5-2B \
  --data data/training_qwen35_2b_v1 \
  --output runs/verl_sft_pilot \
  --global-batch-size 32 --gpus 1 --epochs 1 --max-steps 3

.venv-verl/bin/python -m harness_awm.training.verl.sft check --run runs/verl_sft_pilot
.venv-verl/bin/python -m harness_awm.training.verl.sft train --run runs/verl_sft_pilot
```

`prepare` 校验数据与 tokenizer 哈希、所有序列长度和标签，并生成只包含文本模型的配置；不加载模型参数或开始训练。`train` 使用 torchrun 启动 verl 原生 SFTTrainer/FSDP2。正式训练去掉 `--max-steps 3`，使用新输出目录；`--global-batch-size` 是所有卡和累积微批合计的 batch，必须可被 GPU 数整除。

监督位置严格来自现有 `labels != -100`；loss 按全局 optimizer batch 的监督 token 总数归一化。每轮数据排序按长度分组；一个 epoch 内每个真实样本恰好一次。最后不足整批时使用 loss mask 全为零的占位行补齐，既不静默丢弃数据，也不重复真实样本。最长样本超过 `--max-length` 时明确报错。

checkpoint 位于 `checkpoints/global_step_N/`，同时保存 HF 文本模型导出，可交给 GRPO。原始模型不被覆写。本版 SFT 不提供断点恢复命令。SwanLab 在设置密钥或 `SWANLAB_MODE=offline` 时由 verl 主 rank 记录；`SWANLAB_MODE=disabled` 关闭。

SFT 禁用模型层 fused-kernel 和跨样本 attention packing；本机 SFT 的 CE 兼容路径见上文。相对原 TRL fused LM head 的实际显存/吞吐变化需实测，迁移框架本身不保证加速。

## GRPO：准备、检查、启动

沿用 `configs/training/grpo-local.example.json` 的运行配置。必须填写真实的 `reward_admissions`，并保持其与当前 Judge 配置、任务和奖励策略版本一致。程序不会生成或自动签发奖励准入。

先准备一组试跑计划；把 `MODEL` 替换为已训练 checkpoint 的实际路径：

```bash
MODEL=/absolute/path/to/sft/checkpoint
.venv-verl/bin/python -m harness_awm.training.verl prepare \
  --config configs/training/grpo-local.example.json \
  --tasks data/training_runtime_20261106_v1 \
  --model "$MODEL" \
  --output runs/verl_grpo_pilot \
  --groups 1 --group-size 4 --concurrency 4 --gpus 1
```

`prepare` 只读取资产并写运行计划，不加载权重到 GPU，不调用 Judge。即使缺少准入或环境尚未就绪，也可以生成计划并输出 `ready=false` 和原因。可重复 `--task 精确任务ID` 选择训练集试跑子集。

```bash
.venv-verl/bin/python -m harness_awm.training.verl check --run runs/verl_grpo_pilot
```

`check` 验证模型和代码哈希、奖励准入、解释器、核心依赖、CUDA 和 bubblewrap。没有全部通过时返回非零退出码。`--skip-runtime` 只跳过运行环境检查，不表示可以训练。Judge 密钥仅从现有 `.env` / 环境变量读取，不写入计划。

检查通过后启动真实采集、Judge 请求与训练：

```bash
.venv-verl/bin/python -m harness_awm.training.verl train --run runs/verl_grpo_pilot
```

`train` 会消耗运行配置中的 Judge 请求/token 预算。每个计划仅可启动一次；发生失败后保留证据，重新准备新目录。修改代码、模型或计划后也需重新 prepare。

先完成试跑和独立效果检查，再增加 `--groups`。`groups` 包含 held 和零优势组，不等于优化器更新次数。`--concurrency` 限制独立环境数；Judge 并发仍受原配置的 `judge_concurrency` 限制。一次运行共享同一个 SQLite 累计预算。

## 保留的训练含义

1. 同组所有候选共享同一策略版本、任务和初态，各有独立数据库与目录。候选并发运行，全部完成后才允许参数更新。
2. 保存每轮真正送入 vLLM 的 prompt IDs、completion IDs、原始采样 logprob、停止 token、策略版本和 action mask；压缩前后不拼接成虚构的完整上下文。
3. 重新验证封存轨迹和实时奖励准入。旧终局策略使用 1 / 0.3 / 0；CST 策略接受按有效 C/S/T 分量重算一致的连续奖励。同组不能混用有效奖励策略。默认排除单条 null；显式 `--null-reward-policy zero`（`single-gpu-800` 已启用）将具有可验证动作记录的最终 null 映射为训练奖励 0，保留原评测结果。可训练候选不足两条、准入撤销或变更时 held；证据或策略身份被篡改仍拒绝训练。
4. 使用总体标准差计算优势。每条轨迹内部按全部生成 token 平均，再对 K 条轨迹平均。展开为逐轮样本后的系数为 `B / (K * T_i)`，B 是含 DP 对齐行的批次数量，T_i 是该轨迹的生成 token 数；padding 行的 loss mask 完全为零。
5. 每组全部轮次属于同一个全局 mini-batch，只执行一次 optimizer update。全组奖励相同则跳过更新，避免 AdamW 权重衰减造成无信号更新。
6. 更新前在训练侧重算 logprob，并按原有 `atol=5e-3, rtol=1e-3` 检查推理/训练一致性。出现偏差会中止，不用重算值静默覆盖实际采样概率。vLLM 与 Transformers 的内核差异可能触发此检查，GPU 试跑时需实测。

CST 训练中，主评分已经不满足现有终局奖励有效性条件时，跳过无法改变结算结果的轨迹质量评分，记录 `invalid_primary_assessment`。完整准入、执行证据和奖励结算检查仍然执行；原本的 null 保持 null，仅由既有训练策略决定是否映射为 0。有效成功和 provisional 结果仍须质量评分，`purpose=evaluation` 仍保留独立过程评价。

## 产物与验证边界

运行目录包含 `plan.json`、代码/模型身份、解析后的 `verl-config.yaml`、每组环境证据、`update.json`、共享预算和 `summary.json`。verl checkpoint 保存在 `checkpoints/global_step_N/actor/`，包含模型、优化器、训练状态及 HF 模型导出。SwanLab 使用已有项目配置记录组数、更新次数和预算。

启用 `pilot_guards` 时，完成组交给单个后台线程归档，最多保留两个待完成归档任务；达到上限时先等待再采集下一组。`progress.json` 中的 `archive` 是目标路径，只有对应 `archive_manifest` 文件发布后，才表示逐文件哈希验证和落盘完成，允许现有清理器删除源目录。训练正常结束或异常退出时均等待已提交归档；归档失败会抛错，并在 `summary.json` 记录 `failed=true` 和 `archive_error`，不会静默视作成功。`wall_seconds` 仍是归档前的组耗时，整体吞吐需使用运行总耗时。代码修改需新建计划后启动，已有训练进程不热加载这些优化。

离线测试：

```bash
SWANLAB_MODE=disabled OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-verl/bin/python -m pytest -q \
  tests/test_training_verl.py tests/test_training_verl_sft.py tests/test_training_grpo.py
```

测试覆盖 SFT 数据/标签校验、全量样本覆盖、verl 原生 SFT loss 与 masked cross-entropy 梯度一致性、纯文本/多模态基模的文本参数加载，以及 GRPO 并发隔离/取消、终止 token、finalization 输出预算、奖励准入撤销、不同长度和压缩上下文、DP 对齐行、零优势/无效组不更新。GRPO 的 CPU 测试用等价索引操作替代 FlashAttention 的 gather helper，不代表已运行 CUDA 内核。

GPU FSDP2、vLLM 纯文本模型注册后的实际加载、权重同步、显存峰值、完整环境训练和吞吐仍需在满足依赖及准入条件的机器上验收。不能将离线测试通过称为完整 GPU 训练已跑通。
