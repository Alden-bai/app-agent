# Qwen3.5-27B / 单机 8×A100 80GB 交接说明

此包用于重新运行纯文本全参数 SFT → 验证选模 → GRPO → 验证／最终测试。
不含模型权重、checkpoint、优化器状态、虚拟环境和密钥。对方单独下载官方权重。
GPU 训练尚需在目标八卡服务器完成下面的短测；本地验证状态见 `validation.json`。

## 1. 数据到底有多少

| 数据 | 数量 | 含义 |
|---|---:|---|
| SFT | **758 条完整轨迹** | 沿用原筛选结果，没有扩充轨迹 |
| SFT 展开样本 | **6,714 个对话轮次** | 总共 7,071 轮，357 轮不监督；每个样本只监督当前 assistant 回复 |
| GRPO train | 2,746 道任务 | 在线生成新轨迹，默认每题 4 个候选 |
| validation | 589 道任务 | 选择 checkpoint |
| test | 588 道任务 | 仅最终评测，不选模、不更新参数 |

27B tokenizer 已按固定 revision 重新编码。`data/training_qwen35_27b_v1/report.json`
记录模板、tokenizer、划分和文件哈希。最长样本 17,743 tokens，上限 32K；未截断、未丢弃。
本批 27B 编码结果恰好与旧 2B 编码相同，模板／tokenizer 配置文件的身份仍分别记录。
验证和测试集是可执行环境任务，没有参考轨迹，不能作为 SFT eval_loss 数据。

## 2. 包内目录与完整性

- `harness_awm/`：任务执行、Judge、奖励、SFT、GRPO 和评测代码。
- `agent-world-model/`：实际使用的 AWM 核心源码；不需要 benchmark 扩展子模块。
- `vendor/verl/`：固定 veRL commit 的完整源码和原有许可证。
- `assets/qwen35-27b/`：官方模型配置、tokenizer、模板和 revision 锁，**无权重**。
- `data/training_split_20261106_v1/`：冻结划分、SFT 轨迹索引和监督标注。
- `data/training_runtime_20261106_v1/`：全部训练／验证／测试任务、冻结数据库、环境和 Judge 私有材料。
- `runs/`、`cleaning_runs/`：仅按来源清单收集的必要轨迹和筛选材料，不是全部历史实验。
- `calibration/history/`：既有校准报告及控制轨迹，原始字节保留。报告内旧绝对路径仅作来源记录。
- `DELIVERY_MANIFEST.json`：逐文件大小和 SHA256、代码快照状态、验收边界。

收到包后先执行（校验文件与 tar.gz 放在同一目录）：

```bash
sha256sum -c awm-qwen35-27b-a100x8-delivery.tar.gz.sha256
tar -xzf awm-qwen35-27b-a100x8-delivery.tar.gz
cd awm-qwen35-27b
python3.12 scripts/verify_training_delivery.py .
```

冻结清单里的 `/home/bjx/awm-agent` 是历史来源根目录。读取器用当前仓库根目录重定位，
不要搜索替换这些原始 JSON，否则会破坏哈希。新训练计划必须在目标主机重新 prepare。
Judge rubric 只交给评判器，不进入模型 prompt 或 SFT 标签。

## 3. 服务器与安装

Linux x86_64、Python 3.12、8 张 A100 80GB、可联网访问包源；Landlock 功能必须可用。
沿用的 GRPO 环境使用 CUDA 13，要求 NVIDIA 驱动 580 或更新；SFT 使用 CUDA 12.8。
不要将原机器的虚拟环境、PRO 6000 内核缓存或 85GB 显存阈值搬过来。
推荐至少 512 GiB 系统内存（预检要求至少 384 GiB 当前可用），预留 checkpoint 存储约 3 TiB。
27B 全状态 checkpoint 含模型、Adam 状态和 HF 导出，可达数百 GB；默认保存 2 个 SFT 和 4 个 GRPO 阶段 checkpoint。
队列、预算 SQLite 和活跃 rollout 放本地盘，不要放不支持可靠锁的网络文件系统。

三个环境彼此独立，可通过安装命令 `--venv` 改路径：

```bash
python3.12 scripts/setup_delivery.py env
python3.12 scripts/setup_delivery.py sft
python3.12 scripts/setup_delivery.py grpo
cp delivery/env.example .env
# 自行填写 .env 中的 OPENAI_API_KEY；默认关闭 SwanLab。
```

安装脚本使用 `requirements/delivery-{sft,grpo,env}.lock.txt` 的完整传递依赖版本锁，
校验 veRL 固定源码及核心版本，执行 `pip check`，记录 `installed.freeze.txt`。
GRPO 仅对 vendored veRL 的安装元数据进行既有 numpy2 兼容修改，训练源码保持固定哈希。
不要把 SFT 和 GRPO 的依赖装进同一个环境。

## 4. 模型与工作目录

使用 `assets/qwen35-27b/model-assets.lock.json` 中的 repository 和 revision 下载完整官方模型到包外。
可用 `huggingface_hub.snapshot_download(repo_id=..., revision=..., local_dir=...)`；不要使用 FP8/GPTQ/LoRA 版本。
只训练全部文本参数，不加载视觉塔；不使用 LoRA、量化或冻结文本层。

下面示例将 `MODEL`、`WORK`、`SFT_MODEL` 换成对方机器的实际路径：

```bash
MODEL=/data/models/Qwen3.5-27B
WORK=/data/local/awm27b-run1
.venv-delivery-env/bin/python scripts/a100_delivery.py init --work "$WORK"
.venv-delivery-env/bin/python scripts/check_delivery_data.py
.venv-delivery-sft/bin/python scripts/a100_delivery.py doctor --work "$WORK"
.venv-delivery-sft/bin/python scripts/a100_delivery.py distributed-check --work "$WORK"
```

`init` 生成本机路径的 `config.json` 和 `eval-config.json`。Judge 沿用交付模板，不含密钥。
默认 Judge 累计请求／token 预算为 null（沿用现有设置），正式评测和训练会消耗 API；
需要限额时在工作目录配置里填写正整数。SwanLab 默认 disabled，可自行启用。
CPU 数据重建入口为 `a100_delivery.py prepare-data --work "$WORK"`，输出新目录，不覆盖已打包数据。

## 5. SFT 与选模

```bash
.venv-delivery-sft/bin/python scripts/a100_delivery.py sft-smoke --work "$WORK" --model "$MODEL"
.venv-delivery-sft/bin/python scripts/a100_delivery.py sft --work "$WORK" --model "$MODEL"
```

短测使用最长的 8 个真实样本，八卡各参与一次更新，保存在独立 `sft-smoke/`。
正式训练仍从原始基模开始。默认 FSDP2、BF16 计算、FP32 模型参数、优化器 CPU 卸载、激活检查点，
每卡微批 1、全局 batch 64、学习率 1e-5、2 epochs，预计每 epoch 105 次更新。
默认禁用仅支持单卡的动态 token 微批。可传 `--epochs`、`--learning-rate`、`--global-batch-size`。
模型导出通常在 `sft/checkpoints/global_step_N/huggingface/`，训练状态在其父目录。

先退出训练进程释放 GPU，再做加载验收（将 N 换为实际步数）：

```bash
CUDA_VISIBLE_DEVICES=0 .venv-delivery-sft/bin/python scripts/verify_delivery_checkpoint.py \
  --checkpoint "$WORK/sft/checkpoints/global_step_N" --output "$WORK/sft-reload-N.json"
SFT_MODEL="$WORK/sft/checkpoints/global_step_N/huggingface"
CUDA_VISIBLE_DEVICES=0 .venv-delivery-sft/bin/python scripts/a100_delivery.py evaluate \
  --work "$WORK" --model "$SFT_MODEL" --split validation --output "$WORK/validation-sft-N"
```

对两个 epoch 都运行 validation；先核对完整覆盖 589 题，再按确认成功率选择，
同分时看平均 eval_reward，再同分选较早 checkpoint。基模单独评测作对照，不参加 SFT 选优。
评测默认单 GPU BF16、greedy、关闭 thinking、pass@1，不启动原机的多副本 worker 脚本。
使用 `--task 场景:题号` 可以先做单题短测；子集结果不能当作全量结果。

## 6. 校准与 GRPO

现有历史校准的 Judge 实现哈希与当前代码不一致，**不能直接用于此次正式训练**。
这不是换了模型导致失效；校准绑定的是 Judge／环境／任务代码。旧证据作为审计材料保留。
下面两步会对既有两道训练控制题重新执行独立正反例和质量控制，再建立全量训练题来源绑定：

```bash
.venv-delivery-env/bin/python scripts/a100_delivery.py calibrate --work "$WORK"
.venv-delivery-env/bin/python scripts/a100_delivery.py admissions --work "$WORK"
```

任一控制不通过就停止，不修改通过标记，不伪造准入。全量准入是 shared-suite，
不是 2,746 题每题做过正反校准。修改 Judge 模型、参数、实现或环境后须重新校准。

```bash
.venv-delivery-grpo/bin/python scripts/a100_delivery.py doctor --work "$WORK"
.venv-delivery-grpo/bin/python scripts/a100_delivery.py distributed-check --work "$WORK"
.venv-delivery-grpo/bin/python scripts/a100_delivery.py grpo-probe --work "$WORK" --model "$SFT_MODEL"
.venv-delivery-grpo/bin/python scripts/a100_delivery.py grpo-smoke --work "$WORK" --model "$SFT_MODEL"
.venv-delivery-grpo/bin/python scripts/a100_delivery.py grpo --work "$WORK" --model "$SFT_MODEL"
```

`grpo-probe` 不调用 Judge，但会做独立诊断更新和权重同步，不能作为正式训练结果。
`grpo-smoke` 最多采样 4 组，要求至少 1 次有信号更新；全同分未更新时会明确失败，应检查原始轨迹和奖励。
正式 GRPO 从选出的 SFT 权重重新开始，不继承 probe/smoke 状态。

八卡 FSDP2；vLLM TP=4、两个副本，每题仍采样 4 条轨迹，并发最多 4 个环境。
逐轮训练样本补齐到八卡，补齐行 loss=0；每组仅一次 optimizer update。
默认学习率 1e-6、目标 800 次有效更新、最多 3,200 组、每 200 次有效更新保存；
`--target-updates`、`--groups`、`--save-every` 可覆盖。候选数保持 4。
保留 CST 奖励、null-as-zero 训练映射、工具错误恢复和 token-IS 修正；原始 Judge null 记录不改成失败。
vLLM 显存比例初始 0.35，优化器卸载。不得仅把单卡显存百分比改成 0.82 直接启动。

GRPO checkpoint 位于 `grpo/checkpoints/global_step_N/actor/`；
使用 `verify_delivery_checkpoint.py --checkpoint .../actor` 验证八份状态文件并在单卡 BF16 重载。
保留全部 rank 文件。此工具验证导出可加载和推理，不声称已验证优化器断点恢复。

## 7. 最终评测、失败处理与验证边界

对基模、选出的 SFT 和 GRPO 最佳模型，用同一 `eval-config.json` 运行 validation/test；
只用 validation 选模型，选定后再运行 `--split test`。报告目录内有逐题 CSV、JSON 指标和中文说明。
保持 EOS、关闭 thinking、Judge 配置和任务预算一致；比较成功率时同时检查评分覆盖率与基础设施错误。

所有训练计划绑定模型、源码和绝对运行路径，不能移动已 prepare 的计划继续用。
修改代码／配置后使用新输出目录重新 prepare。旧一机恢复、后台监控和清理脚本属于历史工具，
本交付默认只使用 `a100_delivery.py`，不要运行含旧绝对路径的脚本。
SFT 入口没有断点续训命令；GRPO 旧恢复工具针对原单卡实验，本交付不承诺八卡 optimizer 恢复。
异常留下完整证据，不自动清空重跑、不混入测试题、不把 OOM 或缺失准入静默当作成功。

若 GPU 内存不足，先检查是否有残留模型服务，保留长样本和上下文长度；使用新的运行计划调节卸载，
不要通过删样本或缩短标签掩码掩盖问题。首次 Triton/vLLM 编译耗时不等同于稳定吞吐。
仅当 NCCL、最长 SFT 样本、GRPO 权重同步、真实环境短测和 checkpoint 重载均通过，才视为目标机器可启动正式训练。

## 8. SkillRL 技能库补充交接

新版 `awm-qwen35-27b-a100x8-with-skillrl.tar.gz` 已包含冻结的 173 条技能、迁移配置和有无技能对照入口。
详见 `delivery/skillrl/README_技能库交接.md`。技能仅用于推理，最佳 GRPO 模型按验证集选定后，
用 `skills-prepare` 初始化本机路径，再用 `skills-evaluate --split test --limit 0 --execute` 运行双组对照。
新版包的解压／校验文件名以该技能库交接说明为准；原训练安装及数据流程保持一致。
