# TRL GRPO GPU 小规模诊断与推理加速

2026-11-10 在单张 RTX PRO 6000 Blackwell 96GB 上完成真实 `trl.GRPOTrainer` 运行：从验证集选出的 Qwen3.5-2B SFT checkpoint-210 开始，对全部 1,881,825,088 个文本参数开放训练，完成 3 个优化器 step、每组 4 个回答，共 12 个回答。入口是 `harness_awm.training.trl_smoke`。

这次按用户确认的“先跑通 TRL 并测加速”范围执行。使用冻结 **SFT 训练集**中的 3 个真实上下文，分别为 689、1301、782 tokens；没有使用验证集或测试集作为训练样本，也没有截断输入。模型初始化来自既有验证集选择结果。

## 奖励与验证边界

奖励仅用于诊断：合法 JSON 动作得 0.25，动作类型与参考动作一致再得 0.25，规范化 JSON 与参考动作的字符相似度贡献最多 0.5。非法输出为 0。这是单轮参考动作奖励，不是 Judge V2 的业务奖励，不证明生成的工具调用可执行或任务完成。

实际训练仍由 TRL 完成组采样、组内优势计算、GRPO 损失、反向传播和优化器更新。没有伪造组内奖励、重采样挑选组，也没有调用付费 Judge。使用 `loss_type=grpo`、`scale_rewards=group`、`beta=0`、temperature=1、top_p=1、top_k=0，学习率 2e-5，weight_decay=0。TRL 此版本的组标准差采用样本估计，与项目自写 GRPO 的总体标准差约定有差异。

训练段耗时 **7.91 秒**，含真实采样与优化，不含模型加载、独立基准和模型保存。梯度范数为 **38.75 / 20.75 / 18.125**，每个参数张量抽取约 128 个位置检查，发现 **231 个张量**的抽样权重变化。峰值已分配显存 **17.03 GiB**，峰值 PyTorch 保留显存 **18.06 GiB**。loss 接近 0 是单次 on-policy 更新时组优势均值接近 0 的结果；非零梯度和权重变化是此次更新验收依据。

最后一组有 3 个回答达到 192-token 上限，因此本轮共有 3/12 个回答被截断；这轮配置保留其诊断损失，没有把截断回答当作成功样本。三个 step 和参考动作奖励都不足以判断模型质量提升，诊断模型不能代替正式 SFT 最优模型。

正式多轮 AWM GRPO 仍须处理宿主机 strict namespace 不可用、奖励准入配置哈希过期的问题；本入口没有修改这些正式奖励门槛。

## 推理基准

同一 checkpoint、BF16、SDPA、KV cache，两个训练上下文各重复 4 次，总共 8 条序列，每条固定生成 48 tokens。三种模式均关闭 EOS 提前停止，以保证完全相同的 384-token 工作量。预热不计时，交替测量两次并取中位数；分词和输入搬运不计入生成计时。实际 GRPO 训练正常使用 EOS。

| 生成方式 | 耗时 | 总输出吞吐 |
|---|---:|---:|
| Transformers 参考内核，串行 | 7.302 秒 | 52.6 tokens/s |
| FLA + causal-conv1d，串行 | 6.686 秒 | 57.4 tokens/s |
| FLA + causal-conv1d，每组 4 条批量生成 | 1.758 秒 | 218.4 tokens/s |

内核部分加速 **1.09 倍**，在快速内核上增加批量生成再加速 **3.80 倍**，合计 **4.15 倍**。三种模式所有贪心输出 token 完全一致。该倍率仅适用于这次固定长度生成基准，不是完整训练、多轮环境或 Judge 的加速倍率；只有两个上下文，不能外推到所有长度和 batch size。

TRL 训练直接使用其原生批量 `generate`，保留 KV cache；FLA 与 causal-conv1d 在导入 Transformers 前加载，并验证四个内核实际绑定。梯度检查点降低训练显存。vLLM 未安装，本轮没有使用 vLLM、量化或 LoRA，也没有更换原有 Torch 环境。

## 本机复现

已实测版本：Torch 2.10.0+cu128、Transformers 5.22.1、TRL 1.17.1、Accelerate 1.15.0、Datasets 5.1.0、flash-linear-attention 0.6.0、causal-conv1d 1.7.0。依赖目录是本机已有产物，迁移时需准备同版本环境。

从仓库根目录运行，输出目录必须不存在：

```bash
PYTHONPATH=runs/sft_eval_kernel_deps:runs/sft_kernel_deps \
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
runs/qwen35_base_setup/venv/bin/python -u -m harness_awm.training.trl_smoke \
  --model /autodl-fs/data/awm-sft-20261109-v2/training/lr1e-5/checkpoint-210 \
  --output runs/trl_grpo_smoke_repeat \
  --steps 3 --group-size 4 --save-model
```

`--skip-benchmark` 可跳过独立基准。不加 `--save-model` 时只保存日志与证据，避免占用约 3.5 GiB 模型空间。保存的是可加载的完整模型和 tokenizer，不含优化器状态，不支持从此次输出精确恢复中断训练。

本次结果位于 `runs/trl_grpo_smoke_20261110/attempt1/`：

- `summary.json`：训练结果、显存和权重变化记录。
- `runtime.json`：版本、原模型哈希、源脚本哈希、内核绑定、数据来源和参数。
- `examples.json`、`sampled_rewards.json`：输入来源、真实生成和诊断奖励。
- `benchmark_summary.json`、`benchmark_measurements.json`：计时和逐 token 对比。
- `training_logs.json`、`generation_times.json`：逐 step 指标与生成耗时。
- `diagnostic_model/`：独立的诊断模型。
- `reload_check.json`：重新加载、有限前向 logits 与实际生成验证。

参考接口：[TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)、[Transformers Qwen3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5)。实际参数和行为以本机锁定版本及运行证据为准。
