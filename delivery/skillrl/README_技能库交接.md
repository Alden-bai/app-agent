# SkillRL 技能库交接

这次补齐的是可直接用于 AWM 评测的外部技能库：不参与 SFT/GRPO，不更新模型参数。
它借鉴 SkillRL 的推理增强思路，采用规则抽取和关键词检索，不是论文完整训练流程复现。

## 发哪个包

- 没发过旧包：发送 `awm-qwen35-27b-a100x8-with-skillrl.tar.gz` 和对应 `.sha256`，这是包含原训练材料与技能库的新完整包。
- 已经发过旧包：补发 `awm-skillrl-addon.tar.gz` 和对应 `.sha256`。对方应在**全新目录**解压旧包，再在同一目录解压补充包，得到同名 `awm-qwen35-27b/`。补充包含更新的完整清单，合并后可以按完整包校验。不要覆盖已有实验使用的代码快照。
- 两种方式选一种即可。小包依赖原训练交付包，不是独立训练环境。模型权重和 API 密钥仍需对方准备。

```bash
# 完整包
sha256sum -c awm-qwen35-27b-a100x8-with-skillrl.tar.gz.sha256
tar -xzf awm-qwen35-27b-a100x8-with-skillrl.tar.gz
cd awm-qwen35-27b
python3.12 scripts/verify_training_delivery.py .
```

补发方式在新空目录内执行：

```bash
sha256sum -c awm-qwen35-27b-a100x8-delivery.tar.gz.sha256
sha256sum -c awm-skillrl-addon.tar.gz.sha256
tar -xzf awm-qwen35-27b-a100x8-delivery.tar.gz
tar -xzf awm-skillrl-addon.tar.gz
cd awm-qwen35-27b
python3.12 scripts/verify_training_delivery.py .
```

## 库里是什么

| 文件 | 用途 |
|---|---|
| `delivery/skillrl/bank.json` | 冻结技能库，173 条：171 条工具流程、2 条失败提醒，覆盖 103 个场景 |
| `delivery/skillrl/SKILLS.md` | 人可读的完整技能索引 |
| `delivery/skillrl/inference_skills.json` | 迁移模板，`bank_path` 相对本文件所在目录；请通过 prepare 命令使用 |
| `delivery/skillrl/build_report.json` | 历史构建计数和库哈希 |
| `delivery/skillrl/verification.json` | 本次离线来源、数据绑定和检索覆盖检查 |
| `scripts/prepare_delivery_skills.py` | 校验哈希和训练来源，生成接收方本机的绝对路径配置 |
| `harness_awm/skillbank_build.py` | 从训练轨迹封存包重新抽取技能的源码 |

来源为历史 GRPO 的 241 个封存组、964 条候选轨迹；285 条可抽象成功轨迹与 52 条可分类失败轨迹提供技能。
库的 SHA256 为 `e206e99591bf5acd097ee79966d4d7bc6ae97aa299d29752d8f666af0d12407a`。
库内旧绝对路径仅记录历史来源，不参与加载或检索，不要替换它们，否则会改变冻结哈希。
本交接包含抽取结果和来源哈希，不包含这 241 个原始 GRPO 归档，不能据此独立重放原始抽取。
若后续从新训练轨迹重新构建，使用新目录并重新冻结，在查看测试结果前完成。

技能仅源自冻结的 `grpo_train`，不从验证／测试轨迹、答案或评分抽取。
运行时要求场景、工具 schema 和工具名匹配，不检索任务自身经验。
最多注入 3 条、5000 UTF-8 字节；无匹配时保留原上下文。
589 道验证题中，147 题可命中至少一条技能，其中 57 题命中场景工作流；这是覆盖率，不是成功率。

## 对方怎么跑

先按 `delivery/README_交接说明.md` 安装环境、准备模型并初始化工作目录，完成 SFT/GRPO 及验证选模。
本补充只增加技能交付和评测入口，训练参数仍按双方约定设置；如 GRPO 要 600 次有效更新，应显式传 `--target-updates 600 --save-every 200`（旧入口默认 800）。
固定用 **589 道验证题选出的最佳 GRPO 的 Hugging Face 导出模型**做两组评测，不按测试结果重新选 checkpoint。
技能配置先保持 top_k=3、max_context_bytes=5000；需要调参时仅使用验证集，定稿后再跑测试集。

下面在解压后的项目根目录运行。`WORK` 为已初始化的实验目录，`BEST_GRPO` 替换为实际导出目录。

```bash
WORK=/data/local/awm27b-run1
BEST_GRPO=/data/local/awm27b-run1/grpo/checkpoints/global_step_400/actor/huggingface

# CPU 校验及路径重定位：不加载模型、不调用 Judge；目标配置必须不存在。
.venv-delivery-env/bin/python scripts/a100_delivery.py skills-prepare --work "$WORK"

# 离线预检：读模型文件哈希和任务资产，生成计划；不启动 GPU / API。
# 此处与正式运行要使用不同的新输出目录。
CUDA_VISIBLE_DEVICES=0 .venv-delivery-grpo/bin/python scripts/a100_delivery.py skills-evaluate \
  --work "$WORK" --model "$BEST_GRPO" --split test --limit 0 \
  --output "$WORK/skillrl-test-preflight"

# 正式对照：同一模型先跑 baseline，再跑 skills；总计 588 × 2 次任务执行。
CUDA_VISIBLE_DEVICES=0 .venv-delivery-grpo/bin/python scripts/a100_delivery.py skills-evaluate \
  --work "$WORK" --model "$BEST_GRPO" --split test --limit 0 \
  --output "$WORK/skillrl-test-paired" --execute
```

`skills-prepare` 生成 `$WORK/inference_skills.json` 和同名 `.verification.json`，校验冻结库及训练来源。
若整个包再次搬家，使用新的配置输出路径重新 prepare；不要复用旧机器绝对路径。
`skills-evaluate` 默认只生成计划，只有 `--execute` 才加载模型并调用 Judge。
可先在 validation 用 `--limit 8` 做短测；全量 validation 使用 `--limit 0`，共 589 题。
同一 checkpoint、同一任务、greedy、BF16、同一 Judge／预算，仅技能开关不同。
单进程单 GPU 加载模型；27B 目标机仍需实际验证显存和完整任务运行。
不要运行历史 `skill_infer` 的多副本入口，它的显存调度针对旧机器与小模型。
正式运行前结束训练并释放 GPU。Judge 费用来自两组评测；预算在 `$WORK/eval-config.json` 中设置，分别对每组生效。

若已跑完同条件 baseline，只想补跑有技能一组，可以使用：

```bash
CUDA_VISIBLE_DEVICES=0 .venv-delivery-grpo/bin/python scripts/a100_delivery.py evaluate \
  --work "$WORK" --model "$BEST_GRPO" --split test \
  --skills "$WORK/inference_skills.json" --output "$WORK/test-best-grpo-skills"
```

注意：普通 `evaluate` 命令直接执行，和 `skills-evaluate` 的预检默认行为不同。
独立单组运行不自动生成配对报告；正式对照推荐上面的双组入口。

## 收哪些结果

从 `skillrl-test-paired/` 收回 `plan.json`、`comparison.json`、`status.json` 和两组 `metrics.json`、逐题报告。
`comparison.json` 给出两组固定分母成功率、百分点差、改善／退步数量、未评分数量及调用统计。
逐题 `episodes/*/a0/skill_selection.json` 记录真正注入的技能，不能只看配置判断是否生效。
未评分样本仍保留在分母；有错误或预算耗尽时先核查覆盖率，不能把检索命中当作成功率提升。
本次交接只完成离线校验和测试，没有运行新的 27B GPU / Judge 对照，效果以接收方实测为准。
