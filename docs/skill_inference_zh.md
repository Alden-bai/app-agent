# 冻结模型的技能增强推理

2026-11-17 的带技能验证批次已完成独立 DeepSeek 判分。全部 589 题中，275 题确认成功（46.69%），262 题确认失败，47 题无法判定，5 题推理未完成。同一 checkpoint 缺少同条件的无技能对照，技能净提升仍待测量。下面保留了构库、离线预览和推理阶段的操作记录。[结果与来源哈希](qwen35-2b-results.json)

此版本借鉴 SkillRL 的外部技能库：从已有训练轨迹提炼程序性经验，推理时检索并注入上下文。
不更新模型权重，不启动 SFT/GRPO，也不实现训练期间的技能递归演化。
第一版采用可复现的规则提炼，不使用教师模型自由生成技能；它是工具工作流基线，不能等同于论文完整复现。

## 已生成的技能库

`runs/skillbank_20261115_v1/bank.json` 来自
`runs/grpo_full_2746_400steps_deepseek_20261113_v5/persistent/archives` 的 241 个封存组、964 条候选轨迹。
只允许冻结的 `grpo_train` 任务；校验 task/trajectory 的封存哈希，并绑定训练任务的原始指令、环境及 schema。
验证和测试任务只用于分割交叉检查，不读取其轨迹、答案或评分来生成技能。

285 条有效、Judge 确认成功且可抽象的轨迹产生 171 个去重工作流，覆盖 103 个场景。
52 条有效失败轨迹提供 2 类失败提醒，总计 173 条技能。
152 条评分无效或测试样本、39 条带显式工具错误或过长流程、436 条无法按当前规则分类的结果被排除。
`build_report.json` 记录数量；每条技能保留来源归档哈希、成员路径、任务 ID 和文件哈希。
这些是历史 Judge 标签，不是对每个步骤的独立正确性证明。规则抽取无法判断所有业务错误，需通过实际对照评测验证。

工作流只保留工具名、参数字段及步骤，不保留参数值、具体对象 ID、工具返回内容或最终答案。
相邻同类步骤合并；模型按当前请求决定是否重复。失败提醒只针对可观察的终止模式，不把相关性描述为已证明的失败原因。

## 检索与注入

`RunConfig.skills` 默认为空，现有运行关闭技能功能。启用时填写 `inference_skills.json` 中的配置：

```json
{"bank_path":"/absolute/path/bank.json","bank_sha256":"<sha256>","top_k":3,"max_context_bytes":5000}
```

仅支持 v2 的 `purpose=evaluation`。加载时验证库哈希，单个 runner 持有不可变的内存快照。
按任务关键词的 IDF 加权重叠检索；工作流限定相同场景和 schema，且所有工具必须存在于当前目录。
无匹配则不注入，禁止检索任务自身的经验。最多 3 条、5000 UTF-8 字节，每次最多 1 条通用失败提醒。
这里是字节上限，token 数仍由原 runner 的上下文预算检查。

技能置于 `WorkingContext.base`，压缩和最后收尾时保留。每个 episode 的 `skill_selection.json`、
事件和真实输入记录可核查实际注入内容；选择文件进入证据封存。
技能文字不替代当前工具 schema，也不代表当前状态或执行成功。

## 复现构建

在仓库根目录，用含项目依赖的 Python 执行；输出目录必须不存在：

```bash
python -m harness_awm.skillbank_build \
  --tasks data/training_runtime_20261106_v1 \
  --archives runs/grpo_full_2746_400steps_deepseek_20261113_v5/persistent/archives \
  --output runs/skillbank_new
```

## GPU 可用后进行对照

已准备 `configs/training/skill-inference-eval.json`。运行前检查 `runtime_python` 和 Judge API 环境变量。
对照双方使用同一 checkpoint、相同验证任务、greedy 采样、同一 Judge 配置和执行预算。
任务按 seed 和任务 ID 的哈希选取，默认 64 题；`--limit 0` 使用全部 589 道验证题。
这个新实验重新跑无技能基线，不直接与不同运行条件的历史 38.88% 比较。

```bash
python -m harness_awm.training.skill_eval \
  --config configs/training/skill-inference-eval.json \
  --tasks data/training_runtime_20261106_v1 \
  --model runs/grpo_full_2746_400steps_deepseek_20261113_v5/persistent/checkpoints/group_step_200/actor/huggingface \
  --skills runs/skillbank_20261115_v1/inference_skills.json \
  --output runs/skill_eval_pilot \
  --limit 64 --execute
```

不加 `--execute` 仅校验文件并写入计划和检索预览，不加载 GPU、不调用 Judge。
每次选择新的输出目录。执行时先跑 baseline，再跑 skills；模型只加载一次且关闭梯度。
`comparison.json` 包含固定分母成功率、成对改善/退步、未评分数量、调用和 token 总量，以及配对精确检验。
小规模试验只提供初步信号；未评分样本不能静默排除，通用提示的命中率也不等于效果提升。
本轮按用户要求仅构建、接入并做离线测试，实际成功率尚未测量。

已完成全部 589 道验证题的离线检索预览：147 题命中至少一条技能，57 题命中场景专用工作流。
这是检索覆盖率，不是成功率；当前库仅覆盖 103 个场景，未覆盖题目不会获得场景流程帮助。
预览和冻结计划保存在 `runs/skillbank_20261115_v1/validation_preview/`，checkpoint 哈希已与此前选定的 checkpoint-200 核对一致。
人可读的技能索引在 `runs/skillbank_20261115_v1/SKILLS.md`。

已有评测入口也支持 `python -m harness_awm.training.evaluate ... --skills runs/skillbank_20261115_v1/inference_skills.json`。

## 只推理，充值后手动 Judge

2026-11-17 的运行目录为 `runs/skill_inference_checkpoint200_20261117_v1`，
使用验证集选出的 checkpoint-200，全部 589 道验证题，greedy、bfloat16。
最初使用 12 个本地 GPU worker；用户要求提高显存使用率后，在 85 道题完成时切换为自适应并发，
目标 89.5GiB、预留 6GiB、最多 18 个 worker。只在显存允许时逐个扩容，占用超线时在题目边界退场。
`runtime_revision_1` 保留切换前的 manifest 与代码快照，已完成轨迹没有重跑。
这是单个带技能实验；未匹配到技能的任务保持原上下文。本次不重新运行无技能对照。
`status.json` 查看进度；结束后的 `summary.json` 只统计执行情况，成功率为 null，不能把 `finish` 当成任务成功。

纯推理入口是 `python -m harness_awm.training.skill_infer`：先用 `prepare` 冻结模型、任务和技能库，
再用 `run` 执行。执行器固定传入 `evaluate_after=False`，API 请求/Token 预算均为 0，
worker 不加载 `.env` 并移除 API key 环境变量。Judge 配置保留用于兼容证据结构，实际不会调用。
各 episode 保存模型输入输出、`skill_selection.json`、工具轨迹、初始/终态数据库和证据哈希。

本次工作目录位于 `/root/autodl-tmp/awm-skill-checkpoint200-validation-20261117-v1`，
项目内上述路径是快捷链接；完成后自动备份到
`/autodl-fs/data/awm-skill-checkpoint200-validation-20261117-v1/inference.tar.gz`，并写入 SHA-256。
若需恢复，解压保留完整目录。本次另存了全部 589 题的 `judge_sources.json` 及其哈希，
后续脚本优先使用这份冻结原题/rubric，因此评分无需重新执行任务或加载模型。

充值后自行执行以下命令。**只有加 `--execute` 才会调用 Judge**；不需要再加载模型或重新推理。
请先核对配置内的 API 地址、模型和预算（示例配置两个预算为 null，表示不设总量上限）。

```bash
runs/qwen35_base_setup/venv/bin/python scripts/judge_skill_inference.py \
  --run runs/skill_inference_checkpoint200_20261117_v1 \
  --config configs/training/skill-inference-eval.json \
  --output runs/skill_inference_checkpoint200_20261117_judge \
  --execute
```

该脚本校验证据后把评分保存到新的输出目录，不修改原始推理轨迹；
去掉 `--execute` 只查看待评数量。不要使用原始推理 manifest 中预算为 0 的配置启动评分。
