# AWM Harness

P0–P3 的共享执行内核：冻结任务 → 专属 MCP 环境 → 真实轨迹与一致终态 → 官方 SQL+LLM 适配 → 评测／采集／SFT 导出。

首次获取与安装（Python 3.12+）：

```bash
git clone --recurse-submodules https://github.com/Alden-bai/app-agent.git
cd app-agent
python3.12 -m venv agent-world-model/.venv
agent-world-model/.venv/bin/python -m pip install -e ./agent-world-model -e '.[test]'
cp .env.example .env
```

仅在本地 `.env` 填写自己的 API 密钥。仓库不包含运行日志、数据库、数据集和实验产物；文档中引用的 `data/`、`runs/`、`cleaning_runs/` 需在本地另行准备。`evidence/legacy/source/` 保留脚本依赖的历史源码。官方 AWM 通过固定版本子模块获取。

- [使用说明](docs/harness_readme_zh.md)
- [验收报告](docs/harness_validation_zh.md)
- [设计基线](docs/harness_design_awm_v1_zh.md)
- [Qwen3.5-Flash 配置与调用](docs/qwen_setup_zh.md)
- [D1 rubric 模板、提示词与逐项判定校验](docs/rubric_d1_zh.md)（后续 rubric 开发入口；历史冻结批次沿用旧版）
- [Judge V2：主动只读取证、逐项判定与 unknown 复核](docs/judge_v2_zh.md)（独立冻结轨迹重评及可选自动入口；奖励接口预留）

在本目录运行：

```bash
agent-world-model/.venv/bin/python -m harness_awm --help
```

默认配置使用脚本 Actor、模拟 Judge，付费请求上限为 0；模拟结果不能导出为正式训练数据。真实 API 接口已实现，GRPO 更新和过程 Judge 校准未实现。官方源码及既有清洗产物保持独立。
