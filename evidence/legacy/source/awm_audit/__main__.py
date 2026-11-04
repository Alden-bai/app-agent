import argparse
import asyncio
import json
import sys
from pathlib import Path

from .assets import prepare
from .common import ROOT, SEED, save_json
from .config import load_config
from .report import generate
from .runner import preflight, run
from .recovery import resume_judge
from .recheck import recheck


def main():
    p = argparse.ArgumentParser(description="独立审计AWM任务与官方验证器，不评测解题模型。")
    p.add_argument("command", choices=["preflight", "prepare", "run", "report"])
    p.add_argument("--run-dir", type=Path, default=ROOT / "audit_runs" / "20261021_seed20261021")
    p.add_argument("--env-file", type=Path, default=ROOT / ".env")
    p.add_argument("--base-url")
    p.add_argument("--model")
    p.add_argument("--offline", action="store_true", help="preflight仅检查配置与依赖")
    p.add_argument("--source", type=Path, help="已有官方完整JSONL目录；必须同时提供版本SHA")
    p.add_argument("--revision")
    p.add_argument("--endpoint", default="https://huggingface.co")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--task", action="append", help="冻结样本内的scenario:task_idx，可重复指定")
    p.add_argument("--limit", type=int)
    p.add_argument("--mode", choices=["both", "code", "sql"], default="both")
    p.add_argument("--extract-only", action="store_true", help="仅运行SQL证据提取，最终Judge标记未测试")
    p.add_argument("--resume-judge", action="store_true", help="仅恢复传输失败或未测试的Judge，保留首次有效判定和真实轨迹")
    p.add_argument("--recheck", action="store_true", help="按修订后的独立规则复核指定任务的原始轨迹，不重新采样Judge")
    p.add_argument("--reuse-recorded-cases", "--reuse-no-action", dest="reuse_no_action", action="store_true", help="配方修订时保留相同配方及初态、经新规则再次确认的原始案例")
    args = p.parse_args()
    config = load_config(args.env_file, base_url=args.base_url, model=args.model)
    try:
        if args.command == "preflight":
            result = asyncio.run(preflight(config, offline=args.offline))
            save_json(args.run_dir / "preflight.json", result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["llm_status"] in ("reachable", "not_tested") else 2
        if args.command == "prepare":
            result = prepare(args.run_dir, source=args.source, revision=args.revision, endpoint=args.endpoint, seed=args.seed)
            print(json.dumps({"revision": result["revision"], "scenarios": result["scenarios"], "tasks": len(result["tasks"]), "asset_problems": result.get("asset_problems", [])}, ensure_ascii=False, indent=2))
        elif args.command == "run":
            if args.recheck:
                if args.resume_judge:
                    raise ValueError('Use recheck and resume-judge sequentially')
                recheck(args.run_dir, config, tasks=args.task)
            elif args.resume_judge:
                asyncio.run(resume_judge(args.run_dir, config, tasks=args.task, limit=args.limit))
            else:
                asyncio.run(run(args.run_dir, config, tasks=args.task, mode=args.mode, limit=args.limit, judge_enabled=not args.extract_only, reuse_no_action=args.reuse_no_action))
            generate(args.run_dir)
        else:
            generate(args.run_dir)
        print(args.run_dir.resolve() / "report.md" if args.command != "prepare" else args.run_dir.resolve() / "manifest.json")
        return 0
    except Exception as e:
        print(config.redact(f"{type(e).__name__}: {e}"), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
