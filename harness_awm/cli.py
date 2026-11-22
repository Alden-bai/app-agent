import argparse
import asyncio
import json
from pathlib import Path
from .contracts import RunConfig
from .export import export_sft
from .io import read_json
from .registry import prepare
from .stages import diagnose, reevaluate, run_stage


def main():
    from dotenv import load_dotenv
    from .contracts import ROOT
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description="AWM Harness P0-P3; no paid requests by default")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("swanlab-check", help="create a SwanLab connection-check experiment; no model calls")
    p.add_argument("--run", required=True, help="local directory for SwanLab logs")
    p.add_argument("--mode", choices=("online", "offline"), default="online")
    p = sub.add_parser("prepare", help="verify cleaning inputs and rebuild frozen initial states")
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--splits", help="JSON mapping scenario to train/dev/test")
    p.add_argument("--admissions", help="JSON mapping task_instance_id to explicit reward admission")
    for name in ("collect", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--registry", required=True)
        p.add_argument("--run", required=True)
        p.add_argument("--config", required=True)
        p.add_argument("--scripts", help="JSON mapping scenario:task_id to scripted actions")
        p.add_argument("--task", action="append", dest="task_ids")
    for name in ("reevaluate", "diagnose"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
    p = sub.add_parser("export-sft")
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tokenizer")
    p.add_argument("--chat-template")
    p = sub.add_parser("judge-prepare", help="bind source rubric and sealed episode for Judge V2; no model calls")
    p.add_argument("--episode", required=True)
    p.add_argument("--original", required=True)
    p.add_argument("--rubric", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("judge-replay", help="evaluate a frozen JudgeInput without modifying its episode")
    p.add_argument("--input", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("judge-report", help="summarize V2 assessments independently of rewards")
    p.add_argument("--results", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if hasattr(args, "config"):
        config = RunConfig.model_validate(read_json(args.config))
        config = config.model_copy(update={"official_root": str(Path(config.official_root).resolve()),
                                           "runtime_python": str(Path(config.runtime_python).absolute())})
    if args.command == "swanlab-check":
        from .tracking import check_connection
        result = check_connection(args.run, args.mode)
    elif args.command == "judge-prepare":
        from .judge_inputs import load_json, prepare_input
        from .io import canonical
        result = prepare_input(args.episode, load_json(args.original), load_json(args.rubric))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            stream.write(canonical(result) + "\n")
        result = {"input": str(output.resolve()), "task_id": result["original"]["task_id"]}
    elif args.command == "judge-replay":
        from .backend import Budget
        from .judge_inputs import load_json
        from .judge_runtime import evaluate_input
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        budget = Budget(output / "budget.sqlite", config.max_physical_requests, config.max_total_tokens)
        result = asyncio.run(evaluate_input(load_json(args.input), config, budget, asyncio.Semaphore(config.judge_concurrency), output))
    elif args.command == "judge-report":
        from .judge_report import report_directory
        result = report_directory(args.results, args.output)
    elif args.command == "prepare":
        result = asyncio.run(prepare(args.source, args.output, config, args.splits, args.admissions))
    elif args.command in {"collect", "evaluate"}:
        result = asyncio.run(run_stage(args.command, args.registry, args.run, config, args.scripts, args.task_ids))
    elif args.command == "reevaluate":
        result = asyncio.run(reevaluate(args.run))
    elif args.command == "diagnose":
        result = diagnose(args.run)
    else:
        result = export_sft(args.run, args.output, args.tokenizer, args.chat_template)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

