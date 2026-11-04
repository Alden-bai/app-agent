"""User-authorized 12-concurrency continuation of the same frozen pilot.

Only scheduling changes. Every task, business request and model identity stays
unchanged, allowing the original completed stages to be reused exactly.
"""
import asyncio
import fcntl
from pathlib import Path

import rubric_d1_repair100 as pilot

WORKERS = 12


def main():
    root = pilot.RUN
    with (root / "runner.lock").open("a") as lock:
        # Acquiring the same lock proves the prior runner has drained its calls.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = pilot.verify_manifest()
        identity = dict(version="d1-repair100-execution-v2", parent_manifest_sha256=pilot.digest(root / "manifest.json"),
            implementation=str(Path(__file__).resolve()), implementation_sha256=pilot.digest(__file__),
            workers=WORKERS, max_concurrency=WORKERS, model=pilot.MODEL, reasoning_effort=pilot.EFFORT,
            authorization="User: 帮我改为12并发跑一下", request_identity_changed=False,
            business_protocol_changed=False, resample=False, retries_added=False)
        pilot.freeze(root / "execution_manifest_v2.json", identity)
        original_prompt = (root / "prompts/run_100.txt").read_text()
        resolved = original_prompt.replace("全局最多3个在途阶段", "全局最多12个在途阶段")
        resolved += "\n执行修订v2：用户明确要求改为12并发。原3并发入口先停止新增派发并收妥在途；按相同100题和请求身份继续。原manifest及所有业务提示词保留，执行参数以execution_manifest_v2.json为准。\n"
        pilot.freeze(root / "run_100_effective_v2.txt", resolved, raw=True)
        stop = root / "STOP"
        if stop.exists():
            stop.unlink()  # This explicit continuation is the requested resume.
        pilot.WORKERS = WORKERS
        print(pilot.compact(dict(resume=True, workers=WORKERS, completed_before=pilot.report()["completed"])), flush=True)
        asyncio.run(pilot.Runner().run(manifest))
        summary = pilot.report()
        summary["execution_manifest_ref"] = str(root / "execution_manifest_v2.json")
        summary["max_concurrency"] = WORKERS
        pilot.snapshot(root / "summary.json", summary)
        print(pilot.compact(summary), flush=True)


if __name__ == "__main__":
    main()
