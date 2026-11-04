"""Frozen D1 repair pilot: 100 accepted rubrics with unresolved issues.

Independent fresh Luna sessions; no actor runs, no production-bank mutation.
"""
import argparse
import asyncio
from collections import Counter
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys

import rubric_d1 as d1
import rubric_luna100 as old

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/rubric_d1_repair100_20261103_v1"
PROMPTS = ROOT / "configs/rubric/d1_repair100_v1"
MODEL, EFFORT, SIZE, SEED, WORKERS = "gpt-6-luna", "high", 100, 20261103, 3
read, freeze, snapshot, digest, now = old.read, old.freeze, old.snapshot, old.digest, old.now
old.MODEL, old.EFFORT = MODEL, EFFORT


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def htext(text):
    return hashlib.sha256(text.encode()).hexdigest()


def source_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def prepare():
    if (RUN / "manifest.json").exists():
        raise ValueError("already_prepared_use_frozen_manifest")
    summary_path = ROOT / "runs/accepted_rubric_unknown_audit_20261103/summary.json"
    summary = read(summary_path)
    exports = summary["source_exports"]
    for source in exports.values():
        if digest(source["path"]) != source["sha256"]:
            raise ValueError("source_export_hash_mismatch")
    accepted = {r["task_id"]: r for r in source_rows(exports["accepted_rubrics"]["path"])}
    bank = {r["task_id"]: r for r in source_rows(exports["question_bank"]["path"])}
    eligible = sorted(k for k, r in accepted.items() if r["rubric"]["unresolved"])
    assert len(eligible) == summary["questions_with_nonempty_unresolved"] == 4909
    selected = sorted(random.Random(SEED).sample(eligible, SIZE))
    tasks = []
    for task_id in selected:
        a, b = accepted[task_id], bank[task_id]
        if digest(a["source_artifact_ref"]) != a["source_artifact_sha256"] or digest(b["original_ref"]) != b["original_sha256"]:
            raise ValueError("historical_artifact_hash_mismatch:" + task_id)
        original, rubric = read(b["original_ref"]), read(a["source_artifact_ref"])
        if rubric != a["rubric"] or original["task"] != b["task"] or original["task_id"] != task_id:
            raise ValueError("historical_content_mismatch:" + task_id)
        for source in b["sources"].values():
            if digest(source["path"]) != source["sha256"]:
                raise ValueError("historical_source_hash_mismatch:" + task_id)
        view, pack = d1.base.prepare_input(original)
        errors = d1.validate_output(rubric, original)
        if errors:
            raise ValueError("invalid_old_rubric:" + str(errors))
        folder = RUN / "inputs" / task_id.replace(":", "__")
        row = dict(task_id=task_id, old_run=b["run_id"], historical_original_ref=b["original_ref"],
                   historical_rubric_ref=a["source_artifact_ref"], historical_sources=b["sources"],
                   old_unresolved=len(rubric["unresolved"]))
        for name, data in [("original", original), ("old_rubric", rubric), ("view", view), ("audit", pack)]:
            path = freeze(folder / (name + ".json"), data)
            row[name + "_ref"], row[name + "_sha256"] = str(path), digest(path)
        tasks.append(row)
    policy = (ROOT / "configs/rubric/d1/policy.txt").read_text() + "\n" + (PROMPTS / "business_boundaries.txt").read_text()
    execute = (PROMPTS / "execute.txt").read_text()
    for role, base_role, extra in [("repair", "generate", "repair"), ("review", "judge", "review")]:
        protocol = execute + "\n" + (ROOT / "configs/rubric/d1" / (base_role + ".txt")).read_text() + "\n" + policy + "\n" + (PROMPTS / (extra + ".txt")).read_text()
        freeze(RUN / "prompts" / (role + ".txt"), protocol, raw=True)
    freeze(RUN / "prompts/run_100.txt", (PROMPTS / "run_100.txt").read_text(), raw=True)
    freeze(RUN / "eligible_task_ids.json", eligible)
    ledger = read(ROOT / "runs/rubric_luna_registry/batches.json")
    manifest = dict(protocol="rubric-d1-repair100-v1", created_at=now(), model=MODEL, reasoning_effort=EFFORT,
        sample_size=SIZE, eligible_size=len(eligible), seed=SEED, sampling="sorted(Random(seed).sample(sorted(eligible_ids),100))",
        scope="accepted rubrics with nonempty unresolved; not executed unknown trajectories", tasks=tasks,
        source_exports=exports, source_summary_sha256=digest(summary_path),
        prompt_hashes={p.name: digest(p) for p in (RUN / "prompts").glob("*.txt")},
        implementation_hashes={str(p): digest(p) for p in [Path(__file__).resolve(), ROOT / "scripts/rubric_d1.py", ROOT / "scripts/rubric_candidates_100.py", ROOT / "scripts/rubric_candidates.py", ROOT / "scripts/rubric_pilot.py", ROOT / "scripts/rubric_luna100.py", ROOT / "harness_awm/rubric_judge.py"]},
        workers=WORKERS, max_stage_calls=400, max_additional_revisions=1, automatic_retries=0,
        cumulative_token_budget=None, authorization="User requested this prompt and actual 100-task repair pilot; no full-bank run.",
        historical_ledger_ref=str(ROOT / "runs/rubric_luna_registry/batches.json"), historical_ledger_sha256=digest(ROOT / "runs/rubric_luna_registry/batches.json"),
        historical_luna_known_tokens=sum((v.get("luna_actual_usage") or {}).get("total_tokens", 0) for v in ledger.values()),
        pass_rule="five valid integer scores in 0/1/2, none zero; all-2 reported separately",
        semantic_review_by_coordinator=False, actual_runtime_unknown_measured=False, production_promotion=False)
    freeze(RUN / "manifest.json", manifest)
    print(compact(dict(prepared=SIZE, old_unresolved=sum(r["old_unresolved"] for r in tasks), model_calls=0)))


def verify_manifest():
    m = read(RUN / "manifest.json")
    assert m["model"] == MODEL and m["reasoning_effort"] == EFFORT and m["sample_size"] == SIZE
    for name, value in m["prompt_hashes"].items():
        if digest(RUN / "prompts" / name) != value:
            raise ValueError("prompt_hash_mismatch")
    for path, value in m["implementation_hashes"].items():
        if digest(path) != value:
            raise ValueError("implementation_hash_mismatch:" + path)
    for t in m["tasks"]:
        for name in ("original", "old_rubric", "view", "audit"):
            if digest(t[name + "_ref"]) != t[name + "_sha256"]:
                raise ValueError("input_hash_mismatch")
    return m


def review_payload(original, selection, rubric):
    view, _ = d1.base.prepare_input(original)
    # No duplicate items/predicates or generator reasoning. The current mapping
    # is retained so candidate IDs can be used to identify omitted obligations.
    mapping = [{k: s[k] for k in ("candidate_id", "action", "item_ids")} for s in selection["selection"]]
    return dict(input=view, candidate_mapping=mapping, rubric=rubric)


class Paused(Exception):
    pass


class CallFailed(Exception):
    pass


class Runner:
    def __init__(self):
        self.stopped = False
        self.failures = 0
        self.stop_reason = None

    def stop(self, reason):
        self.stopped, self.stop_reason = True, reason
        snapshot(RUN / "stop.json", dict(at=now(), reason=reason))

    async def call(self, task_id, stage, role, payload):
        folder = RUN / "tasks" / task_id.replace(":", "__") / stage
        protocol = (RUN / "prompts" / (role + ".txt")).read_text()
        message = compact(dict(protocol=protocol, data=payload))
        identity = dict(task_id=task_id, stage=stage, model=MODEL, reasoning_effort=EFFORT,
                        message_sha256=htext(message), prompt_sha256=htext(protocol))
        freeze(folder / "request.json", dict(identity=identity, protocol=protocol, data=payload))
        completed = folder / "completed.json"
        if completed.exists():
            prior = read(completed)
            if prior["identity"] != identity:
                raise ValueError("cache_identity_mismatch")
            if prior["status"] != "completed":
                raise CallFailed(prior["status"])
            if digest(folder / "response.txt") != prior["response_sha256"]:
                raise ValueError("cached_response_hash_mismatch")
            return (folder / "response.txt").read_text()
        if (folder / "started.json").exists():
            self.stop("sent_request_requires_reconciliation:" + str(folder))
            raise Paused(self.stop_reason)
        if self.stopped or (RUN / "STOP").exists():
            raise Paused("dispatch_stopped")
        freeze(folder / "started.json", dict(identity=identity, started_at=now()))
        cmd = ["codex", "exec", "--skip-git-repo-check", "--ignore-user-config", "-m", MODEL,
               "-c", 'model_reasoning_effort="high"', "-s", "read-only", "--json",
               "-o", str(folder / "response.txt"), "-"]
        with (folder / "events.jsonl").open("xb") as out, (folder / "stderr.txt").open("xb") as err:
            proc = await asyncio.create_subprocess_exec(*cmd, stdin=asyncio.subprocess.PIPE, stdout=out, stderr=err, cwd=ROOT)
            snapshot(folder / "process.json", dict(pid=proc.pid, started_at=now()))
            await proc.communicate(message.encode())
        tid, done, problem = None, False, []
        for line in (folder / "events.jsonl").read_text().splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("type") == "thread.started":
                tid = e["thread_id"]
            if e.get("type") == "turn.completed":
                done = True
            if e.get("type") in ("error", "turn.failed"):
                problem.append(e)
        try:
            usage = old.thread_usage(tid) if tid else dict(usage_unknown=True, error="no_thread")
        except Exception as exc:
            usage = dict(usage_unknown=True, error=str(exc), thread_id=tid)
        freeze(folder / "usage.json", dict(**identity, usage=usage))
        success = proc.returncode == 0 and done and (folder / "response.txt").exists() and usage.get("actual_config_verified") and not usage.get("tool_call_count")
        status = "completed" if success else "call_failed"
        freeze(completed, dict(identity=identity, status=status, returncode=proc.returncode, thread_id=tid,
            finished_at=now(), errors=problem, response_sha256=digest(folder / "response.txt") if (folder / "response.txt").exists() else None))
        self.failures = 0 if success else self.failures + 1
        error_text = compact(problem).lower()
        if any(t in error_text for t in ("rate limit", "rate_limit", "usage limit", "quota", "429")):
            self.stop("rate_or_usage_limit")
        elif usage.get("model") and not usage.get("actual_config_verified"):
            self.stop("model_configuration_mismatch")
        elif any(t in error_text for t in ("model_not_found", "unsupported model", "model is not supported")):
            self.stop("model_unavailable")
        elif self.failures >= 3:
            self.stop("three_consecutive_infrastructure_failures")
        print(compact(dict(task_id=task_id, stage=stage, status=status, tokens=(usage.get("actual_usage") or {}).get("total_tokens"))), flush=True)
        if not success:
            raise CallFailed(status)
        return (folder / "response.txt").read_text()

    async def task(self, t):
        folder = RUN / "tasks" / t["task_id"].replace(":", "__")
        if (folder / "result.json").exists():
            return
        original, previous_rubric = read(t["original_ref"]), read(t["old_rubric_ref"])
        view, pack = d1.base.prepare_input(original)
        history, previous, feedback = [], None, None
        status = "pending"
        try:
            for revision in range(2):
                data = dict(input=view, old_rubric=previous_rubric, output_schema=d1.selection_schema(original, pack))
                if revision:
                    data.update(previous_output=previous, revision_feedback=feedback)
                raw = await self.call(t["task_id"], f"repair_{revision}", "repair", data)
                row = dict(revision=revision, status="validation_pending")
                history.append(row)
                previous = dict(invalid_raw=raw)
                try:
                    previous = d1.base.json_spans(raw)[0]
                    rubric = d1.compile_rubric(previous, original, pack)
                except (ValueError, TypeError, KeyError, IndexError) as exc:
                    status = "invalid_generation"
                    feedback = dict(kind="structure", error=str(exc), legal_source_ids=[s["id"] for s in original["sources"]])
                    row.update(status=status, feedback=feedback)
                    freeze(folder / f"round_{revision}_validation.json", row)
                    continue
                freeze(folder / f"selection_{revision}.json", previous)
                freeze(folder / f"rubric_{revision}.json", rubric)
                row["status"] = "review_pending"
                raw_judge = await self.call(t["task_id"], f"review_{revision}", "review", review_payload(original, previous, rubric))
                try:
                    judgment = d1.base.json_spans(raw_judge)[0]
                    evaluation = d1.base.evaluate(judgment, t["task_id"], {r["id"] for r in rubric["rubric"]}, {c["candidate_id"] for c in pack["candidates"]})
                except (ValueError, TypeError, KeyError, IndexError) as exc:
                    judgment = dict(invalid_raw=raw_judge)
                    evaluation = dict(status="invalid_judge", errors=[str(exc)], scores={})
                freeze(folder / f"judgment_{revision}.json", judgment)
                freeze(folder / f"evaluation_{revision}.json", evaluation)
                status = evaluation["status"]
                row.update(status=status, scores=evaluation.get("scores", {}), new_unresolved=len(rubric["unresolved"]), rubric_ref=str(folder / f"rubric_{revision}.json"))
                if status == "passed":
                    freeze(RUN / "accepted" / (t["task_id"].replace(":", "__") + ".json"), rubric)
                    break
                if status == "invalid_judge":
                    break
                feedback = dict(kind="independent_review", judgment=judgment)
        except Paused:
            snapshot(folder / "pending.json", dict(task_id=t["task_id"], rounds=history, at=now()))
            return
        except CallFailed as exc:
            status = "call_failed"
            history.append(dict(status=status, error=str(exc)))
        freeze(folder / "result.json", dict(task_id=t["task_id"], status=status, rounds=history,
            old_unresolved=t["old_unresolved"], finished_at=now()))

    async def run(self, m):
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self.stop, "user_or_process_stop")
        queue = asyncio.Queue()
        for t in m["tasks"]:
            queue.put_nowait(t)

        async def worker():
            while not self.stopped and not queue.empty():
                t = queue.get_nowait()
                try:
                    await self.task(t)
                    report()
                except Exception as exc:
                    self.stop("runner_error:" + type(exc).__name__ + ":" + str(exc))
                    raise
                finally:
                    queue.task_done()
        await asyncio.gather(*(worker() for _ in range(WORKERS)))


def report():
    m = read(RUN / "manifest.json")
    rows, accepted = [], []
    for t in m["tasks"]:
        folder = RUN / "tasks" / t["task_id"].replace(":", "__")
        result = read(folder / "result.json") if (folder / "result.json").exists() else dict(status="pending", rounds=[])
        row = dict(task_id=t["task_id"], status=result["status"], old_unresolved=t["old_unresolved"],
                   old_rubric_ref=t["old_rubric_ref"], rounds=result["rounds"], new_unresolved=None)
        if row["status"] == "passed":
            p = RUN / "accepted" / (t["task_id"].replace(":", "__") + ".json")
            new = read(p)
            row.update(new_unresolved=len(new["unresolved"]), new_rubric_ref=str(p),
                       all_two=all(v == 2 for v in result["rounds"][-1]["scores"].values()))
            accepted.append(row)
        rows.append(row)
    stages, unknown, used = {}, [], set()
    for p in sorted((RUN / "tasks").glob("*/*/usage.json")):
        u = read(p)
        details = u["usage"]
        tid = details.get("thread_id")
        if tid and tid in used:
            raise ValueError("duplicate_thread_usage")
        used.add(tid or str(p))
        if details.get("usage_unknown"):
            unknown.append(str(p))
            continue
        counts = stages.setdefault(u["stage"], dict(calls=0, **{k: 0 for k in old.USAGE_KEYS}))
        counts["calls"] += 1
        for k in old.USAGE_KEYS:
            counts[k] += details["actual_usage"].get(k, 0)
    totals = {k: sum(s[k] for s in stages.values()) for k in old.USAGE_KEYS}
    before = sum(r["old_unresolved"] for r in accepted)
    after = sum(r["new_unresolved"] for r in accepted)
    status_counts = dict(Counter(r["status"] for r in rows))
    summary = dict(sample_size=SIZE, status_counts=status_counts, completed=SIZE-status_counts.get("pending", 0),
        accepted=len(accepted), first_round_passed=sum(bool(r["rounds"]) and r["rounds"][0]["status"] == "passed" for r in rows),
        additional_revision_tasks=sum(any(x.get("revision") == 1 for x in r["rounds"]) for r in rows),
        all_two=sum(r["all_two"] for r in accepted), passed_with_ones=sum(not r["all_two"] for r in accepted),
        paired=dict(tasks=len(accepted), before_unresolved_entries=before, after_unresolved_entries=after,
            cleared_tasks=sum(r["new_unresolved"] == 0 for r in accepted),
            fewer_tasks=sum(r["new_unresolved"] < r["old_unresolved"] for r in accepted),
            equal_tasks=sum(r["new_unresolved"] == r["old_unresolved"] for r in accepted),
            more_tasks=sum(r["new_unresolved"] > r["old_unresolved"] for r in accepted)),
        conservative_full100=dict(before_unresolved_entries=sum(t["old_unresolved"] for t in m["tasks"]),
            after_unresolved_entries=sum(r["new_unresolved"] if r["status"] == "passed" else r["old_unresolved"] for r in rows),
            remaining_nonempty_tasks=SIZE-sum(r["new_unresolved"] == 0 for r in accepted)),
        usage=dict(totals=totals, stages=stages, unknown_usage_refs=unknown, known_stage_calls=sum(s["calls"] for s in stages.values()),
                   total_recorded_stage_calls=len(used), tokens_per_selected_task=totals["total_tokens"]/SIZE,
                   tokens_per_accepted=totals["total_tokens"]/len(accepted) if accepted else None),
        actual_runtime_unknown_measured=False, semantic_accuracy_measured=False,
        production_bank_modified=False, updated_at=now())
    snapshot(RUN / "summary.json", summary)
    snapshot(RUN / "results.json", rows)
    lines = ["# D1：已有未决项100题修复试跑", "", "本报告统计rubric未决事项，不是实际轨迹unknown率或人工准确率。", "",
        f"冻结100题；终态{summary['completed']}题；审查通过{len(accepted)}题（全2分{summary['all_two']}，含1分{summary['passed_with_ones']}）；状态：{status_counts}。",
        f"首轮通过{summary['first_round_passed']}，追加纠错{summary['additional_revision_tasks']}题。", "",
        f"通过稿同题配对：未决条数{before}→{after}；{summary['paired']['cleared_tasks']}题未决项清零。",
        f"全部100题保守口径（未通过/未完成保留旧稿）：未决条数{summary['conservative_full100']['before_unresolved_entries']}→{summary['conservative_full100']['after_unresolved_entries']}。",
        f"已知Luna总tokens {totals['total_tokens']:,}；已记录阶段调用{len(used)}次；未知用量{len(unknown)}次。缓存/思考已包含，不重复相加；主调度另计。", "",
        "未决项减少不等于问题已在真实运行中解决；运行条件化、条目拆并、规则完整性须结合具体稿件与独立评分。五维无0是沿用的质量门槛；含1分仍有待核验边界。历史题库未覆盖，未通过稿不替代旧稿。", "",
        f"[完整启动提示词]({(RUN/'prompts/run_100.txt').resolve()}) · [实际修复协议]({(RUN/'prompts/repair.txt').resolve()}) · [独立审查协议]({(RUN/'prompts/review.txt').resolve()})", "",
        "| 题目 | 状态 | 旧未决数 | 新未决数 | 版本 |", "|---|---|---:|---:|---|"]
    for r in rows:
        links = f"[旧稿]({r['old_rubric_ref']})"
        if r.get("new_rubric_ref"):
            links += f" / [新稿]({r['new_rubric_ref']})"
        lines.append(f"| {r['task_id']} | {r['status']} | {r['old_unresolved']} | {r['new_unresolved'] if r['new_unresolved'] is not None else '—'} | {links} |")
    (RUN / "report_zh.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "report", "verify"))
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "prepare":
            prepare()
        elif args.command == "verify":
            verify_manifest()
            print("frozen inputs, implementations and prompts verified")
        elif args.command == "run":
            m = verify_manifest()
            asyncio.run(Runner().run(m))
            print(compact(report()))
        else:
            print(compact(report()))


if __name__ == "__main__":
    main()
