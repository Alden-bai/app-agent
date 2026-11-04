from __future__ import annotations

import asyncio
import importlib.util
import importlib.metadata
from pathlib import Path

from .assets import validate_assets
from .common import AWM, ROOT, code_hash, digest, now, read_json, save_json
from .config import validate_config
from .oracle import NeedsReview, evaluate, load_spec
from .runtime import execute_case, initialize
from .verification import execute_verifier, judge_sql, verifier_entry


async def preflight(config, *, offline=False):
    modules = ["fastapi", "fastapi_mcp", "sqlalchemy", "mcp", "mcp_agent", "openai", "dotenv", "uvicorn"]
    dependencies = {name: importlib.util.find_spec(name) is not None for name in modules}
    validate_config(config)
    result = {"checked_at": now(), "dependencies": dependencies, "llm_config": config.public(),
              "llm_status": "not_tested"}
    if not all(dependencies.values()):
        raise RuntimeError("Missing dependencies: " + ", ".join(n for n, ok in dependencies.items() if not ok))
    if not offline:
        from .verification import LoggedClient
        import sys
        sys.path.insert(0, str(AWM)) if str(AWM) not in sys.path else None
        from awm.llm_compat import completion_token_limit
        client = LoggedClient(config)
        try:
            response = await client.create(model=config.model, messages=[{"role": "user", "content": "Reply with OK."}],
                                           **completion_token_limit(config.base_url, 64))
            result["llm_status"] = "reachable" if response.choices and response.choices[0].message.content else "empty_response"
            result["usage"] = response.usage.model_dump() if response.usage else None
        except Exception as e:
            result["llm_status"] = "failed"
            result["error"] = config.redact(f"{type(e).__name__}: {e}")
        finally:
            await client.client.close()
    return result


def _native_hash():
    paths = [AWM / "awm" / p for p in ("core/verify.py", "core/server.py", "tools.py", "llm_compat.py")]
    return digest({str(p.relative_to(AWM)): digest(p) for p in paths if p.exists()})


async def run(run_dir, config, *, tasks=None, mode="both", limit=None, judge_enabled=True, reuse_no_action=False):
    run_dir = Path(run_dir).resolve()
    manifest = read_json(run_dir / "manifest.json")
    validate_assets(run_dir, manifest)
    modes = ["code", "sql"] if mode == "both" else [mode]
    if "sql" in modes and judge_enabled:
        validate_config(config)
        preflight_file = run_dir / "preflight.json"
        prior = read_json(preflight_file) if preflight_file.exists() else {}
        if prior.get("llm_status") != "reachable" or prior.get("llm_config") != config.public():
            prior = await preflight(config)
            save_json(preflight_file, prior)
            if prior["llm_status"] != "reachable":
                raise RuntimeError("LLM preflight failed; no batch Judge requests were submitted")
    selected = [t for t in manifest["tasks"] if not tasks or f"{t['scenario']}:{t['task_idx']}" in tasks]
    if tasks and len(selected) != len(set(tasks)):
        raise ValueError("One or more requested tasks are outside the frozen sample")
    selected = selected[:limit] if limit is not None else selected
    implementation, native = code_hash(), _native_hash()
    dependencies = {n: importlib.metadata.version(n) for n in
                    ("fastapi", "fastapi-mcp", "mcp", "mcp-agent", "sqlalchemy", "pydantic", "openai")}
    states = read_json(run_dir / "task_states.json") if (run_dir / "task_states.json").exists() else {}
    circuit = {}
    for task in selected:
        key = f"{task['scenario']}:{task['task_idx']}"
        taskdir = run_dir / "tasks" / task["scenario"] / str(task["task_idx"])
        taskdir.mkdir(parents=True, exist_ok=True)
        if key in states:
            save_json(taskdir / 'state_history' / (digest(states[key]) + '.json'), states[key])
        state = {**task, "updated_at": now(), "status": "pending", "results": [], "rule_status": "missing"}
        initial = run_dir / "initials" / (task["scenario"] + ".db")
        try:
            if not initial.exists():
                initialize(run_dir, task["scenario"], initial)
            init_evidence = read_json(initial.with_suffix(".initialization.json"))
            if init_evidence.get("database_sha256") != digest(initial):
                raise ValueError("Frozen initial database hash missing or changed")
            if init_evidence["errors"] or init_evidence["foreign_key_check"] or init_evidence["integrity_check"] != [["ok"]]:
                state.update(status="runtime_blocked", reason="official_initialization_invalid",
                             evidence=str(initial.with_suffix(".initialization.json").relative_to(run_dir)))
                # Still try to compile independent rules, but never score a
                # case on a partially built database.
                try:
                    spec = load_spec(task["scenario"], task["task_idx"], initial)
                    blocked_rule = taskdir / 'rules' / (digest(spec.frozen()) + '.blocked.json')
                    save_json(blocked_rule, {'task': task, 'spec': spec.frozen(), 'execution_blocked': True,
                                            'reason': 'Official initial database failed integrity/insert checks; rule is unexecuted'})
                    state['rule_path'] = str(blocked_rule.relative_to(run_dir))
                    state["rule_status"] = spec.status
                except Exception as e:
                    state["rule_error"] = str(e)
                continue
            # Construct and freeze the oracle BEFORE any official verifier read.
            spec = load_spec(task["scenario"], task["task_idx"], initial)
            frozen = spec.frozen()
            frozen_hash = digest(frozen)
            save_json(taskdir / "rules" / (frozen_hash + ".json"), {"task": task, "rule_hash": frozen_hash,
                       "initial_sha256": digest(initial), "basis": "task + initial database + tools; excludes official verifier", "spec": frozen})
            state.update(rule_status=spec.status, rule_hash=frozen_hash,
                         rule_path=str((taskdir / "rules" / (frozen_hash + ".json")).relative_to(run_dir)))
            if spec.status != "ready":
                state.update(status="confirmed_defect" if spec.status == "confirmed_defect" else "needs_review",
                             reason=spec.notes, evidence=spec.evidence)
                # A stronger diagnosis does not erase earlier real executions
                # or their independently checked incorrect cases.
                if key in states:
                    state['results'] = states[key].get('results', [])
                    state['prior_rule_path'] = states[key].get('rule_path')
                continue
            fingerprint = digest({"implementation": implementation, "native": native,
                                  "rule": frozen_hash, "initial": digest(initial), "assets": manifest["files"],
                                  "llm": config.public() if "sql" in modes else None, "modes": modes,
                                  "judge_enabled": judge_enabled})
            fingerprint = digest([fingerprint, dependencies])
            state["attempt"] = fingerprint
            attempt = taskdir / "attempts" / fingerprint
            attempt.mkdir(parents=True, exist_ok=True)
            has_positive = False
            for case in spec.cases():
                folder = attempt / case["name"]
                result_file = folder / "evaluation.json"
                if result_file.exists():
                    result = read_json(result_file)
                    if result.get("fingerprint") != fingerprint:
                        raise ValueError("Cache fingerprint mismatch")
                else:
                    if reuse_no_action:
                        from .reuse import reuse_case
                        reused = reuse_case(run_dir, taskdir, states.get(key), spec, case, initial, config, modes, fingerprint)
                        if reused is not None:
                            save_json(result_file, reused)
                            state['results'].append(str(result_file.relative_to(run_dir)))
                            has_positive |= reused.get('oracle',{}).get('correct') is True
                            print(f"{key} {case['name']}: preserved original decision", flush=True)
                            continue
                    # An interrupted attempt is retained, never overwritten.
                    if folder.exists():
                        folder.rename(folder.with_name(folder.name + ".interrupted-" + now().replace(":", "")))
                    trajectory = await execute_case(run_dir, task, case, initial, folder)
                    result = {"fingerprint": fingerprint, "case": case["name"], "intended_label": case["expected"],
                              "task": key, "modes": {}, "path": str(folder.relative_to(run_dir)),
                              "llm_config": config.public() if "sql" in modes else None}
                    if trajectory["failure"]:
                        result.update(status="runtime_blocked", failure=trajectory["failure"])
                    else:
                        oracle = evaluate(spec, folder / "initial.db", folder / "final.db", trajectory["final_answer"], trajectory["trajectory"])
                        result["oracle"] = oracle
                        if oracle["status"] != "valid":
                            result["status"] = "oracle_unknown"
                        elif oracle["correct"] != case["expected"]:
                            result["status"] = "reference_failed" if case["expected"] else "negative_not_wrong"
                        else:
                            result["status"] = "valid"
                            for m in modes:
                                try:
                                    entry = verifier_entry(run_dir, task, m)
                                except ValueError as e:
                                    result["modes"][m] = {"status": "asset_error", "accepted": None, "error": str(e)}
                                    continue
                                check = execute_verifier(entry, m, folder)
                                if m == "sql" and check["status"] == "valid":
                                    result["sql_extraction"] = check
                                    # Explicitly unknown until the independent
                                    # evidence review is supplied. Judge accuracy
                                    # is measured separately from extraction quality.
                                    result["sql_evidence_review"] = {"status": "needs_review", "path": "sql_extraction"}
                                    check = await judge_sql(config, task, check["details"], trajectory["trajectory"], folder, circuit=circuit) if judge_enabled else {"status": "not_tested", "accepted": None}
                                result["modes"][m] = check
                    save_json(result_file, result)
                has_positive |= result.get("status") == "valid" and result.get("oracle", {}).get("correct") is True
                state["results"].append(str(result_file.relative_to(run_dir)))
                print(f"{key} {case['name']}: {result['status']} " + " ".join(f"{m}={r.get('accepted')}({r['status']})" for m, r in result.get("modes", {}).items()), flush=True)
            state["status"] = "completed_by_tools" if has_positive else "needs_review"
            if not has_positive:
                state["reason"] = "Reference did not establish a correct solution; not proof of unsolvability"
        except (ModuleNotFoundError, NeedsReview) as e:
            state.update(status="needs_review", reason=str(e), rule_status="needs_review")
            diagnostic = taskdir / "rule_needs_review.json"
            save_json(diagnostic, {"task": task, "status": "needs_review", "independent_target_error": str(e),
                                   "correctness_established": False, "basis": "task + initial database + tools"})
            state["rule_path"] = str(diagnostic.relative_to(run_dir))
        except Exception as e:
            state.update(status="runtime_blocked", reason=f"{type(e).__name__}: {config.redact(e)}")
        finally:
            states[key] = state
            save_json(taskdir / "state.json", state)
            save_json(run_dir / "task_states.json", states)
    return states
