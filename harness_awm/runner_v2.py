"""Auditable v2 execution; v1 remains available for frozen experiments."""
import asyncio
import time
from pathlib import Path

from .backend import APIBackend, BudgetExceeded
from .contracts import EpisodeResult
from .evaluation import apply_evaluation, evaluate_episode
from .io import Recorder, canonical, digest, seal, write_json
from .protocol import ProtocolError, estimate_tokens, parse_action, validate_call
from .protocol_v2 import (DISCOVERY_ACTIONS, PROTOCOL, SYSTEM_PROMPT, ObservationsV2,
                          ToolCatalog, WorkingContext, context_limit, fits_context)


def final_messages(context, input_limit):
    """Bound the protected final request; all omitted evidence remains addressable."""
    directive = {"role": "user", "content": canonical({"source": "harness", "mode": "finalize_only",
        "instruction": "Return finish now. No tools are allowed. Report only supported results; identify omissions and blockers. Referenced evidence you cannot see is not proof of success."})}
    ledger, _ = context.observations.present(context.ledger)
    messages = [*context.base, {"role": "user", "content": canonical({"source": "harness", "execution_ledger": ledger})}, directive]
    if estimate_tokens(messages) > input_limit:
        oid, _ = context.observations.register(context.ledger)
        messages = [*context.base, {"role": "user", "content": canonical({"source": "harness",
            "execution_ledger_ref": oid, "rounds": len(context.rounds), "evidence_omitted": True})}, directive]
    recent = []
    for pair in reversed(context.rounds[-context.recent:]):
        proposal = [*messages[:-1], *pair, *recent, directive]
        if estimate_tokens(proposal) > input_limit:
            break
        recent = pair + recent
    return [*messages[:-1], *recent, directive]


async def run_episode(runner, task, folder, rollout_id, backend, *, evaluate_after=True):
    config = runner.config
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    recorder = Recorder(folder, rollout_id)
    env = runner.env_factory(task, config, folder)
    observations = ObservationsV2(recorder, config.observation_chars, config.max_read_chars)
    base = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task.instruction}]
    context = WorkingContext(base, observations, config.context_recent_turns)
    full_messages = list(base)
    write_json(folder / "task.json", task.model_copy(update={"protocol_id": PROTOCOL}).model_dump())
    write_json(folder / "protocol.json", {"protocol_id": PROTOCOL, "system_prompt": SYSTEM_PROMPT,
                                          "config": config.model_dump()})
    trace, answer, completion = [], "", None
    reason, origin, integrity, terminal = "runtime_error", "runtime", "valid", "unavailable"
    phase, error, finished, quiet = "preparing", None, False, False
    started = time.monotonic()
    turns = calls = accounted = reported = estimated = unknown_usage = 0
    corrections = malformed = repetitions = duplicate_discovery = 0
    zero_finish_corrected = False
    previous_signature = None
    discovery_tokens = discovery_turns = 0
    first_call = None
    queue_seconds = model_seconds = tool_seconds = 0.0
    finalizing, final_trigger, final_attempted = False, None, False
    limit = context_limit(config)
    margin = max(512, config.safety_margin)
    final_output = min(config.final_output_tokens, config.max_output_tokens)
    final_input = min(16384, limit - final_output - margin)
    final_reserve = max(0, final_input) + final_output + 512
    hold_key = rollout_id + "/finalize"
    uses_budget = isinstance(backend, APIBackend)
    recorder.emit("lifecycle", state="CREATED")

    def enter_final(trigger):
        nonlocal finalizing, final_trigger
        if not finalizing:
            finalizing, final_trigger = True, trigger
            recorder.emit("finalization_started", trigger=trigger)

    try:
        recorder.emit("lifecycle", state="PREPARING")
        await env.prepare()
        catalog = ToolCatalog(env.tools, config.observation_chars)
        startup = {"source": "harness", "tool_index": catalog.page(names_only=True)}
        base.append({"role": "user", "content": canonical(startup)})
        full_messages.append(base[-1])
        # Do not give undisplayed names an implicit discovery claim.
        known_tools = {t["name"]: catalog.tools[t["name"]] for t in startup["tool_index"]["tools"]}
        recorder.emit("environment_ready", tools_hash=digest(env.tools), isolation=config.isolation)
        recorder.emit("observation_delivered", "model_visible", 0,
                      visible_result_ref=recorder.blob(startup, "model_visible"), truncated=False)
        if uses_budget:
            runner.budget.hold(hold_key, final_reserve)
        recorder.emit("lifecycle", state="RUNNING")
        for turn in range(1, config.max_turns + 1):
            phase = "planning"
            remaining = config.episode_timeout - (time.monotonic() - started)
            if remaining <= 0:
                reason = "wall_clock_limit"
                break
            if turn == config.max_turns:
                enter_final("max_turns")
            elif calls >= config.max_tool_calls:
                enter_final("tool_call_limit")
            elif remaining <= config.final_timeout:
                enter_final("wall_clock_limit")
            messages = context.build()
            available = limit - config.max_output_tokens - margin
            if estimate_tokens(messages) >= available * config.context_compact_ratio:
                before = estimate_tokens(messages)
                messages = context.build(compact=True)
                recorder.emit("context_compacted", before_estimate=before, after_estimate=estimate_tokens(messages))
            if not fits_context(messages, config.max_output_tokens, limit, margin):
                enter_final("context_limit")
            if accounted + estimate_tokens(messages) + config.max_output_tokens + 512 + final_reserve > config.episode_token_limit:
                enter_final("token_limit")
            output_tokens = config.max_output_tokens
            if finalizing:
                messages = final_messages(context, final_input)
                output_tokens = final_output
                if estimate_tokens(messages) > final_input or not fits_context(messages, output_tokens, limit, margin):
                    reason = "context_limit"
                    break
                if accounted + estimate_tokens(messages) + output_tokens + 512 > config.episode_token_limit:
                    reason = "token_limit"
                    break
            count = estimate_tokens(messages)
            turns = turn
            turn_dir = folder / "model_turns" / f"{turn:04d}"
            write_json(turn_dir / "input.json", {"messages": messages, "input_token_estimate": count,
                "estimator": "utf8_serialized_bytes_plus_128", "reserved_output": output_tokens,
                "finalize_only": finalizing})
            recorder.emit("model_request", "model_visible", turn,
                          input_ref=recorder.blob(messages, "model_visible"), finalize_only=finalizing)
            phase = "model_request"
            request_started = time.monotonic()
            if finalizing:
                final_attempted = True
            try:
                timeout = min(remaining, config.final_timeout) if finalizing else max(.001, remaining - config.final_timeout)
                async with asyncio.timeout(timeout):
                    if isinstance(backend, APIBackend):
                        response = await backend.generate(messages, turn_dir, turn, output_tokens=output_tokens,
                                                          final_hold=hold_key if finalizing else None)
                    else:
                        response = await backend.generate(messages, turn_dir, turn)
            except BudgetExceeded:
                recorder.emit("model_not_dispatched", turn_id=turn, cause="global_budget_limit")
                if finalizing:
                    raise
                enter_final("global_budget_limit")
                continue
            except TimeoutError:
                # Only a model request can be retried as a final request, never a business call.
                record = runner.budget.request_record(f"{rollout_id}/actor/{turn}") if uses_budget else None
                reservation = record["reserved"] if record else (0 if uses_budget else count + output_tokens + 512)
                unknown_usage += int(reservation > 0)
                accounted += reservation
                estimated += reservation
                request_phase = getattr(backend, "phase", "model_request")
                if request_phase == "model_queue":
                    queue_seconds += time.monotonic() - request_started
                else:
                    model_seconds += time.monotonic() - request_started
                recorder.emit("model_timeout", turn_id=turn, token_reservation=reservation, phase=request_phase)
                if finalizing:
                    reason = "wall_clock_limit"
                    break
                enter_final("model_timeout")
                continue
            except BaseException:
                record = runner.budget.request_record(f"{rollout_id}/actor/{turn}") if uses_budget else None
                if record:
                    accounted += record["reserved"]
                    estimated += record["reserved"]
                    unknown_usage += int(record["status"] != "response")
                phase = getattr(backend, "phase", "model_request")
                if phase == "model_queue":
                    queue_seconds += time.monotonic() - request_started
                else:
                    model_seconds += time.monotonic() - request_started
                recorder.emit("model_interrupted", turn_id=turn, phase=phase,
                              reservation=record, elapsed_seconds=time.monotonic() - request_started)
                raise
            model_seconds += response.get("model_seconds", time.monotonic() - request_started)
            queue_seconds += response.get("queue_seconds", 0)
            write_json(turn_dir / "output.json", response)
            recorder.emit("model_response", "model_visible", turn,
                          raw_model_output_ref=recorder.blob(response, "model_visible"),
                          queue_seconds=response.get("queue_seconds", 0),
                          model_seconds=response.get("model_seconds", time.monotonic() - request_started))
            raw = response.get("content") or ""
            usage = (response.get("usage") or {}).get("total_tokens")
            if type(usage) is int and usage >= 0:
                cost = usage
                reported += usage
            else:
                cost = count + max(output_tokens, len(raw.encode())) + 512
                estimated += cost
                unknown_usage += 1
            accounted += cost
            item = {"iteration": turn, "role": "assistant", "content": raw, "tool_calls": []}
            trace.append(item)
            full_messages.append({"role": "assistant", "content": raw})
            action, status = None, "response_received"
            try:
                if response.get("finish_reason") != "stop":
                    raise ProtocolError("model_output_incomplete")
                action = parse_action(raw, PROTOCOL)
                kind = action["type"]
                recorder.emit("action_parsed", "model_visible", turn, parsed_action=action)
                if finalizing and kind != "finish":
                    raise ProtocolError("finalize_only_requires_finish")
                if kind == "finish":
                    if calls == 0 and not zero_finish_corrected and not finalizing:
                        zero_finish_corrected = True
                        corrections += 1
                        rendered = {"source": "harness", "correction": "No business tool has been called. Search the complete catalog and read schemas, then execute, or finish again with a specific blocker. Do not make irrelevant calls."}
                        recorder.emit("early_finish_corrected", turn_id=turn)
                    else:
                        answer, completion = action["answer"], action["status"]
                        item.update(content=answer, is_final=True, raw_action=raw)
                        reason, origin, finished = "finish", "model", True
                        break
                elif kind in DISCOVERY_ACTIONS:
                    discovery_turns += 1
                    discovery_tokens += cost
                    if kind in {"list_tools", "search_tools"}:
                        returned = catalog.page(action.get("query"), action.get("cursor"), names_only=kind == "list_tools")
                        known_tools.update({t["name"]: catalog.tools[t["name"]] for t in returned["tools"]})
                        rendered = {"source": "harness", **returned, "truncated": False}
                    elif kind == "get_tool_schema":
                        schema = catalog.schema(action["name"])
                        known_tools[action["name"]] = schema
                        visible, _ = observations.present(schema)
                        context.selected_schema = visible
                        rendered = {"source": "harness", **visible}
                    else:
                        rendered = {"source": "harness", **observations.read(action)}
                    signature = canonical(action)
                    repetitions = repetitions + 1 if signature == previous_signature else 1
                    duplicate_discovery += int(repetitions > 1)
                    previous_signature = signature
                    if repetitions == 3:
                        corrections += 1
                        rendered["progress_warning"] = "No new information. Change query, follow the next cursor/offset, fetch a schema, or execute a relevant tool."
                    elif repetitions >= 4:
                        enter_final("stalled")
                else:
                    validate_call(action, known_tools)
                    calls += 1
                    previous_signature, repetitions = None, 0
                    if first_call is None:
                        first_call = {"turn": turn, "seconds": time.monotonic() - started, "tokens": accounted}
                    tc_id = f"{rollout_id}_{turn}"
                    item["tool_calls"] = [{"id": tc_id, "name": "call_tool", "arguments": {
                        "tool_name": action["name"], "arguments": action["arguments"]}}]
                    before = env.audit(f"{turn:04d}_before")
                    call_id = recorder.emit("tool_started", turn_id=turn, tool_name=action["name"],
                        arguments_ref=recorder.blob(action["arguments"], "model_visible"),
                        before_state_ref=before, commit_status="outcome_unknown")
                    phase = "business_call"
                    tool_started = time.monotonic()
                    try:
                        remaining = config.episode_timeout - (time.monotonic() - started)
                        async with asyncio.timeout(max(.001, remaining - config.final_timeout)):
                            returned = await env.call(action["name"], action["arguments"])
                    except BaseException as exc:
                        integrity = "outcome_unknown"
                        item["tool_response"] = {"tool_call_id": tc_id, "content": canonical({"source": "harness",
                            "transport_error": type(exc).__name__, "commit_status": "outcome_unknown"})}
                        recorder.emit("tool_outcome_unknown", turn_id=turn, call_event_id=call_id)
                        raise
                    finally:
                        tool_seconds += time.monotonic() - tool_started
                    recorder.emit("tool_completed", "runtime_returned", turn, call_event_id=call_id,
                        raw_result_ref=recorder.blob(returned, "runtime_returned"), transport_status="response",
                        mcp_is_error=returned.get("isError", False), commit_status="response_received",
                        before_state_ref=before, after_state_ref=env.audit(f"{turn:04d}_after"),
                        tool_seconds=time.monotonic() - tool_started)
                    visible, _ = observations.present(returned)
                    rendered = {"source": "awm", **visible}
                malformed = 0
            except ProtocolError as exc:
                status = "protocol_error"
                malformed += 1
                corrections += 1
                rendered = {"source": "harness", "protocol_error": str(exc),
                            "instruction": "Correct the JSON action or use discovery to obtain the exact tool schema."}
                item["protocol_error"] = str(exc)
                recorder.emit("protocol_error", "model_visible", turn, error=str(exc))
                if malformed > 2:
                    enter_final("protocol_error_limit")
            if action and action["type"] != "finish":
                tc_id = f"{rollout_id}_{turn}"
                args = {k: v for k, v in action.items() if k != "type"}
                if action["type"] == "call_tool":
                    args = {"tool_name": action["name"], "arguments": action["arguments"]}
                item.update(tool_calls=[{"id": tc_id, "name": action["type"], "arguments": args}],
                            tool_response={"tool_call_id": tc_id, "content": canonical(rendered)})
            recorder.emit("observation_delivered", "model_visible", turn,
                          visible_result_ref=recorder.blob(rendered, "model_visible"),
                          truncated=rendered.get("truncated", False))
            context.add(raw, rendered, action, status)
            full_messages.append({"role": "user", "content": canonical(rendered)})
            if final_attempted:
                reason = final_trigger or "finalization_failed"
                break
        else:
            reason = final_trigger or "max_turns"
    except BudgetExceeded as exc:
        reason, error = "global_budget_limit", str(exc)
    except asyncio.CancelledError as exc:
        reason, error = "cancelled", "cancelled"
        origin = str(exc.args[0]) if exc.args and str(exc.args[0]) in {"user", "scheduler", "shutdown"} else "unknown"
        if phase == "model_request":
            phase = getattr(backend, "phase", phase)
    except TimeoutError:
        reason = "tool_timeout" if integrity == "outcome_unknown" else "wall_clock_limit"
        error = reason
    except Exception as exc:
        reason, error = "runtime_error", f"{type(exc).__name__}: {exc}"
        if integrity == "valid":
            integrity = "runtime_error"
    finally:
        termination_phase = phase
        if uses_budget:
            runner.budget.release(hold_key)
        recorder.emit("lifecycle", state="STOPPING", termination_reason=reason,
                      termination_origin=origin, termination_phase=termination_phase)
        phase = "cleanup"
        try:
            quiet = await env.close()
            if getattr(env, "cleanup_cancellation", None):
                reason, origin, termination_phase = "cancelled", env.cleanup_cancellation, "cleanup"
                finished = False
        except (Exception, asyncio.CancelledError) as exc:
            quiet, error = False, f"quiescence_failed: {type(exc).__name__}"
            if isinstance(exc, asyncio.CancelledError):
                reason, origin, termination_phase = "cancelled", "unknown", "cleanup"
                finished = False
        recorder.emit("environment_stopped", quiescent=quiet)
        if quiet and env.started:
            try:
                final_hash = env.freeze()
                terminal = "trusted"
                recorder.emit("snapshot_created", final_hash=final_hash)
            except Exception as exc:
                terminal, error = "untrusted", f"snapshot_failed: {exc}"
        elif not quiet:
            terminal, integrity = "untrusted", "outcome_unknown"
    report = {"source": "harness", "termination_reason": reason, "termination_origin": origin,
              "termination_phase": termination_phase, "completion_status": completion,
              "model_finished": finished, "business_calls": calls, "execution_integrity": integrity,
              "finalization_trigger": final_trigger, "finalization_attempted": final_attempted,
              "finalization_succeeded": final_attempted and finished, "error": error}
    trajectory = {"scenario": task.scenario, "task_id": task.native_task_id, "task": task.instruction,
        "trajectory": trace, "messages": full_messages, "final_answer": answer, "finished": finished,
        "completion_status": completion, "protocol_id": PROTOCOL, "harness_termination_report": report}
    write_json(folder / "trajectory.json", trajectory)
    write_json(folder / "termination_report.json", report)
    execution_metrics = {"first_business_call": first_call, "zero_business_calls": calls == 0,
        "empty_model_answer": not answer.strip(), "discovery_turns": discovery_turns,
        "discovery_tokens": discovery_tokens, "discovery_token_fraction": discovery_tokens / accounted if accounted else None,
        "duplicate_discovery_requests": duplicate_discovery, "corrections": corrections,
        "reported_tokens": reported, "estimated_tokens": estimated, "unknown_usage_requests": unknown_usage,
        "queue_seconds": queue_seconds, "model_seconds": model_seconds, "tool_seconds": tool_seconds,
        "finalization_trigger": final_trigger, "finalization_attempted": final_attempted,
        "finalization_succeeded": final_attempted and finished}
    result = EpisodeResult(rollout_id=rollout_id, task_instance_id=task.task_instance_id,
        termination_reason=reason, termination_origin=origin, termination_phase=termination_phase,
        completion_status=completion, execution_integrity=integrity, terminal_state_status=terminal,
        turns=turns, tool_calls=calls, execution_metrics=execution_metrics,
        test_only=config.actor == "scripted" or config.judge == "mock", error=error).model_dump()
    recorder.emit("execution_closed", termination_reason=reason, execution_integrity=integrity,
                  elapsed_seconds=time.monotonic() - started, token_count_or_estimate=accounted)
    write_json(folder / "episode_result.json", result)
    seal(folder)
    if terminal == "trusted" and evaluate_after and reason != "cancelled":
        evaluation = await evaluate_episode(folder, task, config, runner.budget, runner.judge_semaphore)
        result = apply_evaluation(folder, evaluation)
    return result
