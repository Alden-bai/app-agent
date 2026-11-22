import asyncio
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from harness_awm.backend import APIBackend, Budget, BudgetExceeded, ScriptedBackend
from harness_awm.contracts import RunConfig
from harness_awm.environment import complete_tool_catalog
from harness_awm.io import Recorder, digest, read_json, rows, snapshot, verify_seal, write_json
from harness_awm.protocol import ProtocolError, estimate_tokens, parse_action
from harness_awm.protocol_v2 import PROTOCOL, ObservationsV2, ToolCatalog, WorkingContext
from harness_awm.runner import EpisodeRunner
from harness_support import fixture


def finish(answer="done", status="completed", reason=""):
    return dict(type="finish", answer=answer, status=status, reason=reason)


def call(name="write_value", **arguments):
    return dict(type="call_tool", name=name, arguments=arguments)


TOOLS = [{"name": "write_value", "description": "Write a value", "inputSchema": {
    "type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"], "additionalProperties": False}},
    {"name": "read_state", "description": "Read the value", "inputSchema": {"type": "object"}}]


class ProtocolV2Tests(unittest.TestCase):
    def test_validation_and_v1_compatibility(self):
        self.assertEqual(parse_action('{"type":"finish","answer":""}')["answer"], "")
        for value in [finish(""), finish(" "), finish(status="blocked"), finish(status=[]),
                      {"type": "search_tools", "query": ""}, {"type": "list_tools", "cursor": 0},
                      {"type": "finish", "answer": "legacy"}]:
            with self.subTest(value=value), self.assertRaises(ProtocolError):
                parse_action(json.dumps(value), PROTOCOL)
        self.assertEqual(parse_action(json.dumps(finish(status="blocked", reason="permission required")), PROTOCOL)["status"], "blocked")

    def test_large_catalog_tail_search_and_pagination(self):
        tools = [{"name": f"tool_{i:03d}", "description": "long " * 100,
                  "inputSchema": {"type": "object"}} for i in range(200)] + TOOLS
        self.assertGreater(len(json.dumps(tools)), 50000)
        cat = ToolCatalog(tools, 1500)
        names, cursor = [], None
        while True:
            page = cat.page(cursor=cursor, names_only=True)
            names.extend(t["name"] for t in page["tools"])
            cursor = page["next_cursor"]
            if cursor is None:
                break
        self.assertEqual(names, sorted(t["name"] for t in tools))
        self.assertEqual(cat.page("write value")["tools"][0]["name"], "write_value")
        self.assertEqual(cat.schema("write_value"), TOOLS[0])
        overlap = ToolCatalog([{"name": n, "description": "", "inputSchema": {}} for n in ['bulk_update_task', 'update_task']])
        self.assertEqual(overlap.page('update_task')['tools'][0]['name'], 'update_task')
        self.assertEqual(cat.page("zxcvbnm")["total_count"], 0)
        self.assertTrue(cat.page("zxcvbnm")["zero_matches_is_not_capability_absence"])
        cursor = cat.page(names_only=True)["next_cursor"]
        with self.assertRaises(ProtocolError):
            cat.page("different", cursor)

    def test_stable_scoped_observations_escaped_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            obs = ObservationsV2(Recorder(tmp, "one"), 1024, 10000)
            value = {"text": ('\\\"中\n' * 10000)}
            page, _ = obs.present(value)
            again, _ = obs.present(value)
            self.assertEqual(page["observation_id"], again["observation_id"])
            oid, output, start = page["observation_id"], "", 0
            while True:
                page = obs.read({"observation_id": oid, "offset": start, "limit": 999999})
                self.assertLessEqual(len(json.dumps({"source": "harness", **page}, ensure_ascii=False, separators=(',', ':')).encode()), 1024)
                output += page["content"]
                if not page["has_more"]:
                    break
                self.assertGreater(page["next_offset"], start)
                start = page["next_offset"]
            self.assertEqual(json.loads(output), value)
            self.assertEqual(len(obs.allowed), 1)
            other = ObservationsV2(Recorder(Path(tmp)/"other", "two"), 1024, 100)
            with self.assertRaises(ProtocolError):
                other.read({"observation_id": oid, "offset": 0, "limit": 20})

    def test_compaction_preserves_references_and_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            obs = ObservationsV2(Recorder(tmp, "one"), 1024, 100)
            base = [{"role": "system", "content": "rules"}, {"role": "user", "content": "original task"}]
            context = WorkingContext(base, obs)
            for i in range(12):
                visible, _ = obs.present({"value": str(i) * 2000})
                context.add(json.dumps(call(value=i)), visible, call(value=i))
            before = context.build()
            after = context.build(compact=True)
            self.assertLess(estimate_tokens(after), estimate_tokens(before))
            self.assertEqual(after[:2], base)
            oid = context.ledger[0]["observation_ref"]
            self.assertEqual(json.loads(obs.allowed[oid]), {"value": "0" * 2000})
            self.assertEqual(after[-8:], before[-8:])


class PaginationTests(unittest.IsolatedAsyncioTestCase):
    async def test_mcp_pages_and_cycle(self):
        class Tool:
            def __init__(self, name): self.name = name
            def model_dump(self, **kwargs): return {"name": self.name}
        class Client:
            async def list_tools(self, cursor=None):
                return SimpleNamespace(tools=[Tool("b" if cursor else "a")], nextCursor=None if cursor else "next")
        self.assertEqual(await complete_tool_catalog(Client()), [{"name": "a"}, {"name": "b"}])
        class Cycle:
            async def list_tools(self, cursor=None):
                return SimpleNamespace(tools=[], nextCursor="same")
        with self.assertRaisesRegex(ValueError, "cursor_cycle"):
            await complete_tool_catalog(Cycle())
        class Duplicate:
            async def list_tools(self, cursor=None):
                return SimpleNamespace(tools=[Tool("a")], nextCursor="next")
        with self.assertRaisesRegex(ValueError, "duplicate_tool"):
            await complete_tool_catalog(Duplicate())


class BudgetHoldTests(unittest.TestCase):
    def test_hold_prevents_other_requests_stealing_final_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = Budget(Path(tmp)/"budget.db", 2, 5000)
            b.hold("a/final", 2000)
            rid = b.reserve("a/1", "actor", [], 10)
            b.finish(rid, {"total_tokens": 50})
            with self.assertRaises(BudgetExceeded):
                b.reserve("b/1", "actor", [], 10)
            rid = b.reserve("a/final", "actor", [], 10, consume_hold="a/final")
            b.finish(rid, {"total_tokens": 50})
            self.assertEqual(b.summary()["physical_requests"], 2)
            self.assertEqual(b.summary()["held_requests"], 0)

    def test_holds_share_token_cap_and_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = Budget(Path(tmp)/"budget.db", 10, 3000)
            b.hold("a", 2500)
            with self.assertRaises(BudgetExceeded): b.hold("b", 1000)
            with self.assertRaises(BudgetExceeded): b.reserve("b", "actor", [], 1)
            b.release("a")
            b.reserve("b", "actor", [], 1)


class MemoryEnv:
    def __init__(self, task, config, folder):
        self.task, self.folder = task, folder
        self.started, self.tools, self.calls = False, TOOLS, []
    async def prepare(self):
        snapshot(self.task.initial_state_ref, self.folder/"initial.db")
        snapshot(self.task.initial_state_ref, self.folder/"working.db")
        write_json(self.folder/"tool_catalog.json", {"tools": self.tools, "hash": digest(self.tools)})
        self.started = True
    async def call(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "write_value":
            with sqlite3.connect(self.folder/"working.db") as db:
                db.execute("update items set value=?", (arguments["value"],))
        return {"isError": False, "content": [{"type": "text", "text": "value=80"}]}
    def audit(self, label): return {"test_snapshot": label}
    async def close(self): return True
    def freeze(self): return snapshot(self.folder/"working.db", self.folder/"final.db")


class RunnerV2Tests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.task = fixture(self.root/"assets")
        self.cfg = RunConfig(protocol_id=PROTOCOL, context_limit=128000, max_turns=12,
            episode_token_limit=200000, episode_timeout=1200, observation_chars=4000)
        self.budget = Budget(self.root/"budget.db", 0, 0)
    async def asyncTearDown(self): self.temp.cleanup()
    async def run_episode(self, actions, name="ep", env=MemoryEnv, backend=None, **updates):
        runner = EpisodeRunner(self.cfg.model_copy(update=updates), self.budget, asyncio.Semaphore(1), env_factory=env)
        out = self.root/name
        result = await runner.run(self.task, out, name, backend or ScriptedBackend(actions), evaluate_after=False)
        verify_seal(out)
        return result, read_json(out/"trajectory.json"), rows(out/"events.jsonl")

    async def test_schema_and_call_finish(self):
        result, trace, events = await self.run_episode([{"type":"get_tool_schema","name":"write_value"}, call(value=80), finish()])
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(result["completion_status"], "completed")
        self.assertEqual(result["execution_metrics"]["first_business_call"]["turn"], 2)
        self.assertTrue(trace["finished"])
        self.assertEqual(trace["protocol_id"], PROTOCOL)

    async def test_zero_call_correction_once_and_blocked_exit(self):
        action = finish("Need permission", "blocked", "permission missing")
        result, trace, events = await self.run_episode([action, action])
        self.assertEqual(result["completion_status"], "blocked")
        self.assertEqual(sum(e["event_type"] == "early_finish_corrected" for e in events), 1)
        self.assertEqual(result["tool_calls"], 0)

    async def test_invalid_and_truncated_actions_never_execute(self):
        bad = {"raw_output": json.dumps(call(value=80)), "finish_reason": "length"}
        result, trace, events = await self.run_episode([bad, "", "{}", finish("Failed", "incomplete", "invalid model output")])
        self.assertEqual(result["tool_calls"], 0)
        self.assertEqual(result["execution_metrics"]["finalization_trigger"], "protocol_error_limit")
        self.assertTrue(trace["finished"])
        self.assertFalse(any(e["event_type"] == "tool_started" for e in events))

    async def test_empty_final_does_not_become_answer(self):
        result, trace, _ = await self.run_episode([finish("")], max_turns=1)
        self.assertFalse(trace["finished"])
        self.assertEqual(trace["final_answer"], "")
        self.assertEqual(trace["harness_termination_report"]["source"], "harness")

    async def test_repeated_discovery_stalls_and_finalizes(self):
        result, trace, _ = await self.run_episode([{"type":"list_tools"}]*4 + [finish("Stopped", "incomplete", "stalled")])
        self.assertEqual(result["execution_metrics"]["finalization_trigger"], "stalled")
        self.assertEqual(result["execution_metrics"]["duplicate_discovery_requests"], 3)
        self.assertTrue(trace["finished"])

    async def test_final_mode_never_calls_tools(self):
        result, trace, _ = await self.run_episode([call(value=999)], max_turns=1)
        self.assertEqual(result["tool_calls"], 0)
        self.assertFalse(trace["finished"])

    async def test_turn_tool_token_and_time_limits_finalize(self):
        scenarios = [({"max_turns": 2}, [call(value=80), finish()]),
                     ({"max_tool_calls": 1}, [call(value=80), finish()]),
                     ({"episode_token_limit": 10000}, [finish("Budget", "incomplete", "low budget")]),
                     ({"episode_timeout": 30}, [finish("Time", "incomplete", "low time")])]
        for i, (config, actions) in enumerate(scenarios):
            result, trace, _ = await self.run_episode(actions, name=f"limit{i}", **config)
            self.assertTrue(result["execution_metrics"]["finalization_attempted"], result)
            self.assertTrue(trace["finished"], result)

    async def test_unlimited_episode_tokens_preserves_execution_and_turn_guard(self):
        result, trace, _ = await self.run_episode(
            [call(value=80), finish()], episode_token_limit=None)
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(result["termination_reason"], "finish")
        self.assertTrue(trace["finished"])
        result, trace, _ = await self.run_episode(
            [call(value=81), finish()], name="unlimited_turn_guard",
            episode_token_limit=None, max_turns=2)
        self.assertEqual(result["execution_metrics"]["finalization_trigger"], "max_turns")
        self.assertTrue(trace["finished"])

    async def test_small_context_reports_without_call(self):
        result, trace, _ = await self.run_episode([], context_limit=100)
        self.assertEqual(result["termination_reason"], "context_limit")
        self.assertFalse(trace["finished"])
        self.assertEqual(result["turns"], 0)

    async def test_cancel_and_timeout_after_write_not_replayed(self):
        for i, exception in enumerate([asyncio.CancelledError(), asyncio.CancelledError("user"), TimeoutError()]):
            class Interrupted(MemoryEnv):
                async def call(self, name, arguments):
                    await super().call(name, arguments)
                    raise exception
            result, trace, events = await self.run_episode([call(value=81), finish()], name=f"cancel{i}", env=Interrupted)
            self.assertEqual(result["execution_integrity"], "outcome_unknown")
            self.assertEqual(result["tool_calls"], 1)
            self.assertEqual(sum(e["event_type"] == "tool_started" for e in events), 1)
            self.assertEqual(result["termination_phase"], "business_call")
            self.assertEqual(result["termination_origin"], ["unknown", "user", "runtime"][i])
            with sqlite3.connect(self.root/f"cancel{i}"/"final.db") as db:
                self.assertEqual(db.execute("select value from items").fetchone()[0], 81)

    async def test_tool_exception_is_visible_and_actor_can_repair(self):
        class Recoverable(MemoryEnv):
            async def call(self, name, arguments):
                if name == "write_value" and arguments["value"] == 81:
                    await super().call(name, arguments)
                    raise RuntimeError("write failed after commit; inspect state")
                return await super().call(name, arguments)
        result, trace, events = await self.run_episode(
            [call(value=81), call("read_state"), call(value=80), finish()],
            env=Recoverable, recover_tool_errors=True)
        self.assertEqual(result["termination_reason"], "finish")
        self.assertEqual(result["tool_calls"], 3)
        self.assertEqual(result["execution_integrity"], "valid")
        self.assertEqual(result["terminal_state_status"], "trusted")
        following_input = read_json(self.root / "ep/model_turns/0002/input.json")
        self.assertIn("write failed after commit", json.dumps(following_input))
        self.assertIn("outcome_unknown", json.dumps(following_input))
        self.assertTrue(any(e["event_type"] == "tool_error_feedback" for e in events))
        self.assertEqual(sum(e["event_type"] == "tool_started" for e in events), 3)
        with sqlite3.connect(self.root / "ep/final.db") as db:
            self.assertEqual(db.execute("select value from items").fetchone()[0], 80)

    async def test_recovery_does_not_swallow_user_cancellation(self):
        class Cancelled(MemoryEnv):
            async def call(self, name, arguments):
                await super().call(name, arguments)
                raise asyncio.CancelledError("user")
        result, trace, events = await self.run_episode(
            [call(value=81), finish()], env=Cancelled, recover_tool_errors=True)
        self.assertEqual(result["termination_reason"], "cancelled")
        self.assertEqual(result["execution_integrity"], "outcome_unknown")
        self.assertEqual(result["tool_calls"], 1)
        self.assertFalse(any(e["event_type"] == "tool_error_feedback" for e in events))

    async def test_final_request_can_use_held_request_slot(self):
        self.budget = Budget(self.root/"api_budget.db", 1, 50000)
        class FakeAPI(APIBackend):
            async def generate(inner, messages, folder, turn, *, output_tokens=None, final_hold=None):
                rid = inner.budget.reserve(str(turn), "actor", messages, output_tokens, consume_hold=final_hold)
                inner.budget.finish(rid, {"total_tokens": 100})
                return {"content": json.dumps(finish("Budget", "incomplete", "global cap")),
                        "finish_reason": "stop", "usage": {"total_tokens": 100}, "test_only": True}
        backend = FakeAPI(self.cfg, self.budget, asyncio.Semaphore(1), "ep")
        result, trace, _ = await self.run_episode([], backend=backend)
        self.assertTrue(trace["finished"])
        self.assertEqual(result["execution_metrics"]["finalization_trigger"], "global_budget_limit")
        self.assertEqual(self.budget.summary()["physical_requests"], 1)
        self.assertEqual(self.budget.summary()["held_requests"], 0)



class ExportV2Tests(unittest.TestCase):
    def test_blocked_and_system_reports_never_export_as_success(self):
        from test_harness_stages import export_fixture
        from harness_awm.export import export_sft
        from harness_awm.io import file_hash, seal
        for status, finished in [('blocked',True),('incomplete',True),('completed',False)]:
            with self.subTest(status=status,finished=finished), tempfile.TemporaryDirectory() as tmp:
                run = Path(tmp)/'run'
                ep = export_fixture(run)
                write_json(ep/'trajectory.json', {'protocol_id':PROTOCOL,'finished':finished,
                           'completion_status':status,'final_answer':'message'})
                seal(ep)
                report=export_sft(run,Path(tmp)/'export')
                self.assertEqual(report['accepted_episodes'],0)
                self.assertIn('not_completed_v2_answer',report['excluded'])


class CancellationPhaseTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = RunnerV2Tests.asyncSetUp
    asyncTearDown = RunnerV2Tests.asyncTearDown
    run_episode = RunnerV2Tests.run_episode
    async def test_model_queue_cancel_does_not_invent_usage(self):
        self.budget = Budget(self.root/'cancel_budget.db',10,100000)
        cfg = self.cfg.model_copy(update={'actor':'api'})
        backend = APIBackend(cfg,self.budget,asyncio.Semaphore(0),'ep')
        runner = EpisodeRunner(cfg,self.budget,asyncio.Semaphore(1),env_factory=MemoryEnv)
        job = asyncio.create_task(runner.run(self.task,self.root/'queued','ep',backend,evaluate_after=False))
        for _ in range(100):
            if getattr(backend,'phase',None)=='model_queue': break
            await asyncio.sleep(.001)
        job.cancel('scheduler')
        result=await job
        self.assertEqual(result['termination_origin'],'scheduler')
        self.assertEqual(result['termination_phase'],'model_queue')
        self.assertEqual(result['execution_metrics']['unknown_usage_requests'],0)
        self.assertEqual(self.budget.summary()['physical_requests'],0)
        self.assertEqual(self.budget.summary()['held_requests'],0)

    async def test_cleanup_cancellation_skips_evaluation(self):
        class CancelCleanup(MemoryEnv):
            async def close(self):
                self.cleanup_cancellation = 'user'
                return True
        result,trace,_ = await self.run_episode([call(value=80),finish()],env=CancelCleanup)
        self.assertEqual(result['termination_origin'],'user')
        self.assertEqual(result['termination_phase'],'cleanup')
        self.assertEqual(result['termination_reason'],'cancelled')
        self.assertFalse(trace['finished'])


if __name__ == "__main__": unittest.main()
