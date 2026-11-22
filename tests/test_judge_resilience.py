"""Judge recovery regressions: synthetic transport, real accounting and execution."""
import asyncio
import copy
import json
import sqlite3
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, AuthenticationError, RateLimitError

from harness_awm.backend import Budget, BudgetExceeded, ScriptedBackend, api_request
from harness_awm.contracts import APIConfig, RunConfig
from harness_awm.io import canonical, db_check, digest, read_json, rows, write_json
from harness_awm.judge_inputs import strict_json
from harness_awm.judge_output import parse_judge_output
from harness_awm import trajectory_quality as quality_runtime
from test_trajectory_quality_repair import FIRST, assessment, context


def test_judge_normalization_is_lossless_and_sources_stay_strict(tmp_path):
    raw = '{"turn":"0010","quote":"size_bytes","turn":"0010"}'
    value = parse_judge_output(raw, tmp_path / "normalization.json")
    assert value == {"turn": "0010", "quote": "size_bytes"}
    assert read_json(tmp_path / "normalization.json")["semantic_reassessment"] is False
    assert parse_judge_output('```json\n' + raw + '\n```', tmp_path / "fence.json") == value
    for text in ('{"a":1,"a":2}', '{"a":true,"a":1}', '{"a":NaN}'):
        with pytest.raises(ValueError):
            parse_judge_output(text, tmp_path / "invalid.json")
    with pytest.raises(ValueError, match="duplicate_json_key"):
        strict_json(raw)


def run_quality(tmp_path, monkeypatch, responses):
    calls = []
    async def send(config, request, budget, key, stage, folder):
        calls.append(request)
        text = responses.pop(0)
        return {"choices": [{"finish_reason": "stop", "message": {
            "content": text if isinstance(text, str) else canonical(text)}}]}
    monkeypatch.setattr(quality_runtime, "api_request", send)
    ctx = context()
    result = asyncio.run(quality_runtime.assess_payload(
        quality_runtime.compact_quality_payload(ctx, "task", FIRST), ctx, RunConfig(judge="api"),
        Budget(tmp_path / "budget.db", 20, 1000000), asyncio.Semaphore(1), "test/", tmp_path))
    return result, calls


def test_duplicate_turn_review_needs_no_reassessment(tmp_path, monkeypatch):
    value = assessment(grade=.5)
    raw = canonical(value).replace('"turn":"0001"', '"turn":"0001","turn":"0001"')
    result, calls = run_quality(tmp_path, monkeypatch, [value, raw])
    assert result == value and len(calls) == 2
    assert (tmp_path / "call-01/normalization.json").is_file()


def test_local_repair_cannot_overwrite_unaffected_dimension(tmp_path, monkeypatch):
    value = assessment(grade=.5)
    invalid = copy.deepcopy(value)
    invalid["dimensions"][1]["reason"] = ""
    replacement = assessment()  # The repair tries to erase the valid information defect.
    result, calls = run_quality(tmp_path, monkeypatch, [invalid, replacement, value])
    candidate = json.loads(calls[2]["messages"][-2]["content"])
    assert next(d for d in candidate["dimensions"] if d["dimension"] == "information")["grade"] == .5
    assert result == value


def test_malformed_review_preserves_initial_score_with_explicit_status(tmp_path, monkeypatch):
    value = assessment(grade=.5)
    result, calls = run_quality(tmp_path, monkeypatch, [value, '{bad', '{bad', '{bad'])
    assert result == value and len(calls) == 4
    assert read_json(tmp_path / "review_status.json")["reviewed"] is False


def test_semantic_review_failure_cannot_be_hidden_by_later_syntax_errors(tmp_path, monkeypatch):
    value = assessment(grade=.5)
    invalid = copy.deepcopy(value)
    invalid["dimensions"][0]["observations"][0]["quote"] = "not in the original observation"
    with pytest.raises(json.JSONDecodeError):
        run_quality(tmp_path, monkeypatch, [value, invalid, '{bad', '{bad'])
    assert not (tmp_path / "review_status.json").exists()


def test_review_api_recovery_reuses_valid_initial_assessment(tmp_path, monkeypatch):
    stages = []
    failed = [False]
    async def send(config, request, budget, key, stage, folder):
        stages.append(stage)
        if stage.endswith("review") and not failed[0]:
            failed[0] = True
            raise TimeoutError("one interrupted review")
        return {"choices": [{"finish_reason": "stop", "message": {"content": canonical(assessment())}}]}
    monkeypatch.setattr(quality_runtime, "api_request", send)
    ctx, config = context(), RunConfig(judge="api")
    payload = quality_runtime.compact_quality_payload(ctx, "task", FIRST)
    budget = Budget(tmp_path / "budget.db", 20, 1000000)
    with pytest.raises(TimeoutError):
        asyncio.run(quality_runtime.assess_payload(payload, ctx, config, budget, asyncio.Semaphore(1),
                    "a/", tmp_path / "attempt-0000"))
    result = asyncio.run(quality_runtime.assess_payload(payload, ctx, config, budget, asyncio.Semaphore(1),
                        "b/", tmp_path / "attempt-0001"))
    assert result["status"] == "valid"
    assert stages == ["trajectory_quality", "trajectory_quality_review", "trajectory_quality_review"]


@pytest.mark.parametrize("stage,failures,expected_calls", [("judge_v2", 1, 2),
    ("trajectory_quality_review", 9, 3), ("actor", 1, 1)])
def test_network_retry_budget_and_error_redaction(tmp_path, monkeypatch, stage, failures, expected_calls):
    attempts, sleeps = [], []
    class Client:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 0
            self.chat = SimpleNamespace(completions=self)
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
        async def create(self, **kwargs):
            attempts.append(kwargs)
            if len(attempts) <= failures:
                try:
                    raise ConnectionError("secret-test-key connection reset")
                except ConnectionError as cause:
                    raise APIConnectionError(request=httpx.Request("POST", "https://example.test")) from cause
            return SimpleNamespace(model_dump=lambda **kwargs: {"usage": {"total_tokens": 10}})
    async def sleep(seconds): sleeps.append(seconds)
    monkeypatch.setattr("openai.AsyncOpenAI", Client)
    monkeypatch.setattr("harness_awm.judge_recovery.asyncio.sleep", sleep)
    monkeypatch.setenv("JUDGE_TEST_KEY", "secret-test-key")
    budget = Budget(tmp_path / "budget.db", 10, 100000)
    request = api_request(APIConfig(api_key_env="JUDGE_TEST_KEY"),
        {"model": "test", "messages": [], "max_completion_tokens": 10}, budget, "r", stage, tmp_path / "call")
    if expected_calls <= failures:
        with pytest.raises(APIConnectionError): asyncio.run(request)
    else:
        assert asyncio.run(request)["usage"]["total_tokens"] == 10
    assert len(attempts) == expected_calls
    assert budget.summary()["physical_requests"] == expected_calls
    assert budget.summary()["unknown_usage_requests"] == min(failures, expected_calls)
    assert len(sleeps) == expected_calls - 1
    error = (tmp_path / "call/error.json").read_text()
    assert "secret-test-key" not in error and "ConnectionError" in error


@pytest.mark.parametrize("kind", ["auth", "budget", "cancel"])
def test_nonrecoverable_failures_never_retry(tmp_path, monkeypatch, kind):
    from harness_awm.judge_recovery import request_with_retries
    calls = []
    error = {"auth": AuthenticationError("bad credentials", response=httpx.Response(401,
        request=httpx.Request("POST", "https://example.test")), body=None),
        "budget": BudgetExceeded("spent"), "cancel": asyncio.CancelledError()}[kind]
    async def send(*args, **kwargs):
        calls.append(1)
        raise error
    with pytest.raises(type(error)):
        asyncio.run(request_with_retries(send, APIConfig(), {}, None, "r", "judge_v2", tmp_path))
    assert len(calls) == 1


def test_rate_limited_transport_does_not_multiply_retries(tmp_path, monkeypatch):
    from harness_awm import judge_recovery, judge_rate_limit
    sends, reservations, delays = [], [], []
    class Gate:
        async def acquire(self, tokens):
            reservations.append(tokens)
            return len(reservations)
        def defer(self, seconds): delays.append(seconds)
        def refund(self, reservation): raise AssertionError("Dispatched requests remain accounted")
    class Client:
        def __init__(self, **kwargs): self.chat = SimpleNamespace(completions=self)
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
        async def create(self, **kwargs):
            sends.append(kwargs)
            request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
            if len(sends) == 1:
                raise RateLimitError("slow down", response=httpx.Response(429, request=request,
                    headers={"retry-after": "12"}), body={"code": "rate_limit_exceeded"})
            raise APIConnectionError(request=request)
    monkeypatch.setenv("JUDGE_TEST_KEY", "test-only")
    monkeypatch.setattr("openai.AsyncOpenAI", Client)
    monkeypatch.setattr(judge_rate_limit, "estimated_tokens", lambda request: 100)
    monkeypatch.setattr(judge_recovery.random, "uniform", lambda *args: 0)
    budget = Budget(tmp_path / "budget.db", 10, 100000)
    send = judge_rate_limit.paced_transport(api_request, Gate())
    with pytest.raises(APIConnectionError):
        asyncio.run(send(APIConfig(base_url="https://api.openai.com/v1", api_key_env="JUDGE_TEST_KEY"),
            {"model": "gpt-6-luna", "messages": [], "max_completion_tokens": 10},
            budget, "r", "judge_v2", tmp_path / "call"))
    assert len(sends) == len(reservations) == 3
    assert budget.summary()["physical_requests"] == budget.summary()["unknown_usage_requests"] == 3
    assert delays == [12, 10]


def test_retry_cannot_exceed_physical_request_budget(tmp_path, monkeypatch):
    from harness_awm.judge_recovery import request_with_retries
    calls = []
    budget = Budget(tmp_path / "budget.db", 1, 100000)
    async def send(config, kwargs, budget, key, stage, folder, **options):
        rid = budget.reserve(key, stage, [], 10)
        calls.append(key)
        budget.finish(rid)
        raise ConnectionError("transient")
    async def sleep(seconds): pass
    monkeypatch.setattr("harness_awm.judge_recovery.asyncio.sleep", sleep)
    with pytest.raises(BudgetExceeded):
        asyncio.run(request_with_retries(send, APIConfig(), {}, budget, "r", "judge_v2", tmp_path))
    assert calls == ["r"] and budget.summary()["physical_requests"] == 1


@pytest.mark.parametrize("recover", [True, False])
def test_actor_sees_fk_feedback_and_can_correct_before_finish(tmp_path, recover):
    from harness_awm.environment import AWMEnvAdapter
    from harness_awm.runner import EpisodeRunner
    from test_harness_v2 import MemoryEnv, call, finish
    from harness_support import fixture
    class LogicalEnv(MemoryEnv):
        def __init__(self, task, config, folder):
            super().__init__(task, config, folder)
            self.config, self.runtime = config, folder
        async def call(self, name, arguments):
            with sqlite3.connect(self.folder / "working.db") as db:
                if arguments["value"] == 1:
                    db.executescript("CREATE TABLE parent(id INTEGER PRIMARY KEY); "
                        "CREATE TABLE child(id INTEGER PRIMARY KEY,pid INTEGER REFERENCES parent(id)); "
                        "INSERT INTO child VALUES(1,99);")
                else:
                    db.execute("INSERT INTO parent VALUES(99)")
            return {"isError": False, "content": [{"type": "text", "text": "write returned"}]}
        audit = AWMEnvAdapter.audit
        freeze = AWMEnvAdapter.freeze
    config = RunConfig(protocol_id="awm-single-json-v2", context_limit=128000,
                       episode_token_limit=200000, max_turns=8, episode_timeout=1200)
    task = fixture(tmp_path / "assets")
    runner = EpisodeRunner(config, Budget(tmp_path / "budget.db", 0, 0), asyncio.Semaphore(1), env_factory=LogicalEnv)
    actions = [call(value=1)] + ([call(value=2)] if recover else []) + [finish()]
    folder = tmp_path / "episode"
    result = asyncio.run(runner.run(task, folder, "r", ScriptedBackend(actions), evaluate_after=False))
    assert result["execution_integrity"] == "valid" and result["terminal_state_status"] == "trusted"
    assert "foreign_key_violation" in canonical(read_json(folder / "model_turns/0002/input.json"))
    db_check(folder / "final.db", foreign_keys=False)
    if recover:
        db_check(folder / "final.db")
    else:
        with pytest.raises(ValueError, match="foreign_keys"):
            db_check(folder / "final.db")
    final = next(e for e in rows(folder / "events.jsonl") if e["event_type"] == "snapshot_created")
    assert bool(final["foreign_key_violations"]) != recover


def test_one_null_keeps_other_candidates_and_updates_parameters(tmp_path, monkeypatch):
    import torch
    from test_training_grpo import engine_fixture
    from harness_awm.training.verl.core import read_group
    engine, task, *_ = engine_fixture(tmp_path, monkeypatch, test_only=False, states=("pass", "fail", "pass"))
    group = asyncio.run(engine.collect(task))
    manifest = read_json(group / "group.json")
    bad = group / "candidate-002/episode_result.json"
    result = read_json(bad)
    result.update(reward=None, reward_valid=False, reward_admission_id=None, reward_exclusion="judge_transport_error")
    write_json(bad, result)
    manifest["candidates"][2]["result_hash"] = digest(result)
    write_json(group / "group.json", manifest)
    engine.pending = (group.resolve(), digest(manifest))
    trajectories, rewards, advantages = read_group(group, task, engine.runner.config,
        engine.policy_version, 3, engine.tokenizer)
    assert len(trajectories) == 2 and rewards == [1, 0]
    assert advantages == pytest.approx([1, -1], abs=3e-6)
    assert read_json(group / "reward_selection.json")["included_indices"] == [0, 1]
    before = [p.detach().clone() for p in engine.model.parameters()]
    report = engine.update(group)
    assert report["status"] == "updated" and report["rewards"] == [1, 0]
    assert report["excluded_candidates"][0]["path"] == "candidate-002"
    assert any(not torch.equal(p, b) for p, b in zip(engine.model.parameters(), before))


def test_failed_task_still_receives_process_assessment(tmp_path, monkeypatch):
    from harness_awm.judge_runtime import config_hash
    from harness_awm.reward_values import COMPOSITE_VERSION
    from test_training_grpo import engine_fixture
    engine, task, *_ = engine_fixture(tmp_path, monkeypatch, test_only=False, states=("fail", "fail"))
    config = engine.runner.config.model_copy(update={"reward_policy": COMPOSITE_VERSION})
    engine.runner.config = config
    manifest = read_json(config.reward_admissions)
    manifest["tasks"][task.task_instance_id].update(policy_version=COMPOSITE_VERSION, judge_config_hash=config_hash(config))
    write_json(config.reward_admissions, manifest)
    calls = []
    async def send(*args):
        calls.append(args[4])
        return {"choices": [{"finish_reason": "stop", "message": {"content": canonical(assessment())}}]}
    monkeypatch.setattr(quality_runtime, "api_request", send)
    group = asyncio.run(engine.collect(task))
    for path in group.glob("candidate-*/episode_result.json"):
        result = read_json(path)
        assert result["reward"] == 0 and result["process_status"] == "valid"
        assert len(result["trajectory_quality"]["assessment"]["dimensions"]) == 3
    assert len(calls) == 4
