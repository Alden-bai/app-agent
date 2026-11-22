import asyncio
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from harness_awm.backend import Budget, ScriptedBackend
from harness_awm.contracts import RunConfig, SkillConfig
from harness_awm.io import Recorder, canonical, file_hash, read_json, verify_seal, write_json
from harness_awm.protocol_v2 import ObservationsV2, WorkingContext
from harness_awm.runner import EpisodeRunner
from harness_awm.runner_v2 import final_messages
from harness_awm.skillbank import SkillBank, VERSION
from harness_awm.skillbank_build import abstract_workflow, archived_episodes, build
from harness_awm.training.skill_eval import compare
from harness_awm.training.skill_infer import admission, rollout
from harness_support import fixture
from test_harness_v2 import MemoryEnv, TOOLS


def example_skill():
    return {"skill_id": "workflow_1", "kind": "workflow", "scenario": "harness_fixture",
            "tool_schema_hash": None, "tools": ["write_value"], "when_to_apply": "Set a value",
            "procedure": ["Read write_value schema; obtain value from the current request, then verify."],
            "search_text": "set value write", "evidence": [{"split": "train", "task_instance_id": "train:1"}]}


class SkillBankTests(unittest.TestCase):
    def test_gpu_admission_reserves_headroom_and_waits_for_workers(self):
        gpu = {'used_gib': 75, 'total_gib': 95.6, 'processes': {123: 6.5}}
        self.assertTrue(admission(gpu, {0: 123}, {0: 6.5}, set())['add'])
        self.assertFalse(admission(dict(gpu, used_gib=84), {0: 123}, {0: 6.5}, set())['add'])
        self.assertTrue(admission(dict(gpu, used_gib=90), {0: 123}, {0: 6.5}, set())['retire'])
        self.assertFalse(admission(gpu, {0: 999}, {0: 6.5}, set())['add'])
        self.assertFalse(admission(gpu, {0: 123}, {0: 6.5}, {0})['add'])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.task = fixture(self.root / "assets")

    def tearDown(self):
        self.temp.cleanup()

    def bank(self, skills=None, **updates):
        path = self.root / "bank.json"
        write_json(path, {"schema_version": VERSION, "source_split": "grpo_train",
                          "skills": skills if skills is not None else [example_skill()]})
        config = SkillConfig(bank_path=str(path), bank_sha256=file_hash(path), **updates)
        return SkillBank(config)

    def test_retrieval_gates_and_hash(self):
        bank = self.bank()
        self.assertEqual(len(bank.retrieve(self.task, TOOLS)), 1)
        self.assertEqual(bank.retrieve(self.task, []), [])
        self.assertEqual(bank.retrieve(self.task.model_copy(update={"scenario": "other"}), TOOLS), [])
        self.assertEqual(bank.retrieve(self.task.model_copy(update={"instruction": "unrelated"}), TOOLS), [])
        self.assertEqual(bank.retrieve(self.task.model_copy(update={"task_instance_id": "train:1"}), TOOLS), [])
        Path(bank.config.bank_path).write_text("{}")
        with self.assertRaisesRegex(ValueError, "hash_mismatch"):
            SkillBank(bank.config)

    def test_wrong_schema_and_nontraining_evidence(self):
        skill = example_skill()
        skill["tool_schema_hash"] = "different"
        self.assertEqual(self.bank([skill]).retrieve(self.task, TOOLS), [])
        skill["evidence"][0]["split"] = "dev"
        with self.assertRaisesRegex(ValueError, "nontraining"):
            self.bank([skill])

    def test_context_budget_determinism_and_retention(self):
        bank = self.bank()
        selected = bank.retrieve(self.task, TOOLS)
        self.assertEqual(selected, bank.retrieve(self.task, TOOLS))
        recorder = Recorder(self.root / "ep", "ep")
        message = bank.inject(self.task, TOOLS, self.root / "ep", recorder)
        self.assertLessEqual(len(message["content"].encode()), bank.config.max_context_bytes)
        context = WorkingContext([{"role": "system", "content": "system"}, message],
                                 ObservationsV2(recorder, 1024, 500), 1)
        self.assertIn(message, context.build(compact=True))
        self.assertIn(message, final_messages(context, 10000))
        skill = example_skill()
        skill["procedure"] = ["中" * 5000]
        self.assertEqual(self.bank([skill], max_context_bytes=512).retrieve(self.task, TOOLS), [])

    def test_evaluation_only_configuration(self):
        cfg = self.bank().config
        with self.assertRaisesRegex(ValueError, "evaluation only"):
            RunConfig(skills=cfg)

    def test_runner_records_actual_context_and_seal(self):
        bank = self.bank()
        config = RunConfig(protocol_id="awm-single-json-v2", purpose="evaluation", actor="training",
                           judge="api", judge_protocol="rubric_v2", isolation="landlock",
                           skills=bank.config, context_limit=32768)
        out = self.root / "run"
        runner = EpisodeRunner(config, Budget(self.root / "budget.db", 0, 0), asyncio.Semaphore(1), MemoryEnv)
        actor = ScriptedBackend([{"type": "get_tool_schema", "name": "write_value"},
                                 {"type": "call_tool", "name": "write_value", "arguments": {"value": 80}},
                                 {"type": "finish", "status": "completed", "answer": "done"}])
        with patch('harness_awm.runner_v2.evaluate_episode', new_callable=AsyncMock) as judge:
            result = asyncio.run(rollout(runner, self.task, out, "test", actor))
            judge.assert_not_awaited()
        self.assertEqual(result['evaluation_status'], 'not_run')
        self.assertEqual(runner.budget.summary()['physical_requests'], 0)
        self.assertEqual(result["tool_calls"], 1)
        messages = read_json(out / "model_turns/0001/input.json")["messages"]
        self.assertTrue(any('"skill_guidance"' in m["content"] for m in messages))
        self.assertIn("skill_selection.json", verify_seal(out))
        (out / "skill_selection.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "evidence_hash_mismatch"):
            verify_seal(out)


def trajectory():
    return {"trajectory": [{"tool_calls": [{"name": "call_tool", "arguments": {
        "tool_name": "update_record", "arguments": {"record_id": "SECRET-987", "value": "PRIVATE"}}}],
        "tool_response": {"content": '{"content":{"isError":false}}'}}]}


class BuildTests(unittest.TestCase):
    def test_abstraction_removes_values_and_errors(self):
        t = trajectory()
        self.assertEqual(abstract_workflow(t), [{"tool": "update_record", "argument_keys": ["record_id", "value"]}])
        t["trajectory"][0]["tool_response"]["content"] = '{"content":{"isError":true}}'
        self.assertEqual(abstract_workflow(t), [])

    def test_build_split_gate_provenance_and_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle, archives = root / "tasks", root / "archives"
            bundle.mkdir(); archives.mkdir()
            task = dict(task_instance_id="train:1", split="train", scenario="scene", native_task_id="1",
                        instruction="Update SECRET-987 to PRIVATE", initial_state_hash="a", environment_code_hash="b", tool_schema_hash="c")
            files = {}
            for name, tasks in (("grpo_train", [task]), ("validation", []), ("test", [])):
                path = bundle / (name + ".jsonl")
                path.write_text("".join(canonical(t) + "\n" for t in tasks))
                files[path.name] = file_hash(path)
            write_json(bundle / "manifest.json", {"schema_version": "awm-runtime-tasks-v1", "sft_task_ids": [], "files": files})
            def archive(path, corrupt=False):
                with tarfile.open(path, "w:gz") as tar:
                    for i, split in enumerate(("train", "dev")):
                        data = {"task.json": {**task, "split": split}, "trajectory.json": trajectory(),
                                "episode_result.json": {"task_instance_id": "train:1", "test_only": False,
                                    "evaluation_valid": True, "execution_integrity": "valid",
                                    "task_status": "confirmed_pass", "business_label": "pass"}}
                        raw = {k: canonical(v).encode() for k, v in data.items()}
                        seal = {k: hashlib.sha256(v).hexdigest() for k, v in raw.items() if k != "episode_result.json"}
                        if corrupt:
                            seal["trajectory.json"] = "bad"
                        raw["evidence_seal.json"] = canonical(seal).encode()
                        for k, value in raw.items():
                            member = tarfile.TarInfo(f"group-000000/candidate-{i:03d}/{k}")
                            member.size = len(value)
                            tar.addfile(member, io.BytesIO(value))
            path = archives / "group-000000.tar.gz"
            archive(path)
            report = build(bundle, archives, root / "bank")
            self.assertEqual(report["workflows"], 1)
            self.assertEqual(report["counts"]["excluded_nontraining"], 1)
            text = (root / "bank/bank.json").read_text()
            self.assertNotIn("SECRET-987", text)
            self.assertNotIn("PRIVATE", text)
            self.assertIn("archive_sha256", text)
            archive(path, corrupt=True)
            with self.assertRaisesRegex(ValueError, "sealed_training_evidence_mismatch"):
                list(archived_episodes(path))

    def test_comparison_keeps_null_and_missing(self):
        success = {"evaluation_valid": True, "task_status": "confirmed_pass"}
        fail = {"evaluation_valid": True, "task_status": "failed"}
        result = compare({"a": fail, "b": success}, {"a": success, "b": fail, "c": success}, ["a", "b", "c"])
        self.assertEqual(result["both_scored_gains"], 1)
        self.assertEqual(result["both_scored_losses"], 1)
        self.assertEqual(result["totals"]["baseline"]["unscored"], 1)
        self.assertEqual(result["success_over_planned"]["skills"], 2/3)


if __name__ == "__main__":
    unittest.main()
