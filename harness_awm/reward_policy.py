"""Versioned terminal rewards. Admission is external calibration, never inferred.

Judge caches remain assessments. Reward decisions are derived separately so a
revoked admission cannot be resurrected by reusing an otherwise valid cache.
"""
from typing import Literal
from pathlib import Path

from pydantic import Field, model_validator

from .io import db_fingerprint, digest, file_hash, verify_seal
from .judge_inputs import load_json, read_execution, verify_input
from .judge_v2 import JudgeResult, StrictModel
from .reward_values import COMPOSITE_VERSION, TERMINAL_REWARDS, VERSION


class Admission(StrictModel):
    status: Literal["admitted", "revoked"]
    calibration_ref: str = Field(min_length=1)
    policy_version: Literal["judge-v2-terminal-discount-v1", "judge-v2-cst-v1"]
    task_instance_id: str
    original_hash: str
    rubric_hash: str
    initial_state_hash: str
    environment_code_hash: str
    tool_schema_hash: str
    judge_config_hash: str


class DevelopmentAdmission(Admission):
    purpose: Literal["development"]
    isolation: Literal["landlock", "strict", "private-root"]
    calibration_sha256: str = Field(min_length=64, max_length=64)


class SharedCalibrationAdmission(DevelopmentAdmission):
    calibration_scope: Literal["shared-suite"]
    per_task_calibrated: Literal[False]


class AdmissionManifest(StrictModel):
    schema_version: Literal["awm-reward-admissions-v1", "awm-reward-admissions-v2", "awm-reward-admissions-v3"]
    tasks: dict[str, SharedCalibrationAdmission | DevelopmentAdmission | Admission]

    @model_validator(mode="after")
    def version_binding(self):
        for entry in self.tasks.values():
            expected = ("awm-reward-admissions-v3" if isinstance(entry, SharedCalibrationAdmission)
                        else "awm-reward-admissions-v2" if isinstance(entry, DevelopmentAdmission)
                        else "awm-reward-admissions-v1")
            if self.schema_version != expected:
                raise ValueError("reward_admission_schema_mismatch")
        return self


def admission_for(task, config, original, rubric, *, _manifest=None, _judge_hash=None, _shared_verified=None):
    from .judge_runtime import config_hash
    if not config.reward_admissions:
        raise ValueError("reward_admission_missing")
    manifest = _manifest if _manifest is not None else AdmissionManifest.model_validate(load_json(config.reward_admissions))
    entry = manifest.tasks.get(task.task_instance_id)
    if entry is None or entry.status != "admitted" or not entry.calibration_ref.strip():
        raise ValueError("reward_admission_missing_or_revoked")
    if entry.policy_version != config.reward_policy:
        raise ValueError("reward_admission_policy_mismatch")
    expected = dict(task_instance_id=task.task_instance_id, original_hash=digest(original),
                    rubric_hash=digest(rubric), initial_state_hash=task.initial_state_hash,
                    environment_code_hash=task.environment_code_hash, tool_schema_hash=task.tool_schema_hash,
                    judge_config_hash=_judge_hash if _judge_hash is not None else config_hash(config))
    if not task.tool_schema_hash or any(getattr(entry, key) != value for key, value in expected.items()):
        raise ValueError("reward_admission_binding_mismatch")
    if isinstance(entry, DevelopmentAdmission):
        if config.purpose != entry.purpose or config.isolation != entry.isolation:
            raise ValueError("reward_admission_execution_profile_mismatch")
        if isinstance(entry, SharedCalibrationAdmission):
            if task.split != "train":
                raise ValueError("shared_calibration_requires_train_split")
            from .shared_reward_admission import verify_shared_suite
            verify_shared_suite(entry, expected, verified=_shared_verified)
        else:
            verify_development_calibration(entry, task.task_instance_id)
    elif config.isolation != "strict":
        raise ValueError("legacy_reward_admission_requires_strict")
    return entry


def verify_development_calibration(entry, task_instance_id):
    path = Path(entry.calibration_ref)
    if file_hash(path) != entry.calibration_sha256:
        raise ValueError("reward_calibration_changed")
    proof = load_json(path)
    if (proof.get("passed") is not True or proof.get("task_instance_id") != task_instance_id
            or proof.get("judge_config_hash") != entry.judge_config_hash
            or proof.get("isolation") != entry.isolation or len(proof.get("controls", [])) != 4):
        raise ValueError("reward_calibration_invalid")
    for control in proof["controls"]:
        if control.get("matches") is not True or control.get("test_only") is not True:
            raise ValueError("reward_calibration_invalid_control")
        verify_seal(Path(control["path"]))
    if entry.policy_version == COMPOSITE_VERSION:
        quality_controls = proof.get("trajectory_controls", [])
        if (len(quality_controls) != 2
                or {c.get("kind") for c in quality_controls} != {"clean", "redundant"}
                or any(c.get("matches") is not True or c.get("test_only") is not True for c in quality_controls)):
            raise ValueError("trajectory_reward_calibration_missing_or_invalid")
        for control in quality_controls:
            verify_seal(Path(control["path"]))


def map_reward(result, *, approved, execution_valid, test_only):
    """Terminal mapping; required failure may dominate unrelated unknowns.

    Quality does not participate. Genuine, reviewed uncertainty earns 0.3;
    its business label stays unknown and it is not a confirmed success.
    """
    valid = (approved is True and execution_valid is True and test_only is False
             and result.get("test_only") is False and result.get("evaluation_valid") is True
             and result.get("evaluation_status") == "valid"
             and result.get("sample_validity", {}).get("status") == "valid"
             and not result.get("errors")
             and not any(x.get("scope") == "required" for x in result.get("rubric_issues", [])))
    pair = (result.get("business_label"), result.get("task_status"))
    valid = valid and pair in TERMINAL_REWARDS
    return {"reward": TERMINAL_REWARDS[pair] if valid else None, "reward_valid": bool(valid)}


def settle_reward(result, bundle, task, config, *, primary_only=False):
    result = JudgeResult.model_validate(result).model_dump()
    # Always clear any old decision before checking current admission.
    result.update(reward=None, reward_valid=False, reward_policy=config.reward_policy, reward_components=None,
                  reward_admission_id=None, reward_exclusion=None)
    if config.purpose == "evaluation":
        result["reward_exclusion"] = "evaluation_only_no_training_reward"
        return JudgeResult.model_validate(result).model_dump()
    try:
        bundle = verify_input(bundle)
        sealed_task, _, runtime, events, _ = read_execution(bundle["episode"])
        entry = admission_for(task, config, bundle["original"], bundle["rubric"])
        for key in ("task_instance_id", "instruction", "initial_state_hash", "initial_state_fingerprint",
                    "environment_code_hash", "tool_schema_hash"):
            if sealed_task.get(key) != getattr(task, key):
                raise ValueError("reward_task_binding_mismatch")
        if db_fingerprint(Path(bundle["episode"]) / "initial.db") != task.initial_state_fingerprint:
            raise ValueError("reward_initial_state_mismatch")
        if (result["task_instance_id"] != task.task_instance_id
                or result["task_id"] != bundle["original"]["task_id"]
                or result["versions"].get("inputs") != bundle["hashes"]
                or result["versions"].get("configuration") != entry.judge_config_hash):
            raise ValueError("reward_evaluation_binding_mismatch")
        ready = [e for e in events if e["event_type"] == "environment_ready"]
        expected_isolation = entry.isolation if isinstance(entry, DevelopmentAdmission) else "strict"
        if (config.isolation != expected_isolation or config.judge != "api" or len(ready) != 1
                or ready[0].get("isolation") != expected_isolation or ready[0].get("tools_hash") != entry.tool_schema_hash):
            raise ValueError("reward_environment_not_admitted")
        result.update(map_reward(result, approved=True,
            execution_valid=runtime["execution_integrity"] == "valid" and runtime["terminal_state_status"] == "trusted",
            test_only=runtime["test_only"]))
        if primary_only and result["reward_valid"]:
            # The terminal decision passed every admission check. No composite
            # reward exists until the group decides whether to request quality.
            result.update(reward_valid=False, reward=None, reward_exclusion="awaiting_group_decision")
        elif result["reward_valid"] and config.reward_policy == COMPOSITE_VERSION:
            from .composite_reward import components_for, reward_from_components
            components = components_for(result)
            result.update(reward_components=components, reward=reward_from_components(
                components, result["business_label"], result["task_status"]))
        result["reward_admission_id"] = digest(entry.model_dump())
        if not result["reward_valid"] and not result["reward_exclusion"]:
            result["reward_exclusion"] = "unsettled_or_test_only_result"
    except (ValueError, KeyError, OSError) as exc:
        result.update(reward=None, reward_valid=False, reward_components=None, reward_exclusion=str(exc))
    return JudgeResult.model_validate(result).model_dump()
