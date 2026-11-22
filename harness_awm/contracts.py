from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .reward_values import COMPOSITE_VERSION, VERSION, validate_reward_value

ROOT = Path(__file__).resolve().parents[1]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class APIConfig(Contract):
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "unconfigured"
    api_key_env: str = "HARNESS_API_KEY"
    timeout: float = Field(default=120, gt=0)
    temperature: float = Field(default=1.0, ge=0)
    max_tokens_alias: bool = False
    enable_thinking: bool | None = None
    reasoning_effort: Literal['none', 'low', 'medium', 'high', 'xhigh', 'max'] | None = None
    context_limit: int | None = Field(default=None, ge=1)
    judge_retry_attempts: int = Field(default=3, ge=1, le=6)
    judge_retry_max_delay: float = Field(default=60, gt=0, le=300)


class JudgeV2Config(Contract):
    review_all_decisions: bool = True
    query_seconds: float = Field(default=3, gt=0)
    query_rows: int = Field(default=100, ge=1)
    query_bytes: int = Field(default=24000, ge=256)
    query_rounds: int = Field(default=6, ge=0, le=32)
    queries_per_round: int = Field(default=12, ge=1, le=12)
    query_repairs: int = Field(default=2, ge=0, le=2)
    max_output_tokens: int = Field(default=8192, ge=256)


class SkillConfig(Contract):
    bank_path: str
    bank_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    top_k: int = Field(default=3, ge=1, le=8)
    max_context_bytes: int = Field(default=5000, ge=512, le=12000)


class RunConfig(Contract):
    schema_version: str = "awm-harness-v1"
    protocol_id: Literal["awm-single-json-v1", "awm-single-json-v2"] = "awm-single-json-v1"
    official_root: str = str(ROOT / "agent-world-model")
    runtime_python: str = str(ROOT / "agent-world-model/.venv/bin/python")
    isolation: Literal["strict", "landlock", "private-root", "development"] = "strict"
    purpose: Literal["development", "formal", "evaluation"] = "development"
    actor: Literal["scripted", "api", "training"] = "scripted"
    judge: Literal["mock", "api"] = "mock"
    judge_protocol: Literal["official", "rubric_v2"] = "official"
    judge_inputs_manifest: str | None = None
    reward_admissions: str | None = None
    reward_policy: Literal["judge-v2-terminal-discount-v1", "judge-v2-cst-v1"] = VERSION
    judge_v2: JudgeV2Config = Field(default_factory=JudgeV2Config)
    actor_api: APIConfig = Field(default_factory=APIConfig)
    judge_api: APIConfig = Field(default_factory=APIConfig)
    mock_classification: Literal["complete", "incomplete", "agent_error", "server_error", "judge_error"] = "incomplete"
    mock_finish_reason: str = "stop"
    workers: int = Field(default=1, ge=1, le=32)
    model_concurrency: int = Field(default=1, ge=1)
    judge_concurrency: int = Field(default=1, ge=1)
    judge_rpm: int | None = Field(default=None, ge=1)
    judge_tpm: int | None = Field(default=None, ge=1)
    judge_rate_db: str | None = None
    candidates_per_task: int = Field(default=1, ge=1)
    max_turns: int = Field(default=32, ge=1)
    max_tool_calls: int = Field(default=32, ge=1)
    recover_tool_errors: bool = False
    episode_timeout: float = Field(default=600, gt=0)
    tool_timeout: float = Field(default=60, gt=0)
    startup_timeout: float = Field(default=60, gt=0)
    stop_timeout: float = Field(default=5, gt=0)
    evaluator_timeout: float = Field(default=120, gt=0)
    context_limit: int = Field(default=32768, ge=1)
    max_output_tokens: int = Field(default=2048, ge=1)
    safety_margin: int = Field(default=512, ge=0)
    episode_token_limit: int | None = Field(default=131072, ge=1)
    max_physical_requests: int | None = Field(default=0, ge=0)
    max_total_tokens: int | None = Field(default=0, ge=0)
    observation_chars: int = Field(default=8000, ge=64)
    max_read_chars: int = Field(default=4000, ge=1)
    final_output_tokens: int = Field(default=2048, ge=1)
    final_timeout: float = Field(default=60, gt=0)
    context_compact_ratio: float = Field(default=.75, gt=0, lt=1)
    context_recent_turns: int = Field(default=4, ge=1)
    seed: int = 0
    skills: SkillConfig | None = None
    timezone: str = "Asia/Shanghai"
    clock_policy: Literal["actual_recorded"] = "actual_recorded"

    @model_validator(mode="after")
    def formal(self):
        if self.isolation == 'private-root' and (self.protocol_id != 'awm-single-json-v2' or self.judge_protocol != 'rubric_v2'):
            raise ValueError('private-root requires the frozen V2 tool and Judge protocols')
        if self.skills is not None and (self.protocol_id != "awm-single-json-v2" or self.purpose != "evaluation"):
            raise ValueError("skills currently support v2 inference evaluation only")
        if self.reward_policy == COMPOSITE_VERSION and self.judge_protocol != "rubric_v2":
            raise ValueError("composite_reward_requires_judge_v2")
        if (self.judge_rpm is None) != (self.judge_tpm is None):
            raise ValueError("Judge RPM and TPM must be specified together")
        if self.max_physical_requests is None or self.max_total_tokens is None:
            unlimited_development = (self.purpose == "development" and self.judge == "api"
                and self.judge_protocol == "rubric_v2" and self.isolation in {"strict", "landlock", "private-root"}
                and self.actor in {"training", "scripted"})
            if not (self.purpose == "evaluation" or unlimited_development) or not (
                    self.max_physical_requests is None and self.max_total_tokens is None):
                raise ValueError("unlimited budgets require evaluation or explicit real-Judge development and both limits=null")
        if self.purpose == "evaluation":
            if self.isolation not in {"strict", "landlock", "private-root"} or self.actor not in {"api", "training"} or self.judge != "api" or self.judge_protocol != "rubric_v2":
                raise ValueError("evaluation requires isolated tool execution, real actor and Judge V2")
            if self.reward_admissions:
                raise ValueError("evaluation is scoring-only; omit reward_admissions")
        if self.purpose == "formal" and self.judge_protocol == "rubric_v2" and not self.reward_admissions:
            raise ValueError("rubric_v2_requires_reward_admissions")
        if self.purpose == "formal" and (self.isolation != "strict" or self.actor not in {"api", "training"} or self.judge != "api"):
            raise ValueError("formal runs require strict isolation and real actor/judge")
        return self


class TaskSpec(Contract):
    schema_version: str = "awm-task-v1"
    protocol_id: Literal["awm-single-json-v1", "awm-single-json-v2"] = "awm-single-json-v1"
    dataset_version: str
    split: Literal["development", "train", "dev", "test"] = "development"
    lineage_group_id: str
    scenario: str
    native_task_id: int | str
    task_instance_id: str
    instruction: str
    initial_state_ref: str
    initial_state_hash: str
    initial_state_fingerprint: str
    environment_ref: str
    environment_code_hash: str
    tool_schema_hash: str | None = None
    verifier_ref: str | None = None
    verifier_hash: str | None = None
    judge_config_hash: str | None = None
    readiness_profile: dict = Field(default_factory=lambda: {"runtime_ready": True, "reward_ready": False})
    cleaning_status: str = "development_fixture"
    source_manifest_hash: str
    world_datetime: str | None = None
    timezone: str = "Asia/Shanghai"
    clock_policy: str = "actual_recorded"
    relative_time_resolved: bool = False


class EpisodeResult(Contract):
    schema_version: str = "awm-episode-v1"
    rollout_id: str
    attempt_id: str = "a0"
    task_instance_id: str
    lifecycle: str = "SETTLED"
    termination_reason: str
    termination_origin: str = "runtime"
    execution_integrity: str = "valid"
    terminal_state_status: str = "unavailable"
    evaluation_status: str = "not_run"
    judge_protocol: str = "official"
    evaluation_valid: bool = False
    business_label: str | None = None
    task_status: str | None = None
    outcome_classification: str | None = None
    official_pass: bool | None = None
    reward_valid: bool = False
    reward: int | float | None = None
    reward_policy: str | None = None
    reward_admission_id: str | None = None
    reward_exclusion: str | None = None
    reward_components: dict | None = None
    trajectory_quality: dict | None = None
    process_status: str = "not_evaluated"
    q: None = None
    turns: int = 0
    tool_calls: int = 0
    evaluation_id: str | None = None
    test_only: bool = True
    error: str | None = None
    completion_status: str | None = None
    termination_phase: str | None = None
    execution_metrics: dict = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def reward_type(cls, value):
        if isinstance(value, dict):
            validate_reward_value(value.get("reward"), value.get("reward_policy"))
        return value

