"""Judge V2 assessments; externally admitted rewards are added by reward_policy."""
import copy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .io import digest
from .rubric_judge import decide_item, validate_evidence
from .reward_values import COMPOSITE_VERSION, PROVISIONAL_REWARD, TERMINAL_REWARDS, VERSION as REWARD_VERSION, validate_reward_value


VERSION = "awm-judge-v2"
ADAPTER_VERSION = "source-bound-rubric-v2"
TEXT = {"min_length": 1}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Proof(StrictModel):
    established: bool
    evidence_ids: list[str]
    reason: str = Field(**TEXT)


class Unknown(StrictModel):
    kind: Literal["semantic_ambiguity", "evidence_gap", "evidence_conflict", "evidence_error", "budget_exhausted", "not_investigated"]
    missing_fact: str = Field(**TEXT)
    impact: str = Field(**TEXT)
    checked_evidence_ids: list[str] = Field(min_length=1)
    next_action: str = Field(**TEXT)
    can_resolve_with_tools: bool


class Item(StrictModel):
    id: str
    satisfied: Proof
    violated: Proof
    unknown: Unknown | None


class SampleValidity(StrictModel):
    status: Literal["valid", "invalid", "unknown"]
    evidence_ids: list[str]
    reason: str = Field(**TEXT)


class RubricIssue(StrictModel):
    scope: Literal["required", "quality"]
    affected_items: list[str]
    reason: str = Field(**TEXT)
    source_ids: list[str]


class UnresolvedCheck(StrictModel):
    id: str
    status: Literal["not_triggered", "resolved", "material"]
    affected_items: list[str]
    evidence_ids: list[str]
    reason: str = Field(**TEXT)


class Assessment(StrictModel):
    action: Literal["finish"]
    task_id: str
    items: list[Item]
    sample_validity: SampleValidity
    rubric_issues: list[RubricIssue]
    unresolved_checks: list[UnresolvedCheck]


class EvidenceCall(StrictModel):
    tool: Literal["inspect_schema", "query_snapshot", "compare_records", "read_evidence"]
    objective: str = Field(**TEXT)
    affected_items: list[str] = Field(min_length=1)
    arguments: dict


class QueryAction(StrictModel):
    action: Literal["query"]
    queries: list[EvidenceCall] = Field(min_length=1, max_length=12)


class JudgeResult(StrictModel):
    schema_version: Literal["awm-judge-result-v2"] = "awm-judge-result-v2"
    task_id: str
    task_instance_id: str
    evaluation_id: str
    evaluation_status: Literal["valid", "evaluation_error", "invalid_sample", "rubric_error"]
    evaluation_valid: bool
    business_label: Literal["pass", "fail", "unknown"]
    task_status: Literal["confirmed_pass", "provisional_pass", "failed", "undetermined"]
    items: list[dict]
    required_counts: dict[str, int]
    quality_counts: dict[str, int]
    unresolved_checks: list[dict]
    sample_validity: dict
    rubric_issues: list[dict]
    errors: list[str]
    versions: dict
    usage: dict
    reward: int | float | None = None
    reward_valid: bool = False
    reward_policy: str | None = None
    reward_admission_id: str | None = None
    reward_exclusion: str | None = None
    reward_components: dict | None = None
    trajectory_quality: dict | None = None
    official_pass: None = None
    outcome_classification: None = None
    test_only: bool = False
    attempt: str = ""

    @model_validator(mode="before")
    @classmethod
    def reward_type(cls, value):
        if isinstance(value, dict):
            validate_reward_value(value.get("reward"), value.get("reward_policy"))
        return value

    @model_validator(mode="after")
    def reward_contract(self):
        if self.reward_valid:
            expected = TERMINAL_REWARDS.get((self.business_label, self.task_status))
            if self.reward_policy == COMPOSITE_VERSION:
                from .composite_reward import components_for, reward_from_components
                components = components_for(self.model_dump())
                if components != self.reward_components:
                    raise ValueError("invalid_composite_reward_components")
                expected = reward_from_components(components, self.business_label, self.task_status)
            if (self.reward is None or self.reward != expected or self.test_only
                    or not self.evaluation_valid or self.evaluation_status != "valid"
                    or self.sample_validity.get("status") != "valid" or self.errors
                    or any(x.get("scope") == "required" for x in self.rubric_issues)
                    or not self.reward_policy or not self.reward_admission_id or self.reward_exclusion
                    or (self.reward == PROVISIONAL_REWARD and self.reward_policy not in {REWARD_VERSION, COMPOSITE_VERSION})):
                raise ValueError("invalid_admitted_reward")
        elif self.reward is not None:
            raise ValueError("invalid_reward_must_be_null")
        return self


class RubricError(ValueError):
    pass


def normalize_rubric(original, rubric):
    """Freeze a source-bound interpretation, without silently repairing predicates.

    Legacy P remains verbatim; its independent necessary counterexamples are F.
    This is logical normalization, not a claim of semantic review. The runtime
    must report unsupported/omitted requirements, rather than changing them.
    """
    if original.get("task_id") != rubric.get("task_id") or not original.get("task"):
        raise RubricError("task_rubric_binding_mismatch")
    sources = {}
    for source in original.get("sources", []):
        if not isinstance(source.get("id"), str) or source["id"] in sources or not isinstance(source.get("text"), str):
            raise RubricError("invalid_or_duplicate_source")
        sources[source["id"]] = source["text"]
    if sources.get("task") != original["task"]:
        raise RubricError("missing_exact_task_source")
    result, seen = [], set()
    for row in rubric.get("rubric", []):
        rid = row.get("id")
        if not isinstance(rid, str) or not rid.strip() or rid in seen:
            raise RubricError("duplicate_or_invalid_rubric_id")
        seen.add(rid)
        if row.get("requirement_level") not in {"required", "quality"}:
            raise RubricError("invalid_requirement_level")
        refs = row.get("sources", [])
        if not refs or any(not r.get("quote") or r.get("source_id") not in sources or r["quote"] not in sources[r["source_id"]] for r in refs):
            raise RubricError("invalid_source_quote:" + rid)
        criterion = row.get("criterion", "")
        requirement, sep, rest = criterion.removeprefix("要求：").partition("\nG：")
        gate, sep2, success = rest.partition("\nP：")
        success, sep3, failure = success.partition("\nF：")
        if not criterion.startswith("要求：") or not sep or not sep2 or not all(s.strip() for s in [requirement, gate, success]):
            raise RubricError("unsupported_predicate_format:" + rid)
        if not sep3:
            failure = "可信证据证明上述P中至少一个源自原题或公开契约的必要义务被违反，且该反证不依赖其他未知义务。G缺失本身不是违例。"
        if not failure.strip():
            raise RubricError("empty_failure_predicate:" + rid)
        result.append({"id": rid, "requirement_level": row["requirement_level"], "requirement": requirement,
                       "success_condition": success, "failure_condition": failure, "evidence_guidance": gate,
                       "failure_origin": "explicit_F" if sep3 else "independent_counterexample_to_P",
                       "sources": copy.deepcopy(refs), "original_criterion": criterion})
    if not result or not any(r["requirement_level"] == "required" for r in result):
        raise RubricError("no_required_items")
    coverage = rubric.get("coverage", [])
    for row in coverage:
        if not row.get("quote") or row["quote"] not in original["task"] or not set(row.get("item_ids", [])) <= seen:
            raise RubricError("invalid_coverage_source_or_item")
    unresolved = []
    for i, row in enumerate(rubric.get("unresolved", []), 1):
        if not row.get("issue") or not set(row.get("affected_items", [])) <= seen:
            raise RubricError("invalid_unresolved_item")
        unresolved.append({**copy.deepcopy(row), "id": f"U{i:03d}"})
    return {"version": ADAPTER_VERSION, "task_id": rubric["task_id"], "source_hash": digest(original),
            "rubric_hash": digest(rubric), "items": result, "unresolved": unresolved,
            "source_ids": list(sources),
            "coverage": copy.deepcopy(coverage), "semantic_review": "runtime_reports_issues; no_source_requirements_changed"}


def validate_assessment(value, normalized, evidence):
    response = Assessment.model_validate(value).model_dump()
    validate_evidence(evidence)
    expected = {r["id"]: r for r in normalized["items"]}
    actual = [r["id"] for r in response["items"]]
    if response["task_id"] != normalized["task_id"] or len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError("judge_item_coverage_or_task_mismatch")

    def refs(ids, proof=False):
        if len(ids) != len(set(ids)) or not set(ids) <= evidence.keys():
            raise ValueError("invalid_evidence_reference")
        if proof and (not ids or any(not evidence[x]["valid"] or evidence[x].get("error") for x in ids)):
            raise ValueError("proof_requires_usable_evidence")

    for row in response["items"]:
        for key in ("satisfied", "violated"):
            p = row[key]
            if not p["reason"].strip():
                raise ValueError("empty_proof_reason")
            refs(p["evidence_ids"], p["established"])
            def execution_fact(eid):
                record = evidence[eid]
                if record["kind"] == "evidence_page":
                    return record["content"]["source_kind"] in {"trajectory", "final_answer", "tool_result", "query", "state_validation"}
                return record["kind"] in {"trajectory", "final_answer", "tool_result", "query", "state_validation"}
            if p["established"] and not any(execution_fact(e) for e in p["evidence_ids"]):
                raise ValueError("proof_without_execution_evidence")
        state = decide_item(satisfied=row["satisfied"]["established"], violated=row["violated"]["established"])
        if (state == "unknown") != (row["unknown"] is not None):
            raise ValueError("unknown_details_mismatch")
        if row["unknown"]:
            u = row["unknown"]
            if not all(u[k].strip() for k in ("missing_fact", "impact", "next_action")):
                raise ValueError("unknown_without_specific_gap")
            refs(u["checked_evidence_ids"])
        row.update(status=state, requirement_level=expected[row["id"]]["requirement_level"])
    sample = response["sample_validity"]
    refs(sample["evidence_ids"], sample["status"] != "unknown")
    if not sample["reason"].strip():
        raise ValueError("empty_sample_reason")
    for issue in response["rubric_issues"]:
        if not set(issue["affected_items"]) <= expected.keys() or not issue["reason"].strip():
            raise ValueError("invalid_rubric_issue")
        if not issue["source_ids"] or not set(issue["source_ids"]) <= set(normalized["source_ids"]):
            raise ValueError("invalid_issue_source")
        if any(expected[i]["requirement_level"] == "required" for i in issue["affected_items"]):
            issue["scope"] = "required"
    checks = response["unresolved_checks"]
    expected_u = {u["id"]: u for u in normalized["unresolved"]}
    if len(checks) != len(expected_u) or {c["id"] for c in checks} != expected_u.keys():
        raise ValueError("unresolved_check_coverage")
    for check in checks:
        if set(check["affected_items"]) != set(expected_u[check["id"]]["affected_items"]) or not check["reason"].strip():
            raise ValueError("unresolved_scope_changed")
        refs(check["evidence_ids"])
        states = {r["id"]: r["status"] for r in response["items"]}
        if check["status"] == "material" and not any(states[i] != "pass" for i in check["affected_items"]):
            raise ValueError("material_unresolved_cannot_accompany_all_affected_passes")
    return response


def aggregate(response, *, forced_error=None):
    items = response["items"]
    required = [r for r in items if r["requirement_level"] == "required"]
    quality = [r for r in items if r["requirement_level"] == "quality"]
    states = [r["status"] for r in required]
    label = "fail" if "fail" in states else "pass" if states and all(s == "pass" for s in states) else "unknown"
    status, errors = "valid", []
    blocking = [r for r in required if r["status"] == "unknown" and (
        r["unknown"]["kind"] in {"evidence_error", "budget_exhausted", "not_investigated"} or r["unknown"]["can_resolve_with_tools"])]
    if forced_error:
        status, errors = "evaluation_error", [forced_error]
    elif response["sample_validity"]["status"] == "invalid":
        status = "invalid_sample"
    elif any(i["scope"] == "required" for i in response["rubric_issues"]) or not required:
        status = "rubric_error"
    elif response["sample_validity"]["status"] == "unknown":
        status, errors = "evaluation_error", ["material_sample_precondition_unverified"]
    elif label != "fail" and blocking:
        status, errors = "evaluation_error", ["required_evidence_incomplete:" + ",".join(r["id"] for r in blocking)]
    # A quality-only rule defect affects that item's assessment, never hard completion.
    bad_quality = {i for issue in response["rubric_issues"] if issue["scope"] == "quality" for i in issue["affected_items"]}
    for row in quality:
        if row["id"] in bad_quality:
            row["rubric_issue"] = True
    return {"items": items, "business_label": label, "evaluation_status": status, "evaluation_valid": status == "valid",
            "task_status": {"pass": "confirmed_pass", "fail": "failed", "unknown": "provisional_pass"}[label] if status == "valid" else "undetermined",
            "required_counts": {s: states.count(s) for s in ("pass", "fail", "unknown")},
            "quality_counts": {s: sum(r["status"] == s for r in quality) for s in ("pass", "fail", "unknown")},
            "sample_validity": response["sample_validity"], "rubric_issues": response["rubric_issues"],
            "unresolved_checks": response["unresolved_checks"], "errors": errors}


def error_result(task_id, task_instance_id, error, *, status="evaluation_error"):
    return JudgeResult(task_id=task_id, task_instance_id=task_instance_id, evaluation_id="", evaluation_status=status,
                       evaluation_valid=False, business_label="unknown", task_status="undetermined", items=[],
                       required_counts={}, quality_counts={}, unresolved_checks=[],
                       sample_validity={"status": "unknown", "evidence_ids": [], "reason": error}, rubric_issues=[],
                       errors=[error], versions={"judge": VERSION}, usage={}).model_dump()
