"""Program-owned C/S/T composition; the judges never emit a training reward."""
import math

from .reward_values import COMPOSITE_VERSION, LAMBDA_S, LAMBDA_T, PROVISIONAL_REWARD, TERMINAL_REWARDS

DIMENSIONS = ("information", "feedback", "exploration")
GENUINE_UNKNOWN = {"semantic_ambiguity", "evidence_gap", "evidence_conflict"}


def soft_scores(items):
    scores = []
    for item in items:
        if item["requirement_level"] != "quality":
            continue
        if item.get("rubric_issue"):
            raise ValueError("soft_rubric_issue")
        status = item["status"]
        if status == "unknown":
            unknown = item.get("unknown") or {}
            if unknown.get("kind") not in GENUINE_UNKNOWN or unknown.get("can_resolve_with_tools") is not False:
                raise ValueError("soft_evidence_unsettled:" + item["id"])
        scores.append({"id": item["id"], "status": status,
                       "value": {"pass": 1.0, "fail": 0.0, "unknown": PROVISIONAL_REWARD}[status]})
    return scores


def components_for(result):
    c = TERMINAL_REWARDS.get((result.get("business_label"), result.get("task_status")))
    if c is None:
        raise ValueError("unsettled_completion_tier")
    components = {"policy": COMPOSITE_VERSION, "C": c, "S": None, "T": None,
                  "alpha": PROVISIONAL_REWARD, "lambda_s": LAMBDA_S, "lambda_t": LAMBDA_T,
                  "soft_items": [], "defects": {}}
    # Quality cannot rescue failure, and unrelated quality errors cannot erase it.
    if c == 0:
        return components
    quality = result.get("trajectory_quality") or {}
    if quality.get("status") != "valid":
        raise ValueError("trajectory_quality_unsettled:" + str(quality.get("error", quality.get("status"))))
    from .trajectory_quality import QualityAssessment
    assessment = QualityAssessment.model_validate(quality["assessment"])
    if assessment.status != "valid":
        raise ValueError("trajectory_quality_unsettled")
    soft = soft_scores(result["items"])
    defects = {row.dimension: row.grade for row in assessment.dimensions}
    if set(defects) != set(DIMENSIONS) or len(assessment.dimensions) != len(DIMENSIONS):
        raise ValueError("trajectory_dimension_coverage")
    components.update(S=sum(x["value"] for x in soft) / len(soft) if soft else 1.0,
                      T=1 - sum(defects.values()) / 3, soft_items=soft, defects=defects)
    return components


def reward_from_components(components, business_label, task_status):
    """Recompute at the training boundary instead of accepting an arbitrary float."""
    c = TERMINAL_REWARDS.get((business_label, task_status))
    if (not isinstance(components, dict) or c is None or components.get("policy") != COMPOSITE_VERSION
            or type(components.get("C")) not in (int, float) or components["C"] != c
            or components.get("alpha") != PROVISIONAL_REWARD
            or components.get("lambda_s") != LAMBDA_S or components.get("lambda_t") != LAMBDA_T):
        raise ValueError("invalid_composite_reward_policy_or_tier")
    if c == 0:
        if (components.get("S") is not None or components.get("T") is not None
                or components.get("soft_items") != [] or components.get("defects") != {}):
            raise ValueError("failed_reward_has_quality_components")
        return 0.0
    soft, defects = components["soft_items"], components["defects"]
    if not isinstance(soft, list) or len({x["id"] for x in soft}) != len(soft):
        raise ValueError("invalid_soft_item_coverage")
    for row in soft:
        if (type(row["value"]) not in (int, float)
                or row["value"] != {"pass": 1, "unknown": PROVISIONAL_REWARD, "fail": 0}.get(row["status"])):
            raise ValueError("invalid_soft_score")
    if set(defects) != set(DIMENSIONS) or any(type(x) not in (int, float) or x not in (0, 0.5, 1) for x in defects.values()):
        raise ValueError("invalid_trajectory_defect_grades")
    s = sum(x["value"] for x in soft) / len(soft) if soft else 1.0
    t = 1 - sum(defects.values()) / 3
    for name, expected in (("S", s), ("T", t)):
        value = components.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value != expected:
            raise ValueError("inconsistent_composite_component:" + name)
    return round(c * (1 - LAMBDA_S * (1 - s) - LAMBDA_T * (1 - t)), 12)
