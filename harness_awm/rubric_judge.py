"""D1 proof validation and deterministic aggregation; never calls a model.

Proof truth is a semantic judgment. This layer validates its evidence references
and fixes the truth table; it does not claim to verify arbitrary prose proofs.
"""
from jsonschema import Draft202012Validator


VERSION = "rubric-d1-v1"
DECISIONS = {
    "satisfied_when": "D1：可信证据证明本条P的全部必要义务成立，且无影响该结论的证据冲突。",
    "violated_when": "D1：可信证据证明本条F中至少一个独立必要违例成立，不受其他未知义务阻断。",
    "unknown_when": "D1：既未证明本条全部必要义务满足，也未证明独立必要违例；影响结论的证据冲突未消除。",
}


def obj(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def response_schema(task_id, item_ids, evidence_ids):
    text = {"type": "string", "pattern": r"\S"}
    refs = {"type": "array", "items": {"enum": list(evidence_ids)}, "uniqueItems": True}
    proof = obj({"established": {"type": "boolean"}, "evidence_ids": refs, "reason": text})
    return obj({
        "task_id": {"const": task_id},
        "items": {"type": "array", "minItems": len(item_ids), "maxItems": len(item_ids),
                  "items": obj({"id": {"enum": list(item_ids)}, "satisfied": proof, "violated": proof,
                                "unknown_reason": {"type": "string"}})},
        "sample_validity": obj({"status": {"enum": ["valid", "invalid", "unknown"]},
                                "evidence_ids": refs, "reason": text}),
        "rubric_issues": {"type": "array", "items": text, "uniqueItems": True},
    })


def validate_rubric(rubric):
    if not isinstance(rubric, dict) or not isinstance(rubric.get("task_id"), str):
        raise ValueError("invalid_rubric")
    rows = rubric.get("rubric")
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty_rubric")
    ids = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip() or row["id"] in ids:
            raise ValueError("duplicate_or_invalid_rubric_id")
        ids.add(row["id"])
        if row.get("requirement_level") not in ("required", "quality"):
            raise ValueError("invalid_requirement_level")
        d = row.get("decision")
        if (not isinstance(d, dict) or set(d) != {"method", *DECISIONS}
                or d.get("method") not in ("deterministic", "semantic", "hybrid")
                or any(d.get(k) != v for k, v in DECISIONS.items())):
            raise ValueError("not_d1_rubric_recompile_from_sources")
        criterion = row.get("criterion", "")
        if not isinstance(criterion, str) or not criterion.startswith("要求："):
            raise ValueError("missing_d1_predicates")
        remaining = criterion.removeprefix("要求：")
        for marker in ("\nG：", "\nP：", "\nF："):
            head, separator, remaining = remaining.partition(marker)
            if not separator or not head.strip():
                raise ValueError("missing_d1_predicates")
        if not remaining.strip():
            raise ValueError("missing_d1_predicates")
    return {r["id"]: r for r in rows}


def validate_evidence(evidence):
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError("missing_evidence_registry")
    for key, row in evidence.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(row, dict) or type(row.get("valid")) is not bool or "content" not in row:
            raise ValueError("invalid_evidence_record")


def decide_item(*, satisfied, violated):
    if type(satisfied) is not bool or type(violated) is not bool:
        raise ValueError("proof_flags_must_be_boolean")
    if satisfied and violated:
        raise ValueError("contradictory_proof_flags")
    return "fail" if violated else "pass" if satisfied else "unknown"


def validate_and_aggregate(response, rubric, evidence):
    """Malformed assessments raise; business unknown remains a valid assessment.

    A sample-invalid result retains its business label for diagnosis but cannot
    supply a training reward. Quality never participates in binary aggregation.
    """
    expected = validate_rubric(rubric)
    validate_evidence(evidence)
    schema = response_schema(rubric["task_id"], expected, evidence)
    errors = sorted(Draft202012Validator(schema).iter_errors(response), key=lambda e: str(list(e.path)))
    if errors:
        raise ValueError("invalid_judge_response:" + "; ".join(f"{list(e.path)}: {e.message}" for e in errors))
    if {r["id"] for r in response["items"]} != set(expected):
        raise ValueError("judge_item_coverage")

    def check_refs(refs, *, proof=False):
        if proof and not refs:
            raise ValueError("proof_without_evidence")
        # Every cited record must be usable, not just one valid citation mixed
        # with a failed query. Truncation alone does not invalidate positive facts.
        if proof and any(not evidence[k]["valid"] or evidence[k].get("error") for k in refs):
            raise ValueError("failed_evidence_cannot_prove_label")

    statuses = {}
    for row in response["items"]:
        for key in ("satisfied", "violated"):
            p = row[key]
            check_refs(p["evidence_ids"], proof=p["established"])
        status = decide_item(satisfied=row["satisfied"]["established"], violated=row["violated"]["established"])
        if status == "unknown" and not row["unknown_reason"].strip():
            raise ValueError("unknown_without_specific_gap")
        if status != "unknown" and row["unknown_reason"].strip():
            raise ValueError("decided_item_with_unknown_reason")
        statuses[row["id"]] = status
    sample = response["sample_validity"]
    check_refs(sample["evidence_ids"], proof=sample["status"] != "unknown")
    required = [statuses[rid] for rid, row in expected.items() if row["requirement_level"] == "required"]
    business_label = ("fail" if "fail" in required else "pass" if required and all(s == "pass" for s in required) else "unknown")
    # Source/rubric defects require repair; do not silently judge a different task.
    label = ("invalid" if sample["status"] == "invalid" or response["rubric_issues"] else
             "unknown" if sample["status"] == "unknown" else business_label)
    reward_valid = label in {"pass", "fail"}
    return {"protocol": VERSION, "task_id": rubric["task_id"], "item_statuses": statuses,
            "business_label": business_label, "label": label, "sample_validity": sample,
            "rubric_issues": response["rubric_issues"],
            "required_counts": {s: required.count(s) for s in ("pass", "fail", "unknown")},
            "quality_statuses": {rid: statuses[rid] for rid, row in expected.items() if row["requirement_level"] == "quality"},
            "reward_valid": reward_valid, "reward": int(label == "pass") if reward_valid else None}
