"""Lossless normalization for Judge output only; source evidence stays strict."""
import json

from .io import canonical, write_json


def parse_judge_output(text, audit_path):
    changes = []
    body = text.strip()
    if body.startswith("```json\n") and body.endswith("\n```"):
        body = body[8:-4]
        changes.append({"kind": "json_fence"})

    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                # Canonical comparison also distinguishes true from 1.
                if canonical(result[key]) != canonical(value):
                    raise ValueError("conflicting_duplicate_json_key:" + key)
                changes.append({"kind": "identical_duplicate_key", "key": key})
            else:
                result[key] = value
        return result

    def invalid(value):
        raise ValueError("invalid_json_constant:" + value)

    value = json.loads(body, object_pairs_hook=pairs, parse_constant=invalid)
    if changes:
        write_json(audit_path, {"changes": changes, "normalized": value,
                                "semantic_reassessment": False})
    return value
