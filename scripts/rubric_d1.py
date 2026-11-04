#!/usr/bin/env python3
"""Versioned D1 preparation, compilation and validation (offline; no API calls).

Frozen Luna implementations remain untouched. All CLI outputs are exclusive
creates, so a new protocol can never overwrite a historical result by accident.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jsonschema import Draft202012Validator
import rubric_candidates_100 as base
from rubric_pilot import validate_output
from harness_awm.rubric_judge import DECISIONS, VERSION, response_schema, validate_and_aggregate, validate_evidence, validate_rubric

PROMPTS = ROOT / "configs/rubric/d1"


def load(path):
    # Preserve the existing duplicate-JSON-key rejection used for source spans.
    return base.json_spans(Path(path).read_text())[0]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def policy(role):
    return (PROMPTS / (role + ".txt")).read_text() + "\n\n" + (PROMPTS / "policy.txt").read_text()


def selection_schema(inp, pack):
    schema = base.selection_schema(inp, pack)
    item = schema["properties"]["items"]["items"]
    item["properties"]["failure_predicate"] = {"type": "string", "pattern": r"\S"}
    item["required"].append("failure_predicate")
    unresolved = schema["properties"]["unresolved"]["items"]
    for key in ("trigger", "resolution"):
        unresolved["properties"][key] = {"type": "string", "pattern": r"\S"}
        unresolved["required"].append(key)
    return schema


def legacy_projection(out):
    out = copy.deepcopy(out)
    for row in out["items"]:
        row.pop("failure_predicate")
    for row in out["unresolved"]:
        row.pop("trigger")
        row.pop("resolution")
    return out


def validate_selection(out, inp, pack):
    errors = [{"path": "/" + "/".join(map(str, e.path)), "error": e.message}
              for e in Draft202012Validator(selection_schema(inp, pack)).iter_errors(out)]
    if errors:
        return errors
    return base.validate_selection(legacy_projection(out), inp, pack)


def compile_rubric(out, inp, pack):
    errors = validate_selection(out, inp, pack)
    if errors:
        raise ValueError(json.dumps(errors, ensure_ascii=False))
    # Reuse source hash, exact offset, coverage and ID checks, not old decisions.
    result = base.compile_rubric(legacy_projection(out), inp, pack)
    for row, selected in zip(result["rubric"], out["items"]):
        row["criterion"] += "\nF：" + selected["failure_predicate"]
        row["decision"] = {"method": selected["method"], **DECISIONS}
    result["unresolved"] = copy.deepcopy(out["unresolved"])
    errors = validate_output(result, inp)
    if errors:
        raise ValueError("invalid_compilation:" + ",".join(errors))
    validate_rubric(result)
    return result


def request(role, data):
    system = policy(role)
    return {"protocol": VERSION,
            "prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": json.dumps(data, ensure_ascii=False)}]}


def generation_request(inp, previous=None, feedback=None):
    view, pack = base.prepare_input(inp)
    data = {"input": view, "output_schema": selection_schema(inp, pack)}
    if previous is not None:
        data.update(previous_output=previous, revision_feedback=feedback)
    return request("generate", data)


def review_request(inp, selection):
    view, pack = base.prepare_input(inp)
    return request("judge", {"input": view, "selection": selection, "rubric": compile_rubric(selection, inp, pack)})


def runtime_request(bundle):
    # Explicit original input prevents the official adapter's omission of rubric
    # and required/quality levels. No model-level final aggregate is requested.
    for key in ("original", "rubric", "trajectory", "final_answer", "evidence"):
        if key not in bundle:
            raise ValueError("missing_runtime_input:" + key)
    inp, rubric = bundle["original"], bundle["rubric"]
    errors = validate_output(rubric, inp)
    if errors:
        raise ValueError("invalid_runtime_rubric:" + ",".join(errors))
    ids = validate_rubric(rubric)
    validate_evidence(bundle["evidence"])
    return request("runtime_judge", {**bundle, "output_schema": response_schema(inp["task_id"], ids, bundle["evidence"])})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "compile", "review-request", "review-validate"):
        p = sub.add_parser(command)
        p.add_argument("--input", required=True, help="Source-only original.json")
        p.add_argument("--output", required=True, help="New output path; never overwrites")
        if command != "prepare":
            p.add_argument("--selection", required=True)
        if command == "prepare":
            p.add_argument("--previous")
            p.add_argument("--feedback")
        if command == "review-validate":
            p.add_argument("--response", required=True)
    for command in ("judge-request", "judge-validate"):
        p = sub.add_parser(command)
        p.add_argument("--bundle", required=True)
        p.add_argument("--output", required=True)
        if command == "judge-validate":
            p.add_argument("--response", required=True)
    args = parser.parse_args()
    try:
        if args.command.startswith("judge-"):
            bundle = load(args.bundle)
            built = runtime_request(bundle)  # Validate sources even on response-only replay.
            result = built if args.command == "judge-request" else validate_and_aggregate(load(args.response), bundle["rubric"], bundle["evidence"])
        else:
            inp = load(args.input)
            if args.command == "prepare":
                if bool(args.previous) != bool(args.feedback):
                    raise ValueError("revision_requires_previous_and_feedback")
                result = generation_request(inp, load(args.previous) if args.previous else None, load(args.feedback) if args.feedback else None)
            else:
                selection = load(args.selection)
                _, pack = base.prepare_input(inp)
                rubric = compile_rubric(selection, inp, pack)
                if args.command == "compile":
                    result = rubric
                elif args.command == "review-request":
                    result = review_request(inp, selection)
                else:
                    result = base.evaluate(load(args.response), inp["task_id"], {r["id"] for r in rubric["rubric"]}, {r["candidate_id"] for r in pack["candidates"]})
        write(args.output, result)
        if result.get("status") in {"invalid_judge", "rejected"}:
            return 1
    except (ValueError, TypeError, KeyError, IndexError, OSError) as exc:
        parser.exit(2, f"D1 validation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
