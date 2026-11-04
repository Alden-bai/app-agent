#!/usr/bin/env python3
"""Build source-only D1 requests from the reviewed 20-case pack; never dispatch."""
import argparse
import hashlib
from pathlib import Path

import rubric_d1 as d1


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    manifest = d1.load(source / "selection_manifest.json")
    # Source copy identities were recorded by the material-pack builder.
    provenance = d1.load(source / "provenance.json")
    rows = []
    output.mkdir(parents=True, exist_ok=False)
    for task in manifest["tasks"]:
        case = f'{task["number"]:02d}_{task["id"]}'
        folder = source / "cases" / case / "sources"
        refs = {}
        for name in ("gen_tasks", "gen_spec", "gen_db"):
            path = folder / (name + ".json")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            original = str(Path(manifest["source_run"]) / "inputs" / task["id"] / path.name)
            if provenance["sources"].get(original) != digest:
                raise ValueError("source_copy_hash_mismatch:" + str(path))
            refs[name] = {"path": str(path), "sha256": digest}
        scenario, index = task["task_id"].rsplit(":", 1)
        task_data, spec, db = (d1.load(refs[n]["path"]) for n in ("gen_tasks", "gen_spec", "gen_db"))
        if any(d["scenario"] != scenario for d in (task_data, spec, db)):
            raise ValueError("source_scenario_mismatch")
        instruction = task_data["tasks"][int(index)]
        sources = [{"id": "task", "kind": "task", "text": instruction},
                   {"id": "contract:" + scenario, "kind": "public_contract", "text": Path(refs["gen_spec"]["path"]).read_text()}]
        for table in db["db_schema"]["tables"]:
            sources.append({"id": "schema:" + table["name"], "kind": "schema",
                            "text": table["ddl"] + "\n" + "\n".join(table.get("indexes", []))})
        original = {"task_id": task["task_id"], "task": instruction, "sources": sources, "quality_policy": None,
                    "available_evidence_types": ["本次执行轨迹、真实返回及最终答复", "初末数据库快照", "记录的执行时钟"],
                    "evidence_note": "仅提供将来可取证类型，不提供真实数据、历史rubric或判分。"}
        d1.write(output / case / "original.json", original)
        request = d1.generation_request(original)
        d1.write(output / case / "generate_request.json", request)
        rows.append({"case": case, "task_id": task["task_id"], "sources": refs,
                     "prompt_sha256": request["prompt_sha256"], "request_ref": str(output / case / "generate_request.json")})
    report = {"protocol": d1.VERSION, "kind": "source_only_development_requests", "requests_prepared": len(rows),
              "model_calls": 0, "human_labels_written": 0, "original_rubrics_modified": 0,
              "independent_evaluation": False, "cases": rows}
    d1.write(output / "manifest.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(d1.ROOT / "runs/judge_development20_20261103"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = prepare(args.source, args.output)
    print(f'{report["requests_prepared"]} requests prepared; model calls: 0')
