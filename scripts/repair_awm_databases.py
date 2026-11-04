#!/usr/bin/env python3
"""Build the reviewed AWM repaired release without modifying upstream assets."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness_awm.database import initialize
from harness_awm.database_audit import baseline, compare_unchanged, json_default
from harness_awm.io import file_hash, write_json


def apply_patches(records, patches):
    for patch in patches:
        obj = records[patch["file"]][patch["scenario"]]
        for key in patch["path"][:-1]:
            obj = obj[key]
        key = patch["path"][-1]
        if obj[key] != patch["before"]:
            raise ValueError(f"patch_source_mismatch: {patch['scenario']} {patch['path']}")
        obj[key] = copy.deepcopy(patch["after"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/awm-dde80a0")
    parser.add_argument("--output", type=Path, default=ROOT / "data/awm-dde80a0-repaired-v1")
    parser.add_argument("--scenarios", nargs="+", help="Optional exact scenario names")
    parser.add_argument("--patches", type=Path, default=Path(__file__).with_name("awm_database_repairs_v1.json"))
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        parser.error("source and output must be independent directories")
    started = datetime.now(timezone.utc).isoformat()
    files = sorted(source.glob("gen_*.jsonl"))
    source_hashes = {p.name: file_hash(p) for p in files}
    records = {}
    for name in ("gen_db.jsonl", "gen_sample.jsonl"):
        entries = [json.loads(line) for line in (source / name).open() if line.strip()]
        records[name] = {r["scenario"]: r for r in entries}
        if len(entries) != len(records[name]):
            raise ValueError("duplicate scenario: " + name)
    patches = json.loads(args.patches.read_text())
    originals = copy.deepcopy(records)
    apply_patches(records, patches)
    names = list(records["gen_db.jsonl"])
    if args.scenarios:
        unknown = set(args.scenarios) - set(names)
        if unknown:
            parser.error("Unknown scenarios: " + ", ".join(sorted(unknown)))
        names = [s for s in names if s in args.scenarios]
    output.mkdir(parents=True, exist_ok=True)
    # An interrupted rebuild must not leave a stale successful release marker.
    write_json(output / 'manifest.json', dict(version='awm-dde80a0-repaired-v1',
        source=str(source), source_sha256=source_hashes, started_at=started,
        selected_scenarios=names, release_complete=False, build_status='building'))
    # The unchanged assets remain byte-identical, independent copies.
    for path in files:
        if path.name in records:
            with (output / path.name).open("w") as f:
                for row in records[path.name].values():
                    if path.name == "gen_sample.jsonl":
                        tables = row["sample_data"].get("tables", [])
                        row["tables_count"] = len(tables)
                        row["inserts_count"] = sum(len(t["insert_statements"]) for t in tables)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        elif not (output / path.name).exists() or file_hash(output / path.name) != source_hashes[path.name]:
            shutil.copyfile(path, output / path.name)
    write_json(output / "patches.json", patches)
    results, baselines, regressions, lineage, counts = [], [], [], [], []
    for index, scenario in enumerate(names):
        original_schema = originals['gen_db.jsonl'][scenario]['db_schema']
        original_sample = originals['gen_sample.jsonl'][scenario]['sample_data']
        old = baseline(original_schema, original_sample)
        baselines.append(dict(scenario=scenario, **old))
        scenario_patches = [p for p in patches if p['scenario'] == scenario]
        result = initialize(records["gen_db.jsonl"][scenario]["db_schema"],
                            records["gen_sample.jsonl"][scenario]["sample_data"],
                            output / "databases" / (scenario + ".db"), audit=not old['complete'] or bool(scenario_patches))
        if old['complete']:
            comparison = compare_unchanged(original_schema, original_sample,
                records['gen_db.jsonl'][scenario]['db_schema'], records['gen_sample.jsonl'][scenario]['sample_data'])
            regressions.append(dict(scenario=scenario, **comparison))
        statements = result.pop('statements')
        if scenario_patches or not old['complete']:
            lineage.append(dict(scenario=scenario, patch_ids=[p['id'] for p in scenario_patches], statements=statements))
        original_count = sum(len(t['insert_statements']) for t in original_sample.get('tables', []))
        final_count = sum(len(t['insert_statements']) for t in records['gen_sample.jsonl'][scenario]['sample_data'].get('tables', []))
        counts.append(dict(scenario=scenario, original_table_counts=old['table_counts'],
            repaired_table_counts=result['table_counts'], original_sample_statements=original_count,
            original_record_counts=old['record_counts'], repaired_record_counts=result['record_counts'],
            repaired_sample_statements=final_count, added_records=final_count-original_count,
            merged_records=sum(p['kind']=='merge' for p in scenario_patches)))
        result.pop("details", None)
        result["scenario"] = scenario
        if result["complete"]:
            result["database_sha256"] = file_hash(output / "databases" / (scenario + ".db"))
        results.append(result)
        if (index + 1) % 100 == 0:
            print(f"Validated {index + 1}/{len(names)}", flush=True)
    for filename, rows in [('validation', results), ('baseline', baselines), ('regression', regressions),
                           ('record_lineage', lineage), ('count_comparison', counts)]:
        with (output / (filename + '.jsonl')).open('w') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=json_default) + '\n')
    summary = dict(scenarios=len(results), complete=sum(r["complete"] for r in results),
                   failed=[r["scenario"] for r in results if not r["complete"]],
                   statement_errors=sum(len(r["errors"]) for r in results),
                   foreign_key_violations=sum(len(r["foreign_key_check"]) for r in results),
                   patches=len(patches), patched_scenarios=len({p["scenario"] for p in patches}),
                   baseline_complete=sum(r['complete'] for r in baselines),
                   baseline_statement_errors=sum(len(r['errors']) for r in baselines),
                   unchanged_regressions=len(regressions),
                   regression_failures=[r['scenario'] for r in regressions if not r['equal']],
                   added_records=sum(r['added_records'] for r in counts),
                   merged_records=sum(r['merged_records'] for r in counts))
    summary['baseline_records'] = sum(sum(r['original_record_counts'].values()) for r in counts)
    summary['repaired_records'] = sum(sum(r['repaired_record_counts'].values()) for r in counts)
    write_json(output / "summary.json", summary)
    write_json(output / "manifest.json", dict(
        version="awm-dde80a0-repaired-v1", source=str(source), source_sha256=source_hashes,
        build_status='validated' if not summary['failed'] and not summary['regression_failures'] else 'failed',
        started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
        sqlite_version=__import__("sqlite3").sqlite_version, selected_scenarios=names,
        release_complete=len(names) == len(records["gen_db.jsonl"]) and not summary["failed"] and not summary['regression_failures'],
        initializer_sha256=file_hash(ROOT / "harness_awm/database.py"),
        patches_sha256=file_hash(args.patches),
        output_sha256={p.name: file_hash(p) for p in output.glob("gen_*.jsonl")},
        time_policy="SQLite relative time evaluated at build; reuse validated snapshots"))
    if any(file_hash(source / name) != digest for name, digest in source_hashes.items()):
        raise RuntimeError("source_changed_during_build")
    print(json.dumps(summary, ensure_ascii=False))
    return int(bool(summary["failed"] or summary['regression_failures']))


if __name__ == "__main__":
    sys.exit(main())
