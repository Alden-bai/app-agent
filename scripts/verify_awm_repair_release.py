#!/usr/bin/env python3
"""Verify published databases and recorded statement lineage without rebuilding."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness_awm.database import quote
from harness_awm.database_audit import json_default
from harness_awm.io import file_hash, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=ROOT / 'data/awm-dde80a0-repaired-v1')
    args = parser.parse_args()
    release = args.release.resolve()
    manifest = json.loads((release / 'manifest.json').read_text())
    validation = {r['scenario']: r for r in map(json.loads, (release / 'validation.jsonl').open())}
    lineage = {r['scenario']: r for r in map(json.loads, (release / 'record_lineage.jsonl').open())}
    baselines = {r['scenario']: r for r in map(json.loads, (release / 'baseline.jsonl').open())}
    comparisons = {r['scenario']: r for r in map(json.loads, (release / 'count_comparison.jsonl').open())}
    hashes = []
    for folder, mapping in [(Path(manifest['source']), manifest['source_sha256']), (release, manifest['output_sha256'])]:
        for name, expected in mapping.items():
            hashes.append(dict(path=str(folder / name), matches=file_hash(folder / name)==expected))
    results = []
    for scenario, check in validation.items():
        path = release / 'databases' / (scenario + '.db')
        result = dict(scenario=scenario, database_hash_matches=file_hash(path)==check.get('database_sha256'),
                      traced_insert_records=0, successful_data_statements=0, lineage_errors=[])
        comparison = comparisons[scenario]
        expected = (sum(comparison['original_record_counts'].values()) + len(baselines[scenario]['errors'])
                    + comparison['added_records'] - comparison['merged_records'])
        result['record_accounting_matches'] = expected == sum(check['record_counts'].values())
        with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
            result['integrity_check'] = db.execute('PRAGMA integrity_check').fetchall()
            result['foreign_key_check'] = db.execute('PRAGMA foreign_key_check').fetchall()
            result['record_counts_match'] = all(db.execute(f'SELECT count(*) FROM {quote(t)}').fetchone()[0] == n
                for t,n in check['record_counts'].items())
            for statement in lineage.get(scenario, {}).get('statements', []):
                if statement['phase'] not in ('sample', 'example'):
                    continue
                result['successful_data_statements'] += 1
                if not statement['changes']:
                    result['lineage_errors'].append(dict(statement=statement, error='zero_effect_data_statement'))
                if statement['inserted_table']:
                    cursor = db.execute(f'SELECT * FROM {quote(statement["inserted_table"])} WHERE rowid=?',
                                        (statement['inserted_rowid'],))
                    row = cursor.fetchone()
                    actual = dict(zip([c[0] for c in cursor.description], row)) if row else None
                    actual = json.loads(json.dumps(actual, default=json_default))
                    if actual != statement['final_record']:
                        result['lineage_errors'].append(dict(table=statement['table'], index=statement['index'], error='final_record_mismatch'))
                    result['traced_insert_records'] += 1
        result['passed'] = (result['database_hash_matches'] and result['record_counts_match'] and result['record_accounting_matches']
            and result['integrity_check'] == [('ok',)] and not result['foreign_key_check'] and not result['lineage_errors'])
        results.append(result)
    report = dict(checked_at=datetime.now(timezone.utc).isoformat(),
        scenarios=len(results), passed=sum(r['passed'] for r in results), source_and_release_hashes=hashes,
        lineage_scenarios=len(lineage), traced_insert_records=sum(r['traced_insert_records'] for r in results),
        successful_data_statements=sum(r['successful_data_statements'] for r in results), results=results)
    report['passed_all'] = (manifest.get('release_complete', False)
        and len(results) == len(manifest['selected_scenarios'])
        and all(r['passed'] for r in results) and all(r['matches'] for r in hashes))
    write_json(release / 'published_verification.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('results','source_and_release_hashes')}))
    return int(not report['passed_all'])


if __name__ == '__main__':
    sys.exit(main())
