"""Post-run independent SQL evidence checks; never sent to the official Judge.

Literal row claims are compared against immutable DB snapshots. This is NOT a
claim that every computed statistic or interpretation in the extraction is true.
Semantic selection checks are explicit, task-specific additions below.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from awm_audit.common import digest, now, query, read_json, save_json


def same(left, right):
    if left == right:
        return True
    if isinstance(left, str) and isinstance(right, str):
        try:
            return json.loads(left) == json.loads(right)
        except (ValueError, TypeError):
            pass
        try:
            return datetime.fromisoformat(left) == datetime.fromisoformat(right)
        except ValueError:
            pass
    return False


def inspect_rows(extraction, initial, final):
    tables = {}
    for row in query(final, "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
        info = query(final, f'PRAGMA table_info("{row["name"]}")')
        tables[row['name']] = {'columns': {x['name'] for x in info},
                                'pk': [x['name'] for x in sorted(info, key=lambda x:x['pk']) if x['pk']]}
    checked, errors, unresolved = [], [], []

    def visit(value, path):
        if isinstance(value, list):
            for i, item in enumerate(value):
                visit(item, path + [str(i)])
        elif isinstance(value, dict):
            meaningful = set(value) - {'id','created_at','updated_at','name','status'}
            candidates = []
            context = '_'.join(path).lower()
            for table, schema in tables.items():
                overlap = set(value) & schema['columns']
                if len(overlap) < 2 or not (meaningful & overlap):
                    continue
                pk = schema['pk']
                if not pk or not all(k in value and not isinstance(value[k], (dict,list)) for k in pk):
                    continue
                # Naming is only a binding heuristic; require a unique best
                # match and record that binding with each checked row.
                token = table.rstrip('s')
                hint = int(table in context or (len(token) > 3 and token in context))
                candidates.append((hint, len(overlap), table, overlap))
            candidates.sort(reverse=True)
            if candidates and (len(candidates)==1 or candidates[0][:2] > candidates[1][:2]):
                _, _, table, overlap = candidates[0]
                pk = tables[table]['pk']
                source = initial if re.search(r'(^|_)(initial|before|original|init)(_|$)', context) else final
                fields = [f'"{k}" IS ?' for k in pk]
                records = query(source, f'SELECT * FROM "{table}" WHERE ' + ' AND '.join(fields), [value[k] for k in pk])
                claim = {'path': '.'.join(path), 'table':table,'snapshot':source.name,'pk':{k:value[k] for k in pk}}
                if not records:
                    errors.append({**claim,'error':'claimed row absent from snapshot'})
                else:
                    # Only literal database-column assertions are checked.
                    fields_checked = []
                    for column in sorted(overlap):
                        if isinstance(value[column], (dict,list)):
                            continue
                        fields_checked.append(column)
                        if not same(value[column], records[0][column]):
                            errors.append({**claim,'column':column,'claimed':value[column],'actual':records[0][column]})
                    checked.append({**claim,'columns':fields_checked})
            elif candidates:
                unresolved.append({'path':'.'.join(path),'reason':'ambiguous table binding'})
            for key,item in value.items():
                visit(item,path+[str(key)])
    visit(extraction, [])
    return {'rows_checked':len(checked),'literal_fields_checked':sum(len(c['columns']) for c in checked),
            'bindings':checked,'contradictions':errors,'unresolved_bindings':unresolved,
            'binding_method':'unique context/column match; inspect bindings before treating contradictions as confirmed'}


def semantic_selection(task_key, extraction, rule_artifact):
    checks=[]
    spec=rule_artifact.get('spec',{})
    if task_key=='inventory_management_6:2':
        expected=[r['id'] for r in spec.get('answer_expected',[])]
        for key in ['initial_query_result','final_query_result']:
            value=extraction.get(key,{}) if isinstance(extraction,dict) else {}
            if isinstance(value,dict) and isinstance(value.get('rows'),list):
                actual=[r.get('id') for r in value['rows']]
                checks.append({'name':'independent filtered and sorted vehicle IDs: '+key,
                               'expected':expected,'actual':actual,'passed':actual==expected})
    if task_key=='inventory_management_6:9' and isinstance(extraction,dict):
        expected=[]
        for rule in spec.get('rules',[]):
            if rule['name']=='exact eligible vehicle links':
                expected=json.loads(rule['params'][1])
        key='expected_eligible_vehicles_as_of_2025_01_15'
        if key in extraction:
            actual=[v.get('vehicle_id') for v in extraction[key]]
            checks.append({'name':'eligibility uses current runtime date, not silently fixed 2025-01-15',
                           'expected':sorted(expected),'actual':sorted(actual),
                           'passed':sorted(expected)==sorted(actual)})
    if task_key=='social_networking_8:9' and isinstance(extraction,dict):
        claimed=extraction.get('expected',{})
        start_rule=next((r for r in spec.get('rules',[]) if r['name']=='next Monday and seven days'),None)
        creative_rule=next((r for r in spec.get('rules',[]) if r['name']=='ad_creatives required fields'),None)
        if start_rule:
            start=start_rule['params'][0]
            end=(datetime.fromisoformat(start)+timedelta(days=7)).date().isoformat()
            for key,expected in [('start_date',start),('end_date',end)]:
                if key in claimed:
                    checks.append({'name':'next Monday schedule at frozen episode date: '+key,
                                   'expected':expected,'actual':claimed[key],
                                   'passed':str(claimed[key])[:10]==expected})
        if creative_rule and 'latest_media_id' in claimed:
            expected=creative_rule['params'][0]
            checks.append({'name':'latest relevant product photo, not unrelated newer hiking photo',
                           'expected':expected,'actual':claimed['latest_media_id'],
                           'passed':claimed['latest_media_id']==expected})
    return checks


def review(root):
    root=Path(root).resolve();states=read_json(root/'task_states.json');details=[]
    for key,state in states.items():
        rule = read_json(root/state['rule_path']) if state.get('rule_path') else {}
        for p in state.get('results',[]):
            path=root/p;case=read_json(path)
            extraction=case.get('sql_extraction')
            if not extraction or extraction.get('status')!='valid':
                continue
            folder=root/case['path']
            literal=inspect_rows(extraction['details'],folder/'initial.db',folder/'final.db')
            selection=semantic_selection(key,extraction['details'],rule)
            result={'reviewed_at':now(),'reviewer_source_sha256':digest(Path(__file__)),
                    'status':'partially_checked','literal_fact_check':literal,'semantic_selection_checks':selection,
                    'not_checked':'All other computed aggregates, boolean summaries, omissions, and interpretation require independent review.'}
            if any(not c['passed'] for c in selection):
                result['status']='confirmed_selection_error'
            elif literal['contradictions']:
                result['status']='possible_literal_error_needs_binding_review'
            elif not literal['rows_checked'] and not selection:
                result['status']='needs_review'
            save_json(folder/'sql_evidence_review.json',result)
            case['sql_evidence_review']={'status':result['status'],'path':str((folder/'sql_evidence_review.json').relative_to(root)),
                                         'literal_fields_checked':literal['literal_fields_checked']}
            save_json(path,case)
            details.append({'task':key,'case':case['case'],**case['sql_evidence_review']})
    save_json(root/'sql_evidence_reviews.json',details)
    return details


if __name__=='__main__':
    p=argparse.ArgumentParser(description='独立核对SQL证据中的数据库行事实及显式选择条件，保留未复核范围。')
    p.add_argument('--run-dir',required=True)
    args=p.parse_args()
    result=review(args.run_dir)
    from collections import Counter
    print(dict(Counter(r['status'] for r in result)))
