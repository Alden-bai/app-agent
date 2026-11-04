import sqlite3

from awm_audit.common import backup
from awm_audit.rules.review_evidence import inspect_rows,semantic_selection


def test_literal_evidence_compares_independent_snapshot(tmp_path):
    initial=tmp_path/'initial.db';final=tmp_path/'final.db'
    with sqlite3.connect(initial) as db:
        db.executescript("CREATE TABLE vehicles(id TEXT PRIMARY KEY,vin TEXT,list_price REAL);INSERT INTO vehicles VALUES('v','VIN',10)")
    backup(initial,final)
    with sqlite3.connect(final) as db:
        db.execute('UPDATE vehicles SET list_price=20')
    good={'initial_vehicle':{'id':'v','vin':'VIN','list_price':10},'final_vehicle':{'id':'v','vin':'VIN','list_price':20}}
    result=inspect_rows(good,initial,final)
    assert result['literal_fields_checked']==6
    assert result['contradictions']==[]
    bad={'final_vehicle':{'id':'v','vin':'VIN','list_price':999}}
    assert inspect_rows(bad,initial,final)['contradictions'][0]['actual']==20


def test_selection_proof_not_heuristic_row_matching():
    rule={'spec':{'rules':[{'name':'exact eligible vehicle links','params':[2,'["v1","v2"]']}]}}
    raw={'expected_eligible_vehicles_as_of_2025_01_15':[{'vehicle_id':'v1'}]}
    checks=semantic_selection('inventory_management_6:9',raw,rule)
    assert checks[0]['passed'] is False


def test_ad_schedule_and_photo_semantics_detect_stale_expectations():
    from awm_audit.rules.review_evidence import semantic_selection
    artifact = {'spec': {'rules': [
        {'name': 'next Monday and seven days', 'params': ['2026-10-26']},
        {'name': 'ad_creatives required fields', 'params': [7]}]}}
    extraction = {'expected': {'start_date': '2024-12-23 00:00:00',
                               'end_date': '2024-12-30 00:00:00', 'latest_media_id': 2}}
    checks = semantic_selection('social_networking_8:9', extraction, artifact)
    assert len(checks) == 3 and all(c['passed'] is False for c in checks)
