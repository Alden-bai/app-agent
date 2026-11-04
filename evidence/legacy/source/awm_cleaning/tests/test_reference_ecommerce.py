import json
import sqlite3

import pytest

from awm_cleaning.reference_ecommerce import SKU, TEXT, build, evaluate, sql_review


@pytest.fixture
def databases(tmp_path):
    paths = [tmp_path / 'initial.db', tmp_path / 'final.db']
    for path in paths:
        with sqlite3.connect(path) as db:
            db.executescript('''CREATE TABLE users(id INTEGER PRIMARY KEY,username TEXT);
              CREATE TABLE sku(id INTEGER PRIMARY KEY,sku_code TEXT);
              CREATE TABLE reviews(id INTEGER PRIMARY KEY,user_id INTEGER,sku_id INTEGER,rating INTEGER,
              review_text TEXT,status TEXT,created_at TEXT,updated_at TEXT);
              INSERT INTO users VALUES(1,'current_user');''')
            db.execute('INSERT INTO sku VALUES(5,?)', [SKU])
    return paths


def add(path, rid=1, **changes):
    values = dict(id=rid, user_id=1, sku_id=5, rating=5, review_text=TEXT,
                  status='pending', created_at='2026-10-22 00:00:00', updated_at=None)
    values.update(changes)
    with sqlite3.connect(path) as db:
        db.execute('INSERT INTO reviews VALUES(?,?,?,?,?,?,?,?)', list(values.values()))


def test_noop_and_existing_review_do_not_satisfy_newness(databases):
    initial, final = databases
    add(initial); add(final)
    assert evaluate({}, initial, final, 'completed', [])['correct'] is False


def test_duplicate_reviews_are_not_an_invented_quantity_violation(databases):
    initial, final = databases
    add(final); add(final, 2)
    assert evaluate({}, initial, final, '', [])['correct'] is True


def test_each_review_field_must_match_on_same_new_row(databases):
    initial, final = databases
    add(final, review_text='wrong')
    add(final, 2, rating=1)
    result = evaluate({}, initial, final, '', [])
    assert result['correct'] is False
    assert all(c['passed'] for c in result['checks'] if c['requirement_id'] != 'same_row_binding')


def extraction(initial, final):
    def state(path):
        with sqlite3.connect(path) as db:
            db.row_factory = sqlite3.Row
            rows = [dict(r) for r in db.execute('SELECT * FROM reviews ORDER BY id')]
        return {'sku_id': 5, 'user_id': 1, 'reviews': rows}
    before, after = state(initial), state(final)
    old = {r['id'] for r in before['reviews']}
    added = [r for r in after['reviews'] if r['id'] not in old]
    matching = [r for r in added if r['rating'] == 5 and r['review_text'] == TEXT and r['status'] == 'pending']
    return {'initial_state': before, 'final_state': after, 'new_reviews': added,
            'modified_reviews': [], 'matching_new_reviews': matching, 'matching_modified_reviews': [],
            'summary': {'target_sku_code': SKU, 'target_rating': 5, 'target_text': TEXT,
                        'target_status': 'pending', 'target_username': 'current_user',
                        'initial_sku_id': 5, 'final_sku_id': 5, 'initial_user_id': 1, 'final_user_id': 1,
                        'initial_review_count_for_user_and_sku': len(before['reviews']),
                        'final_review_count_for_user_and_sku': len(after['reviews']),
                        'new_reviews_count': len(added), 'matching_new_reviews_count': len(matching),
                        'matching_modified_reviews_count': 0}}


def test_complete_evidence_and_nulls_checked(databases):
    initial, final = databases
    add(final)
    value = extraction(initial, final)
    result = sql_review({'scenario': 'e_commerce_10', 'task_idx': 7}, value, initial, final)
    assert result['status'] == 'fully_checked'
    assert all(result['aspects'].values())
    value['final_state']['reviews'][0]['updated_at'] = 'not-null'
    assert sql_review({'scenario': 'e_commerce_10', 'task_idx': 7}, value, initial, final)['status'] != 'fully_checked'


@pytest.mark.parametrize('mutation', ['omission', 'extra', 'count', 'boolean_count', 'target'])
def test_incomplete_or_false_sql_export_is_blocked(databases, mutation):
    initial, final = databases
    add(final)
    value = extraction(initial, final)
    if mutation == 'omission':
        value['new_reviews'] = []
    elif mutation == 'extra':
        value['unreviewed_claim'] = True
    elif mutation == 'count':
        value['summary']['new_reviews_count'] = 2
    elif mutation == 'boolean_count':
        value['summary']['new_reviews_count'] = True
    else:
        value['summary']['target_sku_code'] = 'wrong'
    result = sql_review({'scenario': 'e_commerce_10', 'task_idx': 7}, value, initial, final)
    assert result['status'] == 'confirmed_evidence_mismatch'
