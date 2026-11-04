"""Independent review-task audit; never imports or reads official verifiers.

Only task/public-contract values and prior tool results enter recipes. SQL below
is exclusively audit-side. The task asks for a new review, not exactly one review.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from awm_audit.common import query

SKU = 'WHEY-ISOLATE-CHOC-2LB'
TEXT = 'Great taste, mixes easily, and no stomach issues even with my lactose sensitivity'


def literal(value, quote=None):
    return {'$literal': value, 'quote': str(value) if quote is None else quote}


def public(value, citation):
    return {'$public': value, 'citation': citation}


def build(task, initial_db):
    text = task['task']
    for fragment in (SKU, TEXT, '5-star', 'pending', 'new product review'):
        if fragment not in text:
            raise ValueError('Reference recipe does not match the original task text')
    body = {'sku_code': literal(SKU), 'rating': literal(5, '5-star'),
            'review_text': literal(TEXT), 'status': literal('pending')}
    create = {'tool': 'create_review', 'arguments': {'body': body}}
    requirements = [
        {'id': 'new', 'quote': 'Submit a new product review', 'kind': 'operation'},
        {'id': 'sku', 'quote': "SKU '" + SKU + "'", 'kind': 'object'},
        {'id': 'rating', 'quote': '5-star rating', 'kind': 'value'},
        {'id': 'text', 'quote': TEXT, 'kind': 'text'},
        {'id': 'status', 'quote': "moderation status to 'pending'", 'kind': 'state'},
    ]
    def case(name, calls, expected, violates=(), role='counterexample'):
        return {'name': name, 'calls': calls, 'answer': 'Requested review submitted.' if expected else 'Operation completed.',
                'expected': expected, 'violates': list(violates), 'role': role}
    cases = [case('positive', [create], True, role='positive'),
             case('no_action_false_completion', [], False, [r['id'] for r in requirements])]
    for name, field, value, requirement in [
        ('wrong_rating', 'rating', public(1, 'gen_spec.jsonl:create_review:rating description "Star rating from 1 to 5"; lower bound 1'), 'rating'),
        ('partial_text', 'review_text', literal('Great taste'), 'text'),
        ('wrong_status', 'status', literal('new'), 'status'),
    ]:
        step = copy.deepcopy(create)
        step['arguments']['body'][field] = value
        cases.append(case(name, [step], False, [requirement]))
    wrong = copy.deepcopy(create)
    wrong['arguments']['body']['sku_code'] = {'$result': 0, 'path': ['skus', 0, 'sku_code']}
    cases.append(case('wrong_existing_sku', [
        {'tool': 'search_skus', 'arguments': {'sku_code': public('PRE-ENERGY-300GFRT',
          'gen_spec.jsonl:search_skus:request_params.sku_code.example = PRE-ENERGY-300GFRT')}}, wrong], False, ['sku']))
    # Both expose common false assumptions without changing the original goal.
    default = copy.deepcopy(create)
    del default['arguments']['body']['status']
    cases.append(case('equivalent_default_pending', [default], True, role='positive'))
    cases.append(case('challenge_duplicate_review', [copy.deepcopy(create), copy.deepcopy(create)], True, role='challenge'))
    # A reachable near-target rating challenge; the lower legal boundary is covered above.
    boundary = copy.deepcopy(create)
    boundary['arguments']['body']['rating'] = {'$compute': 'subtract', 'args': [
        literal(5, '5-star'),
        public(1, 'gen_spec.jsonl:create_review:rating description \"Star rating from 1 to 5\"; lower bound 1')]}
    cases.append(case('challenge_near_target_rating', [boundary], False, ['rating'], role='challenge'))
    return {'task_key': 'e_commerce_10:7', 'requirements': requirements, 'cases': cases,
            'answer_requirement': 'No final-answer content required by the original task.',
            'process_requirement': 'Creation must be a real exposed tool call; SQL is audit-side only.',
            'forbidden_side_effects': {'task_explicit': [], 'note': 'No extra exactly-one or untouched-table task constraint is invented.'},
            'equivalence': 'At least one new review satisfies all requested fields on the same row; omitted status is allowed if the public default persists pending.',
            'automatic_review': {'source': 'automatic', 'basis': 'Literal task requirements and public create_review contract; never official verifier labels.'}}


def evaluate(spec, initial_db, final_db, answer, trajectory):
    try:
        old = {r['id'] for r in query(initial_db, 'SELECT id FROM reviews')}
        users = query(final_db, 'SELECT id FROM users WHERE username=?', ['current_user'])
        if len(users) != 1:
            raise ValueError('Current user is not uniquely resolved')
        uid = users[0]['id']
        rows = [r for r in query(final_db, '''SELECT r.*, s.sku_code FROM reviews r
                  LEFT JOIN sku s ON s.id=r.sku_id ORDER BY r.id''') if r['id'] not in old and r['user_id'] == uid]
        predicates = {'new': lambda r: True, 'sku': lambda r: r['sku_code'] == SKU,
                      'rating': lambda r: type(r['rating']) is int and r['rating'] == 5,
                      'text': lambda r: r['review_text'] == TEXT, 'status': lambda r: r['status'] == 'pending'}
        checks = [{'requirement_id': k, 'passed': any(f(r) for r in rows),
                   'evidence': {'new_review_ids': [r['id'] for r in rows]}}
                  for k, f in predicates.items()]
        witnesses = [r['id'] for r in rows if all(f(r) for f in predicates.values())]
        checks.append({'requirement_id': 'same_row_binding', 'passed': bool(witnesses),
                       'evidence': {'matching_new_review_ids': witnesses}})
        return {'status': 'valid', 'correct': bool(witnesses), 'checks': checks,
                'facts': {'new_reviews': rows}, 'review_source': 'automatic'}
    except Exception as exc:
        return {'status': 'unknown', 'correct': None, 'checks': [], 'error': f'{type(exc).__name__}: {exc}'}


def isolation_probe():
    """A real legal write; audit checks B snapshots if no reviews-list API exists."""
    task = {'task': "Submit a new product review for SKU '" + SKU + "' with a 5-star rating and the text '" + TEXT + "', and set the review moderation status to 'pending'."}
    return build(task, None)['cases'][0]


def sql_review(entry, extraction, initial_db, final_db):
    """Recompute *every* exported field without invoking released verifier code.

    This certifies evidence for these frozen snapshots, not arbitrary future
    states. Global initial identities prevent moved old reviews counting as new.
    Unexpected fields, omitted rows, summaries, nulls and timestamps all fail.
    """
    aspects = {k: False for k in ('object', 'join', 'filter', 'aggregate', 'null', 'omission')}
    checks = []
    try:
        if entry.get('scenario') != 'e_commerce_10' or entry.get('task_idx') != 7:
            raise ValueError('Wrong task verifier binding')
        if isinstance(extraction, dict) and 'details' in extraction and 'status' in extraction:
            extraction = extraction['details']
        def state(path):
            skus = query(path, 'SELECT id FROM sku WHERE sku_code=?', [SKU])
            users = query(path, 'SELECT id FROM users WHERE username=?', ['current_user'])
            if len(skus) != 1 or len(users) != 1:
                raise ValueError('Target SKU/current user not uniquely resolved')
            sid, uid = skus[0]['id'], users[0]['id']
            # Public create_review contract fixes the current user; confirm its
            # DB identity rather than trusting a verifier-supplied assertion.
            if uid != 1:
                raise ValueError('Current-user mapping differs from exposed contract')
            rows = query(path, '''SELECT r.id,r.user_id,r.sku_id,r.rating,r.review_text,r.status,
                         r.created_at,r.updated_at FROM reviews r JOIN sku s ON s.id=r.sku_id
                         JOIN users u ON u.id=r.user_id WHERE s.sku_code=? AND u.username=? ORDER BY r.id''',
                         [SKU, 'current_user'])
            return {'sku_id': sid, 'user_id': uid, 'reviews': rows}
        before, after = state(initial_db), state(final_db)
        all_old_ids = {r['id'] for r in query(initial_db, 'SELECT id FROM reviews')}
        old_by_id = {r['id']: r for r in before['reviews']}
        added = [r for r in after['reviews'] if r['id'] not in all_old_ids]
        changed = [{'id': r['id'], 'initial': old_by_id[r['id']], 'final': r}
                   for r in after['reviews'] if r['id'] in old_by_id and
                   any(r[k] != old_by_id[r['id']][k] for k in ('rating', 'review_text', 'status'))]
        def matches(r):
            return type(r['rating']) is int and r['rating'] == 5 and r['review_text'] == TEXT and r['status'] == 'pending'
        matching = [r for r in added if matches(r)]
        matching_changed = [r for r in changed if matches(r['final'])]
        expected = {'initial_state': before, 'final_state': after, 'new_reviews': added,
                    'modified_reviews': changed, 'matching_new_reviews': matching,
                    'matching_modified_reviews': matching_changed,
                    'summary': {'target_sku_code': SKU, 'target_rating': 5, 'target_text': TEXT,
                                'target_status': 'pending', 'target_username': 'current_user',
                                'initial_sku_id': before['sku_id'], 'final_sku_id': after['sku_id'],
                                'initial_user_id': before['user_id'], 'final_user_id': after['user_id'],
                                'initial_review_count_for_user_and_sku': len(before['reviews']),
                                'final_review_count_for_user_and_sku': len(after['reviews']),
                                'new_reviews_count': len(added), 'matching_new_reviews_count': len(matching),
                                'matching_modified_reviews_count': len(matching_changed)}}
        canonical = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
        checks.append({'name': 'complete_export_key_set', 'passed': isinstance(extraction, dict) and set(extraction) == set(expected)})
        for key, value in expected.items():
            actual = extraction.get(key) if isinstance(extraction, dict) else None
            checks.append({'name': key, 'passed': canonical(actual) == canonical(value),
                           'expected': value, 'actual': actual})
        passed = all(c['passed'] for c in checks)
        return {'status': 'fully_checked' if passed else 'confirmed_evidence_mismatch',
                'aspects': {k: passed for k in aspects}, 'checks': checks,
                'source': 'automatic', 'method': 'Independent complete relational export reconstruction; global initial review IDs for newness.',
                'coverage': 'All original extraction keys, all relevant rows/columns, joins, selection, counts, target literals, nulls, timestamps and omissions.'}
    except Exception as exc:
        return {'status': 'needs_review', 'aspects': aspects, 'checks': checks,
                'error': f'{type(exc).__name__}: {exc}', 'source': 'automatic'}


def public_contracts(spec):
    """Return only constants verified against the actual frozen public spec."""
    api = spec.get('api_spec', spec)
    operations = {}
    for group in api.get('api_groups', []):
        for endpoint in group.get('endpoints', []):
            key = endpoint.get('operation_id')
            if key in operations:
                raise ValueError('Duplicate public operation contract')
            operations[key] = endpoint
    registry = {}
    rating = operations.get('create_review', {}).get('request_params', {}).get('rating', {})
    if rating.get('type') == 'integer' and rating.get('description') == 'Star rating from 1 to 5':
        registry['gen_spec.jsonl:create_review:rating description "Star rating from 1 to 5"; lower bound 1'] = 1
    sku = operations.get('search_skus', {}).get('request_params', {}).get('sku_code', {})
    if sku.get('type') == 'string' and sku.get('example') == 'PRE-ENERGY-300GFRT':
        registry['gen_spec.jsonl:search_skus:request_params.sku_code.example = PRE-ENERGY-300GFRT'] = sku['example']
    return registry
