#!/usr/bin/env python3
"""Read-only acceptance and request accounting for the durable screening run."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import environment_screen_remaining as screen


def quantiles(values):
    if not values:
        return None
    values = sorted(values)
    return dict(count=len(values), minimum=values[0], median=values[(len(values)-1)//2],
        p90=values[math.ceil(.90*len(values))-1], p95=values[math.ceil(.95*len(values))-1],
        maximum=values[-1], mean=sum(values)/len(values))


def controller_usage(run, manifest):
    master = screen.usage(manifest['master_thread_id'])
    master['matches_verifier_model_configuration'] = master.pop('actual_config_verified')
    master['verifier_configuration_requirement_applies_to_master'] = False
    scope_path = run / 'master_scope.json'
    if not scope_path.exists() or not master.get('rollout_ref'):
        return dict(**master, attribution_exact=False, ongoing=True,
                    attribution_missing='No explicit controller-turn scope or rollout; thread totals are not attributed to this audit.')
    scope = set(screen.read_json(scope_path)['root_turn_ids'])
    requests, complete, cumulative = {}, set(), {}
    for event in screen.stream_events(master['rollout_ref']):
        p = event.get('payload', {})
        root = p.get('root_turn_id') or p.get('turn_id')
        if root not in scope:
            continue
        if event.get('type') == 'token_usage_record' and p.get('response_id'):
            requests[p['response_id']] = dict(response_id=p['response_id'], root_turn_id=root, usage=p['usage'])
            cumulative[root] = p.get('turn_token_usage')
        if event.get('type') == 'event_msg' and p.get('type') == 'task_complete':
            complete.add(root)
    totals = {k: sum(p['usage'].get(k, 0) for p in requests.values()) for k in screen.FIELDS} if requests else None
    if totals:
        totals.update(uncached_input_tokens=totals['input_tokens']-totals['cached_input_tokens'],
                      total_tokens=totals['input_tokens']+totals['output_tokens'])
    cross_checks = {}
    for turn in scope:
        summed = {k:sum(p['usage'].get(k,0) for p in requests.values() if p['root_turn_id']==turn) for k in screen.FIELDS}
        cross_checks[turn] = bool(cumulative.get(turn) and all(summed[k]==cumulative[turn].get(k,0) for k in screen.FIELDS))
    return dict(thread_id=master.get('thread_id'), model=master.get('model'), reasoning_effort=master.get('reasoning_effort'),
        verifier_configuration_requirement_applies_to_master=False, rollout_ref=master['rollout_ref'],
        root_turn_ids=sorted(scope), request_count=len(requests), requests=list(requests.values()), usage=totals,
        turn_cumulative_cross_checks=cross_checks, attribution_exact=True, ongoing=bool(scope-complete),
        usage_complete=not(scope-complete) and all(cross_checks.values()),
        missing_usage_parts=[turn for turn, matches in cross_checks.items() if not matches],
        note='Only explicitly scoped controller turns; unrelated later chat turns excluded. Active-turn final responses may not yet be persisted.')


def audit(run):
    manifest = screen.verify_run(run)
    errors, revisions, distributions = [], [], defaultdict(Counter)
    for r in screen.rows(run / 'inventory.jsonl') + screen.read_json(run / 'regression_inventory.json'):
        case = run / 'inputs' / r['case_id']
        try:
            screen.task_spec(case)
        except Exception as exc:
            errors.append(dict(case_id=r['case_id'], issue='frozen_input', error=repr(exc)))
    with screen.db_open(run) as db:
        tasks = [dict(r) for r in db.execute('SELECT * FROM tasks ORDER BY task_id')]
        jobs = [dict(r) for r in db.execute('SELECT * FROM jobs ORDER BY id')]
    jobmap = {r['id']: r for r in jobs}
    selected = {r['case_id'] for r in screen.read_json(run / 'pass_sample.json')['selected']} if (run / 'pass_sample.json').exists() else set()
    for r in tasks:
        try:
            first = None
            if r['first_ref']:
                first, _ = screen.validate_decision(Path(r['first_ref']).parent, Path(r['first_ref']))
            if r['status'] != 'completed':
                continue
            final, refs = screen.validate_decision(Path(r['final_ref']).parent, Path(r['final_ref']))
            if not first:
                raise ValueError('completed without first judgment')
            required = 'exclude_review' if first['decision'] == 'exclude' else 'supplement' if first['decision'] == 'review' else 'pass_audit' if r['case_id'] in selected else None
            if required and jobmap[r['case_id'] + '-' + required]['status'] != 'completed':
                raise ValueError('required quality check incomplete')
            if required in {'exclude_review', 'pass_audit'}:
                case = Path(jobmap[r['case_id'] + '-' + required]['result_path']).parent
                seal = screen.read_json(case / 'independent.seal.json')
                if seal['sha256'] != screen.file_hash(case / 'independent.frozen.json'):
                    raise ValueError('independent verdict seal mismatch')
            if first['decision'] != final['decision']:
                revisions.append(dict(case_id=r['case_id'], task_id=r['task_id'], first=first['decision'],
                    final=final['decision'], first_ref=r['first_ref'], final_ref=r['final_ref'], reason=r['reason']))
        except Exception as exc:
            errors.append(dict(case_id=r['case_id'], issue='decision_or_required_check', error=repr(exc)))
    columns = ['case_id', 'task_id', 'scenario', 'status', 'rubric_accepted', 'first_decision',
               'first_requirements', 'first_verification_level', 'final_decision', 'requirements',
               'verification_level', 'category', 'summary', 'fault_mechanism', 'alternatives_checked',
               'evidence_refs', 'first_ref', 'final_ref', 'unfinished_reason']
    with (run / 'results.full.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for task in tasks:
            first = screen.read_json(task['first_ref']) if task['first_ref'] else {}
            final = screen.read_json(task['final_ref']) if task['status'] == 'completed' else {}
            defect = final.get('defect') or {}
            writer.writerow(dict(case_id=task['case_id'], task_id=task['task_id'], scenario=task['scenario'],
                status=task['status'], rubric_accepted=True, first_decision=first.get('decision'),
                first_requirements=json.dumps(first.get('requirements'), ensure_ascii=False),
                first_verification_level=first.get('verification_level'), final_decision=final.get('decision'),
                requirements=json.dumps(final.get('requirements'), ensure_ascii=False),
                verification_level=final.get('verification_level'), category=final.get('category'),
                summary=final.get('summary'), fault_mechanism=defect.get('mechanism'),
                alternatives_checked=defect.get('alternatives_checked'),
                evidence_refs=json.dumps(screen.evidence_refs(final), ensure_ascii=False),
                first_ref=task['first_ref'], final_ref=task['final_ref'], unfinished_reason=task['reason']))
    requests, missing, session_sums, intervals, missing_intervals = {}, [], [], [], []
    for p in sorted((run / 'sessions').glob('*/*/usage.json')):
        jobid = p.parent.parent.name
        cid = jobid.split('-', 1)[0]
        record = screen.read_json(p)
        if record.get('missing'):
            missing.append(dict(ref=str(p), reason=record['missing']))
        session_sums.append(dict(ref=str(p), matches=record.get('sum_matches_cumulative')))
        if record.get('usage') and not record.get('sum_matches_cumulative'):
            missing.append(dict(ref=str(p), reason='request sum differs from available session cumulative value; known requests retained, difference unassigned'))
        start, end = None, None
        if record.get('rollout_ref'):
            for event in screen.stream_events(record['rollout_ref']):
                value = event.get('payload', {})
                if event.get('type') == 'event_msg' and value.get('type') == 'task_started':
                    start = start or event['timestamp']
                if event.get('type') == 'event_msg' and value.get('type') == 'task_complete':
                    end = event['timestamp']
        if start and end:
            intervals.append(dict(thread_id=record.get('thread_id'), start=start, end=end))
        else:
            missing_intervals.append(dict(ref=str(p), start=start, end=end))
        for req in record.get('requests', []):
            rid = req['response_id']
            if rid in requests and requests[rid]['usage'] != req['usage']:
                errors.append(dict(issue='conflicting_response_usage', response_id=rid))
            if rid not in requests:
                requests[rid] = dict(**req, job_id=jobid, case_id=cid)
                distributions[cid].update({k: req['usage'].get(k, 0) for k in screen.FIELDS})
    # Requests may be available for a failed/interrupted session before usage.json was written.
    for p in sorted((run / 'sessions').glob('*/*/events.jsonl')):
        if (p.parent / 'usage.json').exists():
            continue
        missing_intervals.append(dict(ref=str(p), reason='active or interrupted session has no finalized usage file'))
        try:
            events = screen.rows(p)
            tid = next((e.get('thread_id') for e in events if e.get('type') == 'thread.started'), None)
            record = screen.usage(tid)
            screen.write_json(p.parent / 'usage.recovered.json', record)
            jobid, cid = p.parent.parent.name, p.parent.parent.name.split('-', 1)[0]
            for req in record.get('requests', []):
                if req['response_id'] not in requests:
                    requests[req['response_id']] = dict(**req, job_id=jobid, case_id=cid)
                    distributions[cid].update({k: req['usage'].get(k, 0) for k in screen.FIELDS})
            if record.get('missing'):
                missing.append(dict(ref=str(p), reason=record['missing']))
        except Exception as exc:
            missing.append(dict(ref=str(p), reason=repr(exc)))
    screen.jsonl(run / 'request_usage.jsonl', requests.values())
    per_task = [dict(case_id=cid, **dict(u), uncached_input_tokens=u['input_tokens']-u['cached_input_tokens'],
                     total_tokens=u['input_tokens']+u['output_tokens']) for cid, u in sorted(distributions.items())]
    screen.jsonl(run / 'task_usage.jsonl', per_task)
    screen.jsonl(run / 'revisions.jsonl', revisions)
    totals = {k: sum(r['usage'].get(k, 0) for r in requests.values()) for k in screen.FIELDS} if requests else None
    if totals:
        totals.update(uncached_input_tokens=totals['input_tokens']-totals['cached_input_tokens'],
                      total_tokens=totals['input_tokens']+totals['output_tokens'])
    master = controller_usage(run, manifest)
    screen.write_json(run / 'master_usage.json', master)
    sweep = []
    for interval in intervals:
        sweep.extend([(datetime.fromisoformat(interval['start'].replace('Z', '+00:00')), 1),
                      (datetime.fromisoformat(interval['end'].replace('Z', '+00:00')), -1)])
    active, observed_peak = 0, 0
    for _, delta in sorted(sweep):
        active += delta
        observed_peak = max(observed_peak, active)
    screen.write_json(run / 'concurrency_audit.json', dict(verified_session_interval_peak=observed_peak,
        method='sweep fresh verifier task_started/task_complete timestamps, ends before starts at ties',
        intervals=intervals, incomplete_intervals=missing_intervals, configured_limit=screen.WORKERS))
    screen.write_json(run / 'usage_detailed.json', dict(updated_at=screen.now(), totals=totals,
        request_count=len(requests), missing=missing, cumulative_cross_checks=session_sums,
        physical_request_count=None, physical_request_count_missing_reason='CLI does not reliably expose every failed pre-response transport request',
        model_transport_retry_count=None, model_transport_retry_count_missing_reason='not reliably exposed by CLI; controller retries are counted separately',
        controller_retry_count=sum(max(0, j['attempts']-1) for j in jobs),
        per_task_total_distribution=quantiles([r['total_tokens'] for r in per_task]),
        subsets='cache is included in input; reasoning is included in output; total=input+output',
        master_usage_ref=str(run / 'master_usage.json'), savings_ratio=None))
    completed = sum(t['status']=='completed' for t in tasks)
    exports = {label: screen.rows(run / (label + '.jsonl')) for label in ['pass', 'exclude', 'review']}
    ids = [r['task_id'] for group in exports.values() for r in group]
    if len(ids) != len(set(ids)) or len(ids) != completed:
        errors.append(dict(issue='final_classification_counts_or_duplicates'))
    scheduled_done = completed==manifest['frozen_count'] and not errors
    gate_path = run/'final_acceptance_gate.json'
    gate = screen.read_json(gate_path) if gate_path.exists() else {}
    master_approved = bool(gate.get('approved') and gate.get('inventory_sha256')==manifest['inventory_sha256']
        and gate.get('prompt_sha256')==manifest['prompt_sha256']
        and gate.get('runner_sha256')==screen.file_hash(screen.__file__)
        and gate.get('correction_scopes_resolved') is True and gate.get('no_outstanding_disputes') is True
        and (run/'pass_sample.json').exists()
        and gate.get('pass_sample_sha256')==screen.file_hash(run/'pass_sample.json'))
    audit_result = dict(at=screen.now(), frozen_count=manifest['frozen_count'], completed=completed,
        unfinished=len(tasks)-completed, structural_errors=errors, source_and_evidence_checks_ok=not errors,
        all_scheduled_case_checks_complete=scheduled_done, master_acceptance_approved=master_approved,
        all_required_work_complete=scheduled_done and master_approved,
        master_acceptance_gate_ref=str(gate_path),
        limitations='Reference/identity/hash/check-completion validation does not replace semantic review.',
        auditor_sha256=screen.file_hash(__file__))
    screen.write_json(run / 'acceptance.json', audit_result)
    print(json.dumps(audit_result, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=screen.DEFAULT)
    audit(p.parse_args().run.resolve())
