#!/usr/bin/env python3
"""Actual MCP read/write smoke tests on disposable repaired snapshot copies; no models."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness_awm.contracts import RunConfig
from harness_awm.environment import AWMEnvAdapter
from harness_awm.io import db_fingerprint, digest, file_hash, write_json


def cases(release):
    with sqlite3.connect(release / 'databases/local_search_1.db') as db:
        slot = db.execute('SELECT slot_time FROM reservation_slots WHERE id=1').fetchone()[0]
    return [
        dict(category='foreign_key', scenario='communication_messaging_1',
             read=('list_emails_in_thread', dict(thread_id=11)), contains='Out of Office',
             write=('update_email_state', dict(email_id=16, is_starred=True)),
             sql='SELECT is_starred FROM email_states WHERE email_id=16 AND user_id=1', expected=[[1]]),
        dict(category='sql_syntax', scenario='q_a_forum_1',
             read=('get_post_by_id', dict(post_id=19374)), contains='19374',
             write=('edit_post_body', dict(post_id=19374, body='Repair smoke: logging helper remains available.')),
             sql='SELECT body FROM posts WHERE id=19374', expected=[['Repair smoke: logging helper remains available.']]),
        dict(category='column_value_arity', scenario='team_messaging_5',
             read=('get_channel_export_by_id', dict(export_id=2)), contains='company-all-hands-h1-2024',
             write=('create_channel_export', dict(channel_id=4, date_from='2024-01-01', date_to='2024-06-30', archive_label='repair-smoke-export')),
             sql="SELECT channel_id, status FROM channel_exports WHERE archive_label='repair-smoke-export'", expected=[[4, 'pending']]),
        dict(category='unknown_column', scenario='booking_reservations_5',
             read=('list_room_types_for_property', dict(property_id=2)), contains='Studio City View',
             write=('create_room', dict(property_id=2, room_type_id=4, room_number='REPAIR-SMOKE', status='vacant')),
             sql="SELECT property_id,room_type_id FROM rooms WHERE room_number='REPAIR-SMOKE'", expected=[[2,4]]),
        dict(category='not_null_date_expression', scenario='local_search_1',
             read=('list_reservation_slots', dict(business_id=16, date_str=slot[:10])), contains=slot[:10],
             write=('create_reservation', dict(business_id=16, reservation_time=slot.replace(' ', 'T'), party_size=1, notes='repair-smoke-booking')),
             sql="SELECT business_id,party_size FROM reservations WHERE notes='repair-smoke-booking'", expected=[[16,1]]),
        dict(category='unique_key', scenario='forum_1',
             read=('list_my_recent_topic_views', dict(limit=20)), contains='"topic_view_id":4',
             write=('upsert_topic_view', dict(topic_id=3, last_viewed_post_id=7)),
             sql='SELECT id,last_viewed_post_id FROM topic_views WHERE user_id=1 AND topic_id=3', expected=[[4,7]]),
        dict(category='check_constraint', scenario='crm_lead_management_2',
             read=('search_inventory_vehicles', dict(status='removed')), contains='2G1WF52E859344567',
             write=('patch_inventory_vehicle_by_id', dict(vehicle_id=10, status_note='repair-smoke-checked')),
             sql='SELECT status,status_note FROM inventory_vehicles WHERE id=10', expected=[['removed','repair-smoke-checked']]),
    ]


def arguments_for(tool, arguments):
    properties = tool['inputSchema'].get('properties', {})
    # FastAPI MCP versions may expose a body wrapper or flattened model fields.
    for key in ('payload', 'body', 'body_data'):
        if key in properties and properties[key].get('type') == 'object':
            path = {k: v for k, v in arguments.items() if k in properties and k != key}
            path[key] = {k: v for k, v in arguments.items() if k not in path}
            return path
    return arguments


async def run(release, output):
    wanted = cases(release)
    scenarios = {c['scenario'] for c in wanted}
    envs = {e['scenario']: e for e in map(json.loads, (release / 'gen_envs.jsonl').open()) if e['scenario'] in scenarios}
    results = []
    for case in wanted:
        scenario = case['scenario']
        folder = output / scenario
        folder.mkdir(parents=True)
        env_path = folder / 'environment.json'
        write_json(env_path, envs[scenario])
        initial = release / 'databases' / (scenario + '.db')
        before = file_hash(initial)
        task = SimpleNamespace(scenario=scenario, initial_state_ref=str(initial), initial_state_hash=before,
            initial_state_fingerprint=db_fingerprint(initial), environment_ref=str(env_path),
            environment_code_hash=digest(envs[scenario]), tool_schema_hash=None, readiness_profile={'runtime_ready':True})
        adapter = AWMEnvAdapter(task, RunConfig(startup_timeout=40, tool_timeout=20), folder)
        result = dict(scenario=scenario, category=case['category'], passed=False, calls=[])
        try:
            await adapter.prepare()
            for kind in ('read', 'write', 'read'):
                name, params = case[kind]
                tool = next(t for t in adapter.tools if t['name'] == name)
                args = arguments_for(tool, params)
                response = await adapter.call(name, args)
                result['calls'].append(dict(kind=kind, name=name, arguments=args, response=response))
                assert not response.get('isError'), response
                if kind == 'read':
                    content = ' '.join(c.get('text', '') for c in response.get('content', []))
                    assert case['contains'].replace(' ', '') in content.replace(' ', ''), content
            with sqlite3.connect(adapter.runtime / 'working.db') as db:
                rows = [list(r) for r in db.execute(case['sql'])]
                result.update(query=case['sql'], rows=rows, expected=case['expected'],
                    integrity_check=db.execute('PRAGMA integrity_check').fetchall(),
                    foreign_key_check=db.execute('PRAGMA foreign_key_check').fetchall())
                assert rows == case['expected'], rows
                assert result['integrity_check'] == [('ok',)] and not result['foreign_key_check']
            result['passed'] = True
        except Exception as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
        finally:
            result['process_stopped'] = await adapter.close()
            if adapter.started:
                adapter.freeze()
            result['initial_snapshot_unchanged'] = file_hash(initial) == before
            result['passed'] = result['passed'] and result['initial_snapshot_unchanged'] and result['process_stopped']
        write_json(folder / 'result.json', result)
        results.append(result)
        print(scenario, result['passed'], result.get('error', ''), flush=True)
    summary = dict(started_from_release=str(release), finished_at=datetime.now(timezone.utc).isoformat(),
        passed=sum(r['passed'] for r in results), total=len(results), results=results,
        scope='Seven category smoke tests, not task success evaluation; no paid model requests')
    write_json(output / 'summary.json', summary)
    return int(any(not r['passed'] for r in results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=ROOT / 'data/awm-dde80a0-repaired-v1')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.release / 'mcp_smoke' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    return asyncio.run(run(args.release.resolve(), output.resolve()))


if __name__ == '__main__':
    sys.exit(main())
