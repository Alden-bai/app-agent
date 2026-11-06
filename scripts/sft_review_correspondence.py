"""Extra read-only event/model/call correspondence acceptance for the full audit."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sft_review_full as f


def check(episode, pack):
    events = list(map(json.loads, (episode / 'events.jsonl').read_text().splitlines()))
    turns = {t['turn_id']: t for t in pack['turns']}
    requests, responses, started, settled = {}, {}, {}, set()
    errors = []
    counts = {'model_requests': 0, 'model_responses': 0, 'tool_pairs': 0, 'raw_visible_matches': 0}

    def load(ref):
        path = (episode / ref['path']).resolve()
        if not path.is_relative_to(episode.resolve()): raise ValueError('reference escapes episode')
        if f.p.sha(path) != ref['sha256']: raise ValueError('event reference hash mismatch')
        return f.p.read(path)

    completed = {}
    for event in events:
        kind, tid = event['event_type'], event.get('turn_id')
        try:
            if kind == 'model_request':
                if tid in requests: raise ValueError('duplicate model request')
                requests[tid] = event
                messages = [pack['messages'][i] for i in turns[tid]['input_refs']]
                if messages != load(event['input_ref']): raise ValueError('model input differs from event')
                counts['model_requests'] += 1
            elif kind == 'model_response':
                if tid in responses: raise ValueError('duplicate model response')
                if event['parent_event_id'] != requests[tid]['event_id']: raise ValueError('model response parent mismatch')
                responses[tid] = event
                output = load(event['raw_model_output_ref']); turn = turns[tid]
                expected = dict(turn['output_metadata'], content=turn['output'], finish_reason=turn['finish_reason'])
                if output != expected: raise ValueError('model output differs from event')
                counts['model_responses'] += 1
            elif kind == 'tool_started':
                action = json.loads(turns[tid]['output'])
                if action['type'] != 'call_tool' or action['name'] != event['tool_name']:
                    raise ValueError('tool call name/turn mismatch')
                if action.get('arguments', {}) != load(event['arguments_ref']): raise ValueError('tool arguments differ')
                started[event['event_id']] = event
            elif kind == 'tool_completed':
                call = started[event['call_event_id']]
                if event['call_event_id'] in settled or call['turn_id'] != tid: raise ValueError('tool return mismatch')
                settled.add(event['call_event_id']); completed[event['event_id']] = event
                load(event['raw_result_ref']); counts['tool_pairs'] += 1
            elif kind == 'observation_delivered' and event['parent_event_id'] in completed:
                raw = load(completed[event['parent_event_id']]['raw_result_ref'])
                visible = load(event['visible_result_ref'])
                # Truncated observations must remain truncated in teacher input.
                # The event/model-input comparison above preserves their exact shape.
                if not visible.get('truncated') and visible.get('source') == 'awm':
                    if visible.get('content') != raw: raise ValueError('tool raw/visible result mismatch')
                    counts['raw_visible_matches'] += 1
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append({'event_seq': event['seq'], 'turn_id': tid, 'error': str(error)})
    if set(requests) != set(turns) or set(responses) != set(turns): errors.append({'error': 'missing model event'})
    if set(started) != settled: errors.append({'error': 'unsettled tool call'})
    return {'errors': errors, **counts}


def main():
    manifest = f.manifest()
    results = []
    for row in manifest['selected']:
        i = row['id']
        if i not in manifest['materials']: continue
        result = check(Path(row['episode']), f.p.read(f.OUT/'materials'/i/'pack.json'))
        results.append(dict(id=i, **result))
    report = {'checked_at': f.p.now(), 'script_sha256': f.p.sha(__file__),
              'passed': not any(r['errors'] for r in results), 'candidates': len(results),
              'totals': {k: sum(r[k] for r in results) for k in ('model_requests', 'model_responses', 'tool_pairs', 'raw_visible_matches')},
              'items': results}
    f.write(f.OUT/'correspondence_verification.json', report)
    print(json.dumps({k:v for k,v in report.items() if k!='items'}, ensure_ascii=False))


if __name__ == '__main__': main()
