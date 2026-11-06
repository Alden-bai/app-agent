"""Read-only-source SFT review preparation, validation and reporting; no training export."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness_awm.io import verify_seal

SOURCE = ROOT / 'runs/bank3000_execution_20261104_v1'
OUT = ROOT / 'cleaning_runs/sft_review_pilot50_v1'
SEED = 20261105
QUOTAS = [('zero', 2), ('recovery', 12), ('long', 10), ('ordinary', 26)]
REPEAT = {'zero': 2, 'recovery': 3, 'long': 2, 'ordinary': 3}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def select(pool):
    rng = random.Random(SEED)
    selected, seen, ids, deficits = [], set(), set(), []
    for group, count in QUOTAS:
        def eligible(r):
            return {'zero': r['calls'] == 0,
                    'recovery': r['calls'] > 0 and r['error_signal'],
                    'long': r['turn_count'] >= 20,
                    'ordinary': r['calls'] > 0 and not r['error_signal'] and r['turn_count'] < 20}[group]
        candidates = sorted((r for r in pool if r['id'] not in ids and eligible(r)), key=lambda r: r['id'])
        rng.shuffle(candidates)
        if len(candidates) < count:
            deficits.append({'stratum': group, 'available': len(candidates), 'required': count})
            extra = sorted((r for r in pool if r['id'] not in ids and not eligible(r)), key=lambda r: r['id'])
            rng.shuffle(extra)
            candidates.extend(extra[:count-len(candidates)])
        for _ in range(count):
            if not candidates:
                raise ValueError('insufficient candidates for frozen pilot')
            i = next((i for i, r in enumerate(candidates) if r['scenario'] not in seen), 0)
            r = candidates.pop(i)
            selected.append(dict(r, stratum=group, stratum_fallback=not eligible(r)))
            ids.add(r['id']); seen.add(r['scenario'])
    return selected, deficits


def snapshot(path):
    con = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
    try:
        names = sorted(r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"))
        data = {}
        for name in names:
            cursor = con.execute('SELECT * FROM "' + name.replace('"', '""') + '"')
            columns = [r[0] for r in cursor.description]
            data[name] = [dict(zip(columns, [v.hex() if isinstance(v, bytes) else v for v in row])) for row in cursor]
        return {'source': str(path), 'sha256': sha(path), 'tables': data}
    finally:
        con.close()


def make_pack(episode):
    messages, by_hash, turns, hashes = {}, {}, [], {}
    for folder in sorted((episode / 'model_turns').iterdir()):
        if not folder.is_dir():
            continue
        inp, out = read(folder/'input.json'), read(folder/'output.json')
        refs = []
        for message in inp['messages']:
            key = digest(message)
            if key not in by_hash:
                mid = f'm{len(messages)+1:04d}'
                by_hash[key] = mid
                messages[mid] = message
            refs.append(by_hash[key])
        turns.append({'turn_id': int(folder.name), 'input_refs': refs, 'output': out['content'],
                      'finish_reason': out['finish_reason'], 'source_input': str(folder/'input.json'),
                      'source_output': str(folder/'output.json')})
        assert [messages[r] for r in refs] == inp['messages']
        for path in [folder/'input.json', folder/'output.json']:
            hashes[str(path)] = sha(path)
    return {'messages': messages, 'turns': turns}, hashes


def render(pack, path):
    lines, seen = [], set()
    for i, turn in enumerate(pack['turns']):
        lines.append(f"\nTURN {turn['turn_id']} INPUT_REFS {','.join(turn['input_refs'])}")
        for ref in turn['input_refs']:
            if ref not in seen:
                msg = pack['messages'][ref]
                lines.append(f"NEW MESSAGE {ref} {msg['role']} {path}#/messages/{ref}\n{msg.get('content', '')}")
                seen.add(ref)
        lines.append(f"OUTPUT {path}#/turns/{i}/output\n{turn['output']}")
    return '\n'.join(lines) + '\n'


def checks(episode, pack, result):
    errors, observations = [], []
    try:
        sealed = verify_seal(episode)
        for name in ['trajectory.json', 'events.jsonl', 'initial.db', 'final.db', 'task.json']:
            if name not in sealed:
                errors.append('unsealed:' + name)
        for t in pack['turns']:
            for k in ['source_input', 'source_output']:
                if str(Path(t[k]).relative_to(episode)) not in sealed:
                    errors.append('unsealed:' + t[k])
    except (OSError, ValueError) as e:
        errors.append(str(e))
    events = [json.loads(s) for s in (episode/'events.jsonl').read_text().splitlines()]
    previous, outstanding = None, set()
    for n, event in enumerate(events, 1):
        if event['seq'] != n or event['parent_event_id'] != previous:
            errors.append(f'event_chain:{n}')
        previous = event['event_id']
        if event['event_type'] == 'tool_started':
            outstanding.add(event['event_id'])
        if event['event_type'] == 'tool_completed':
            if event['call_event_id'] not in outstanding:
                errors.append(f'unmatched_completion:{n}')
            outstanding.discard(event['call_event_id'])
    if outstanding:
        errors.append('unsettled_tool_calls')
    execution = result['execution']
    task = read(episode/'task.json')
    trace = read(episode/'trajectory.json')
    if len(pack['turns']) != execution['turns']:
        errors.append('incomplete_model_turns')
    if [t['turn_id'] for t in pack['turns']] != list(range(1, execution['turns']+1)):
        errors.append('nonconsecutive_model_turns')
    if task['task_instance_id'] != execution['task_instance_id']:
        errors.append('task_instance_mismatch')
    if task['instruction'] != trace['task']:
        errors.append('instruction_mismatch')
    if execution['execution_integrity'] != 'valid' or execution['terminal_state_status'] != 'trusted':
        errors.append('untrusted_execution')
    if execution['termination_reason'] != 'finish' or not trace.get('finished') or trace.get('completion_status') != 'completed':
        errors.append('incomplete_termination')
    for t in pack['turns']:
        if t['finish_reason'] != 'stop':
            observations.append(f"non_stop_output:{t['turn_id']}")
    return {'errors': sorted(set(errors)), 'observations': observations,
            'roundtrip_inputs_verified': True, 'turn_ids': [t['turn_id'] for t in pack['turns']],
            'source_split': task.get('split'), 'lineage_group_id': task.get('lineage_group_id'),
            'scope': 'Mechanical integrity only. No prior semantic judge conclusion.'}


def prepare(prompt_path, baseline=None, rules_path=None):
    if (OUT/'manifest.json').exists():
        raise ValueError('manifest already frozen; use status/verify, do not resample')
    index = SOURCE/'exports/all.jsonl'
    pool = []
    for r in map(json.loads, index.read_text().splitlines()):
        if r['label'] != 'valid_success':
            continue
        d = read(r['result']); e = d['execution']; signals = d.get('signals', {})
        v = d['evaluation']
        pool.append({'id': r['id'], 'scenario': r['scenario'], 'task_id': r['task_id'],
                     'episode': r['episode'], 'result': r['result'], 'calls': e['tool_calls'],
                     'turn_count': e['turns'], 'error_signal': bool(signals.get('protocol_errors') or signals.get('tool_errors') or e.get('execution_metrics', {}).get('corrections')),
                     'result_sha256': sha(r['result']), 'independent_evaluation_valid': v.get('evaluation_valid'),
                     'independent_business_label': v.get('business_label')})
    chosen, deficits = select(pool)
    previous = None
    if baseline:
        baseline = Path(baseline).resolve()
        if baseline == OUT.resolve():
            raise ValueError('new review must use a distinct output directory')
        previous = read(baseline/'manifest.json')
        if sha(baseline/'manifest.json') != read(baseline/'manifest.sha256.json')['sha256']:
            raise ValueError('baseline manifest changed')
        if sha(baseline/'source_hashes.json') != previous['source_hashes_sha256']:
            raise ValueError('baseline source hashes changed')
        if any(not Path(name).is_file() or sha(name)!=expected for name,expected in read(baseline/'source_hashes.json').items()):
            raise ValueError('baseline trajectory evidence changed')
        if pool != read(baseline/'candidates.json') or chosen != previous['selected']:
            raise ValueError('candidate source changed; cannot compare the same frozen pilot')
    OUT.mkdir(parents=True, exist_ok=True)
    prompt = Path(prompt_path).read_text()
    (OUT/'prompt.txt').write_text(prompt)
    rules = read(rules_path) if rules_path else None
    if rules:
        write(OUT/'rules.json', rules)
    write(OUT/'candidates.json', pool)
    sources = {str(index): sha(index)}
    materials = {}
    for r in chosen:
        episode = Path(r['episode']); folder = OUT/'materials'/r['id']
        pack, hashes = make_pack(episode)
        pack['id'] = r['id']; pack['task'] = read(episode/'task.json')['instruction']
        pack['protocol_id'] = read(episode/'trajectory.json')['protocol_id']
        write(folder/'pack.json', pack)
        write(folder/'checks.json', checks(episode, pack, read(r['result'])))
        if rules:
            write(folder/'rules.json', rules)
        write(folder/'private.json', {'notice': 'AUDITOR ONLY; NEVER teacher-visible grounding',
                                    'initial': snapshot(episode/'initial.db'), 'final': snapshot(episode/'final.db')})
        (folder/'review.txt').write_text(render(pack, folder/'pack.json'))
        sources.update(hashes)
        for f in episode.rglob('*'):
            if f.is_file():
                sources[str(f)] = sha(f)
        sources[r['result']] = sha(r['result'])
        ji = episode.parent/'judge_input.json'
        sources[str(ji)] = sha(ji)
        original = read(ji)
        write(folder/'requirements.json', {'task': original['original']['task'],
                                         'requirements': original['normalized_rubric'],
                                         'note': 'Historical requirements; verify against user task, no old judge answers.'})
        materials[r['id']] = {str(p): sha(p) for p in sorted(folder.iterdir()) if p.is_file()}
    repeat = []
    rng = random.Random(SEED+1)
    for group, _ in QUOTAS:
        ids = sorted(r['id'] for r in chosen if r['stratum'] == group)
        rng.shuffle(ids); repeat.extend(ids[:REPEAT[group]])
    if previous and repeat != previous['second_review_ids']:
        raise ValueError('secondary sample changed')
    config = {'model': 'gpt-6.1-terra', 'reasoning_effort': 'high', 'fork_turns': 'none',
              'requested_concurrency': 12, 'actual_concurrency': 3, 'max_attempts': 3,
              'prompt_version': rules['schema_version'] if rules else 'sft-review-v1', 'prompt_sha256': sha(OUT/'prompt.txt')}
    if rules:
        config['rules_sha256'] = sha(OUT/'rules.json')
    manifest = {'created_at': now(), 'seed': SEED, 'source': str(index), 'source_sha256': sha(index),
                'pool_count': len(pool), 'selected': chosen, 'deficits': deficits,
                'second_review_ids': repeat, 'config': config, 'materials': materials,
                'source_hashes_file': str(OUT/'source_hashes.json'), 'human_validation': 'pending'}
    write(OUT/'source_hashes.json', sources)
    manifest['source_hashes_sha256'] = sha(OUT/'source_hashes.json')
    manifest['candidates_sha256'] = sha(OUT/'candidates.json')
    if baseline:
        baseline_hashes = {str(p): sha(p) for p in sorted(baseline.rglob('*')) if p.is_file()}
        write(OUT/'baseline_hashes.json', baseline_hashes)
        manifest['baseline'] = str(baseline)
        manifest['baseline_hashes_sha256'] = sha(OUT/'baseline_hashes.json')
    write(OUT/'manifest.json', manifest)
    write(OUT/'manifest.sha256.json', {'sha256': sha(OUT/'manifest.json')})
    for lane in ['primary', 'secondary', 'adjudication']:
        (OUT/'reviews'/lane).mkdir(parents=True, exist_ok=True)
    print(json.dumps({'selected': [(r['id'], r['stratum'], r['turn_count']) for r in chosen], 'second_review_ids': repeat}, ensure_ascii=False))


def pointer(document, fragment):
    if not fragment.startswith('/'):
        raise ValueError('reference must have nonempty RFC6901 pointer')
    for part in fragment[1:].split('/'):
        part = part.replace('~1', '/').replace('~0', '~')
        document = document[int(part)] if isinstance(document, list) else document[part]
    return document


def validate(record, ident, folder=None):
    folder = folder or OUT/'materials'/ident
    pack = read(folder/'pack.json')
    valid_turns = [t['turn_id'] for t in pack['turns']]
    if record['id'] != ident or record['decision'] not in ['keep', 'reject']:
        raise ValueError('invalid id or decision')
    if sorted(record['coverage_turn_ids']) != valid_turns:
        raise ValueError('incomplete turn coverage')
    if not isinstance(record['reason'], str) or not record['reason'].strip():
        raise ValueError('missing reason')
    excluded = record['exclude_loss_turn_ids']
    rules_path = folder/'rules.json'
    rules = read(rules_path) if rules_path.exists() else None
    v2 = rules and rules.get('schema_version') == 'sft-review-v2'
    if len(excluded) != len(set(excluded)) or any(type(t) is not int or t not in valid_turns for t in excluded):
        raise ValueError('invalid excluded turn')
    def refs(values):
        if not isinstance(values, list) or not values:
            raise ValueError('missing evidence')
        factual = False
        for ref in values:
            name, frag = ref.split('#', 1)
            path = Path(name).resolve()
            if path.parent != folder.resolve() or path.name not in ['pack.json', 'private.json', 'checks.json', 'requirements.json', 'rules.json']:
                raise ValueError('reference outside assigned material')
            pointer(read(path), frag)
            factual |= path.name != 'rules.json'
        if v2 and not factual:
            raise ValueError('rules alone are not trajectory evidence')
    refs(record['evidence_refs'])
    supported = set()
    for issue in record['issues']:
        if not issue['tag'] or not issue['reason']:
            raise ValueError('missing issue explanation')
        if any(type(t) is not int or t not in valid_turns for t in issue['turn_ids']):
            raise ValueError('invalid issue turn')
        refs(issue['evidence_refs'])
        if v2:
            if issue.get('rule_id') not in rules['rule_ids'] or issue.get('certainty') not in ['confirmed','uncertain'] or issue.get('severity') not in ['minor','major']:
                raise ValueError('invalid v2 issue classification')
            if issue['certainty'] == 'confirmed': supported.update(issue['turn_ids'])
        else:
            supported.update(issue['turn_ids'])
    if not set(excluded) <= supported:
        raise ValueError('excluded turn without issue evidence')
    if record['decision'] == 'reject' and not record['issues']:
        raise ValueError('reject without issue')
    if record['decision'] == 'keep' and read(folder/'checks.json')['errors']:
        raise ValueError('keep with unresolved source integrity error')
    if v2:
        major = any(i['severity']=='major' for i in record['issues'])
        if record['decision']=='keep' and major:
            raise ValueError('keep with major unresolved issue')
        if record['decision']=='reject' and not major:
            raise ValueError('reject requires major issue')
        if record['decision']=='keep' and set(excluded)==set(valid_turns):
            raise ValueError('keep without supervised output')
    return record


def verified_manifest():
    if sha(OUT/'manifest.json') != read(OUT/'manifest.sha256.json')['sha256']:
        raise ValueError('manifest changed')
    m = read(OUT/'manifest.json')
    if sha(OUT/'prompt.txt') != m['config']['prompt_sha256']:
        raise ValueError('prompt changed')
    if m['config'].get('rules_sha256') and sha(OUT/'rules.json') != m['config']['rules_sha256']:
        raise ValueError('rules changed')
    for paths in m['materials'].values():
        for name, expected in paths.items():
            if sha(name) != expected:
                raise ValueError('review material changed:' + name)
    return m


def cache_key(m, ident, lane):
    return digest({'materials': m['materials'][ident], 'config': m['config'], 'lane': lane})


def status():
    m = verified_manifest(); output = {}
    for lane in ['primary', 'secondary']:
        ids = [r['id'] for r in m['selected']] if lane == 'primary' else m['second_review_ids']
        done, missing, invalid = [], [], {}
        for ident in ids:
            p = OUT/'reviews'/lane/(ident+'.json')
            if not p.exists():
                missing.append(ident); continue
            try:
                validate(read(p), ident)
                receipt = p.with_suffix('.receipt.json')
                key = cache_key(m, ident, lane)
                if receipt.exists():
                    old = read(receipt)
                    if old['cache_key'] != key or old['result_sha256'] != sha(p):
                        raise ValueError('stale cache receipt')
                else:
                    write(receipt, {'cache_key': key, 'result_sha256': sha(p), 'validated_at': now()})
                done.append(ident)
            except (ValueError, KeyError, TypeError, IndexError, OSError) as e:
                invalid[ident] = str(e)
        output[lane] = {'done': done, 'missing': missing, 'invalid': invalid}
    return output


def verify_sources():
    m = verified_manifest()
    if sha(OUT/'source_hashes.json') != m['source_hashes_sha256'] or sha(OUT/'candidates.json') != m['candidates_sha256']:
        raise ValueError('source index changed')
    sources = read(OUT/'source_hashes.json')
    changed = [name for name, expected in sources.items() if not Path(name).is_file() or sha(name) != expected]
    result = {'checked_at': now(), 'files_checked': len(sources), 'changed': changed}
    if m.get('baseline'):
        if sha(OUT/'baseline_hashes.json') != m['baseline_hashes_sha256']:
            raise ValueError('baseline snapshot changed')
        before = read(OUT/'baseline_hashes.json')
        result['baseline_files_checked'] = len(before)
        result['baseline_changed'] = [name for name, expected in before.items() if not Path(name).is_file() or sha(name) != expected]
    write(OUT/'source_verification.json', result)
    return result


def start_job(ident, lane, agent):
    m = verified_manifest()
    expected = [r['id'] for r in m['selected']] if lane == 'primary' else m['second_review_ids']
    if ident not in expected:
        raise ValueError('unplanned job')
    current = status()[lane]
    if ident in current['done']:
        raise ValueError('already completed; do not resubmit')
    path = OUT/'jobs'/lane/(ident+'.json')
    history = read(path) if path.exists() else []
    if len(history) >= m['config']['max_attempts']:
        raise ValueError('attempt limit reached')
    history.append({'agent': agent, 'registered_at': now(), 'cache_key': cache_key(m, ident, lane),
                    'attempt': len(history)+1})
    write(path, history)


def finalize():
    m = verified_manifest(); state = status()
    final, disagreements, incomplete = [], [], []
    for row in m['selected']:
        ident = row['id']
        if ident not in state['primary']['done']:
            incomplete.append({'id': ident, 'phase': 'primary', 'error': state['primary']['invalid'].get(ident, 'missing')})
            continue
        first = read(OUT/'reviews/primary'/f'{ident}.json')
        result = first
        if ident in m['second_review_ids']:
            if ident not in state['secondary']['done']:
                incomplete.append({'id': ident, 'phase': 'secondary', 'error': state['secondary']['invalid'].get(ident, 'missing')})
                continue
            second = read(OUT/'reviews/secondary'/f'{ident}.json')
            if (first['decision'], sorted(first['exclude_loss_turn_ids'])) != (second['decision'], sorted(second['exclude_loss_turn_ids'])):
                conflict = {'id': ident, 'primary': first['decision'], 'secondary': second['decision'],
                            'primary_exclude': first['exclude_loss_turn_ids'], 'secondary_exclude': second['exclude_loss_turn_ids']}
                disagreements.append(conflict)
                path = OUT/'reviews/adjudication'/f'{ident}.json'
                if not path.exists():
                    incomplete.append({'id': ident, 'phase': 'adjudication', 'error': 'unresolved_disagreement'})
                    continue
                result = validate(read(path), ident)
                if not result.get('adjudication_reason'):
                    raise ValueError('adjudication must explain evidence resolution')
        # A documented evidence audit may correct a primary result even when
        # this case is not in the fixed blind-secondary sample.
        path = OUT/'reviews/adjudication'/f'{ident}.json'
        if path.exists():
            result = validate(read(path), ident)
            if not result.get('adjudication_reason'):
                raise ValueError('adjudication must explain evidence resolution')
        final.append(dict(result, stratum=row['stratum'], task_id=row['task_id'], scenario=row['scenario']))
    write(OUT/'processing_status.json', state)
    write(OUT/'disagreements.json', disagreements)
    write(OUT/'incomplete.json', incomplete)
    (OUT/'reviews.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in final))
    for label in ['keep', 'reject']:
        write(OUT/(label+'.json'), [r['id'] for r in final if r['decision']==label])
    write(OUT/'supervision_annotations.json', {r['id']: r['exclude_loss_turn_ids'] for r in final})
    human_ids = set(m['second_review_ids']) | {r['id'] for r in final if r['exclude_loss_turn_ids'] or r['decision']=='reject'} | {r['id'] for r in disagreements}
    if m.get('baseline'):
        previous = {r['id']: r for r in map(json.loads, (Path(m['baseline'])/'reviews.jsonl').read_text().splitlines())}
        human_ids.update(r['id'] for r in final
                         if (r['decision'], sorted(r['exclude_loss_turn_ids'])) !=
                         (previous[r['id']]['decision'], sorted(previous[r['id']]['exclude_loss_turn_ids'])))
    for label in ['keep', 'reject']:
        group = sorted((r for r in final if r['decision']==label), key=lambda r:r['id'])
        already = sum(r['id'] in human_ids for r in group)
        for r in group:
            if already >= min(5,len(group)): break
            if r['id'] not in human_ids:
                human_ids.add(r['id']); already += 1
    human = []
    by_id = {r['id']: r for r in final}
    lines = ['# 人工抽查材料', '', '人工标签尚未填写；模型复审不等于人工验收。', '']
    for ident in sorted(human_ids):
        if ident not in by_id: continue
        r = by_id[ident]; pack = read(OUT/'materials'/ident/'pack.json')
        issue_turns = set(r['exclude_loss_turn_ids']) | {t for issue in r['issues'] for t in issue['turn_ids']}
        snippets = [dict(turn_id=t['turn_id'], output=t['output'], input_refs=t['input_refs']) for t in pack['turns'] if t['turn_id'] in issue_turns]
        human.append({'id': ident, 'task': pack['task'], 'model_decision': r['decision'], 'model_reason': r['reason'],
                      'exclude_loss_turn_ids': r['exclude_loss_turn_ids'], 'evidence_refs': r['evidence_refs'],
                      'adjudication_reason': r.get('adjudication_reason'),
                      'related_turns': snippets, 'human_decision': '', 'human_exclude_loss_turn_ids': None, 'human_notes': ''})
        material = OUT/'materials'/ident
        final_record = OUT/'reviews/adjudication'/f'{ident}.json'
        if not final_record.exists(): final_record = OUT/'reviews/primary'/f'{ident}.json'
        lines += [f"## {ident} · {r['decision']} · {r['stratum']}", '', pack['task'], '',
                  f"结论：{r['reason']}", f"不监督轮次：{r['exclude_loss_turn_ids']}",
                  f"[完整轨迹]({material/'review.txt'}) · [初末态]({material/'private.json'}) · [最终审核记录]({final_record})", '']
        if r.get('adjudication_reason'):
            lines += ['裁决说明：'+r['adjudication_reason'], '']
        for issue in r['issues']:
            lines.append(f"- {issue['tag']}，轮次 {issue['turn_ids']}：{issue['reason']}")
        for t in snippets:
            lines += ['', f"轮次 {t['turn_id']} 教师输出：", '```text', t['output'], '```']
        lines += ['', '证据定位：'] + ['- `'+ref+'`' for ref in r['evidence_refs']]
        lines += ['', '人工结论：______　人工不监督轮次：______　备注：______', '']
    write(OUT/'human_review.json', human)
    (OUT/'human_review.md').write_text('\n'.join(lines)+'\n')
    verification = verify_sources()
    counts = Counter(r['decision'] for r in final)
    strata = {group: dict(Counter(r['decision'] for r in final if r['stratum']==group)) for group,_ in QUOTAS}
    issue_counts = Counter(i['tag'] for r in final for i in r['issues'])
    summary = {'created_at': m['created_at'], 'finished_at': now(), 'primary_completed': len(state['primary']['done']),
               'secondary_completed': len(state['secondary']['done']), 'finalized': len(final), 'counts': dict(counts),
               'strata': strata, 'issues': dict(issue_counts), 'excluded_turns': sum(len(r['exclude_loss_turn_ids']) for r in final),
               'kept_excluded_turns': sum(len(r['exclude_loss_turn_ids']) for r in final if r['decision']=='keep'),
               'kept_episodes_with_exclusions': sum(bool(r['exclude_loss_turn_ids']) for r in final if r['decision']=='keep'),
               'all_selected_turns': sum(r['turn_count'] for r in m['selected']), 'disagreements': len(disagreements),
               'decision_disagreements': sum(r['primary'] != r['secondary'] for r in disagreements),
               'mask_disagreements': sum(sorted(r['primary_exclude']) != sorted(r['secondary_exclude']) for r in disagreements),
               'spot_check_adjudications': sum(r.get('adjudication_source')=='spot_check' for r in final),
               'incomplete': len(incomplete), 'human_review_count': len(human), 'human_validation': 'pending',
               'tokens': None, 'cost_usd': None, 'usage_note': 'Subagent usage unavailable; no fabricated token/cost estimate.',
               'source_verification': verification, 'config': m['config']}
    summary['wall_seconds'] = (datetime.fromisoformat(summary['finished_at'])-datetime.fromisoformat(m['created_at'])).total_seconds()
    summary['kept_turns'] = sum(len(r['coverage_turn_ids']) for r in final if r['decision']=='keep')
    summary['candidate_supervised_turns'] = summary['kept_turns'] - summary['kept_excluded_turns']
    summary['issue_rules'] = dict(Counter(i.get('rule_id','v1_unspecified') for r in final for i in r['issues']))
    summary['issue_certainty_severity'] = dict(Counter(
        i.get('certainty','v1_unspecified')+'/'+i.get('severity','v1_unspecified') for r in final for i in r['issues']))
    write(OUT/'summary.json', summary)
    report = ['# 50 条 SFT 轨迹审核试跑报告', '',
              f"首审完成 {summary['primary_completed']}/50；独立复审 {summary['secondary_completed']}/10；最终记录 {len(final)} 条。",
              f"keep {counts.get('keep',0)}；reject {counts.get('reject',0)}；处理未完成 {len(incomplete)}。", '',
              '| 分层 | keep | reject |', '|---|---:|---:|']
    report += [f"| {group} | {strata[group].get('keep',0)} | {strata[group].get('reject',0)} |" for group,_ in QUOTAS]
    report += ['', f"入选轨迹共 {summary['all_selected_turns']} 轮；标记不监督 {summary['excluded_turns']} 轮（包括 reject 的诊断标记，不代表训练导出）。",
               f"其中 keep 轨迹有 {summary['kept_episodes_with_exclusions']} 条带不监督标记，共 {summary['kept_excluded_turns']} 轮。",
               f"keep 轨迹共 {summary['kept_turns']} 轮，扣除标记后有 {summary['candidate_supervised_turns']} 个候选监督目标；尚未导出或人工验收。",
               f"独立复审决策/监督标记分歧 {len(disagreements)} 条。人工抽查材料 {len(human)} 条，人工结论尚未填写。",
               f"其中整条决策分歧 {summary['decision_disagreements']} 条，不监督轮次分歧 {summary['mask_disagreements']} 条；一致不等于正确。",
               f"另有主 agent 证据回查裁决 {summary['spot_check_adjudications']} 条，不计入盲复审一致率。",
               f"从材料冻结到报告生成约 {summary['wall_seconds']/60:.1f} 分钟；模型 {m['config']['model']} / high，实际并发 3。",
               '子 agent 的准确 token 和费用不可获取，未进行估造。', '', '## 问题类型', '']
    report += [f'- {tag}: {count}' for tag,count in issue_counts.most_common()]
    report += ['', '## 分歧裁决', '']
    for conflict in disagreements:
        result = by_id.get(conflict['id'])
        if result and result.get('adjudication_reason'):
            report += [f"- **{conflict['id']}**：{result['adjudication_reason']}",
                       f"  [裁决记录](reviews/adjudication/{conflict['id']}.json)"]
    for result in final:
        if result.get('adjudication_source') == 'spot_check':
            report += [f"- **{result['id']}（主 agent 回查）**：{result['adjudication_reason']}",
                       f"  [裁决记录](reviews/adjudication/{result['id']}.json)"]
    report += ['', '## 完整性与范围', '', f"核对 {verification['files_checked']} 个原始文件，发生变化 {len(verification['changed'])} 个。",
               '未修改轨迹或奖励字段，未导出 SFT，未启动训练，未扩大到其余候选。',
               '开发集来源标签保留；审核 keep 不等于已完成训练/评测划分或正式训练准入。', '',
               '## 全量前待解决', '', '- 人工核对抽查表，尤其是错误恢复、零业务调用、分歧和不监督轮次。',
               '- 若发现误杀探索、遗漏副作用或错误屏蔽，冻结新提示词版本并重审受影响样本。',
               '- 分层样本保留率不代表候选总体；模型一致率不是准确率。',
               '- 后续若需要训练，另行实现监督标记导出及任务分组防泄漏；本轮不实施。', '',
               '[人工抽查表](human_review.md) · [逐条记录](reviews.jsonl) · [分歧](disagreements.json) · [未完成项](incomplete.json)']
    (OUT/'report.md').write_text('\n'.join(report)+'\n')
    if m.get('baseline'):
        compare_versions(m, final)
    print(json.dumps(summary, ensure_ascii=False))


def compare_versions(manifest, final):
    before = {r['id']: r for r in map(json.loads, (Path(manifest['baseline'])/'reviews.jsonl').read_text().splitlines())}
    changes = []
    for r in final:
        old = before[r['id']]
        if (old['decision'], sorted(old['exclude_loss_turn_ids'])) != (r['decision'], sorted(r['exclude_loss_turn_ids'])):
            changes.append({'id':r['id'], 'before_decision':old['decision'],'after_decision':r['decision'],
                            'before_exclude':old['exclude_loss_turn_ids'],'after_exclude':r['exclude_loss_turn_ids'],
                            'before_reason':old['reason'],'after_reason':r['reason']})
    result = {'baseline':manifest['baseline'], 'same_50_sample':True,'compared':len(final),
              'decision_changes':sum(r['before_decision']!=r['after_decision'] for r in changes),
              'mask_changes':sum(sorted(r['before_exclude'])!=sorted(r['after_exclude']) for r in changes),
              'changes':changes,'interpretation':'Rule calibration comparison, not measured accuracy improvement.'}
    write(OUT/'comparison.json',result)
    lines = ['# v1 / v2 同批样本比较','',f"对比 {len(final)} 条，决策变化 {result['decision_changes']} 条，监督标记变化 {result['mask_changes']} 条。",'',
             '变化不等于准确率提高；原始 v1 结果保留，人工金标仍待确认。','',
             '| ID | v1 → v2 | 不监督轮次 v1 → v2 |','|---|---|---|']
    lines += [f"| {r['id']} | {r['before_decision']} → {r['after_decision']} | {r['before_exclude']} → {r['after_exclude']} |" for r in changes]
    for r in changes:
        lines += ['',f"## {r['id']}",'','v1：'+r['before_reason'],'','v2：'+r['after_reason']]
    (OUT/'comparison.md').write_text('\n'.join(lines)+'\n')


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'status', 'verify', 'validate', 'finalize', 'start'])
    parser.add_argument('--prompt'); parser.add_argument('--record'); parser.add_argument('--id')
    parser.add_argument('--lane', choices=['primary', 'secondary']); parser.add_argument('--agent')
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--baseline', type=Path); parser.add_argument('--rules', type=Path)
    args = parser.parse_args()
    OUT = args.output.resolve()
    if args.command == 'prepare': prepare(args.prompt, args.baseline, args.rules)
    elif args.command == 'status': print(json.dumps(status(), ensure_ascii=False))
    elif args.command == 'verify': print(json.dumps(verify_sources(), ensure_ascii=False))
    elif args.command == 'validate': validate(read(args.record), args.id); print('valid')
    elif args.command == 'start': start_job(args.id, args.lane, args.agent)
    elif args.command == 'finalize': finalize()


if __name__ == '__main__':
    main()
