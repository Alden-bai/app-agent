"""Frozen, resumable full SFT audit. Never executes business tools or exports SFT.

Dispatch is deliberately external: Codex collaboration creates one fresh reviewer
per job. This program prepares, claims, validates, accepts and reconciles jobs.
All mutable state belongs to OUT; the pilot and original evidence are read only.
"""
import argparse
from collections import Counter
from datetime import datetime
import fcntl
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sft_review_pilot as p

ROOT = p.ROOT
OUT = ROOT / 'cleaning_runs/sft_review_full966_v1'
BASE = ROOT / 'cleaning_runs/sft_review_pilot50_v2'
SOURCE = p.SOURCE / 'exports/all.jsonl'
REVISION = 'sft-review-v2.1'
LANES = ('primary', 'targeted', 'secondary', 'adjudication')
CLARIFICATIONS = """

本轮冻结规则修订：sft-review-v2.1（记录schema仍为sft-review-v2，全部严格校验继续有效）。
工具schema只说明执行语义，参数合法/工具成功不证明副作用符合任务。必须独立核对实际覆盖、数据回退、错对象、额外或重复配置等影响；已证实不可接受的重大副作用应reject，不能靠mask消除。工具故障不直接归咎教师，但必要任务未完成仍不能keep。影响可接受性无法确认时明确缺失依据，不把不确定写为已证明违规。
检索全部用于选择/执行不自动要求最终列出未选对象；明确要求完整报告则完整交付。合理探索、空查询、核验及正确恢复均可监督。
只有confirmed问题支持对应实际错误轮次mask；关键依据缺失可uncertain/major并reject，但不能自动mask。局部非关键无依据币种或单位可minor并mask该输出；正确工具选择、对象解析、参数化调用有独立价值，不因最终答复mask自动拒绝整条。忠实引用工具异常数值、诚实披露错误/未完成不自动算错误输出。
动作依据只用该轮input_refs所限定实际可见信息或可靠推导；终态/后台证据仅供审核，后来的成功不补先前依据。区分相似字段/对象，如candidate.source和application.source。
SFT审核不要求reward；episode_result.json缺少reward不构成拒绝理由。独立Judge与执行记录不同，旧成功标签不等于keep。
审核员只能读取分配材料及对应episode原始证据；禁止读取任何既有审核、Judge结论、result.json、evaluations、候选标签或全量manifest。禁止业务工具、API、重放、修改源数据、创建子agent。必须读完全部轮次，截断分段补齐；只写分配的唯一紧凑JSON文件。
以本轮调度消息指定的新入口执行 validate；旧提示词末尾的v2命令示例由本轮调度消息替代。reject_category须为 confirmed_quality_issue/material_corruption/insufficient_evidence；keep无需该字段。
"""


def write(path, value):
    """Atomic replacement within the current run only."""
    path = Path(path)
    if not path.resolve().is_relative_to(OUT.resolve()):
        raise ValueError('write outside current audit output')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def append_event(value):
    with (OUT / 'scheduler.jsonl').open('a') as f:
        f.write(json.dumps(dict(at=p.now(), **value), ensure_ascii=False) + '\n')


def file_hashes(folder):
    return {str(f.resolve()): p.sha(f) for f in sorted(folder.rglob('*')) if f.is_file()}


def changed(hashes):
    return [name for name, h in hashes.items() if not Path(name).is_file() or p.sha(name) != h]


def manifest(ident=None, full=False):
    """Only one assigned item is hashed during dispatch/acceptance."""
    if p.sha(OUT / 'manifest.json') != p.read(OUT / 'manifest.sha256.json')['sha256']:
        raise ValueError('manifest changed')
    m = p.read(OUT / 'manifest.json')
    if changed(m['frozen_files']):
        raise ValueError('frozen configuration changed')
    ids = m['materials'] if full else ([ident] if ident else [])
    for i in ids:
        if changed(m['materials'].get(i, {})):
            raise ValueError('material changed: ' + i)
    return m


def impact_scan(rows, old):
    """Conservative semantic triage: every actual mutation gets fresh review."""
    result = []
    for row in sorted(rows, key=lambda x: x['id']):
        i = row['id']; folder = BASE / 'materials' / i
        private = p.read(folder / 'private.json')
        before, after = private['initial']['tables'], private['final']['tables']
        tables = sorted(t for t in set(before) | set(after) if before.get(t) != after.get(t))
        rereview = bool(tables) or i == 't1848'
        record = old[i]
        # Read/validate the *final* record, including any original adjudication.
        p.validate(record, i, folder)
        result.append({'id': i, 'action': 'targeted' if rereview else 'reuse',
                       'reason': ('实际状态变化：逐项新审覆盖/回退/重复配置/关联与任务边界；不预判违规。'
                                  if rereview else
                                  '初末态逐表相等；旧最终记录对探索恢复、报告范围、confirmed mask及监督价值的处理与v2.1澄清兼容。'),
                       'changed_tables': tables,
                       'screening_evidence': [str(folder / 'private.json') + '#/initial/tables',
                                              str(folder / 'private.json') + '#/final/tables'],
                       'old_final_line': list(old).index(i) + 1,
                       'old_final_record_sha256': p.digest(record),
                       'old_rule_version': 'sft-review-v2'})
    return result


def prepare():
    if (OUT / 'manifest.json').exists():
        raise ValueError('already frozen; use next/status/verify/finalize')
    OUT.mkdir(parents=True, exist_ok=True)
    started = OUT / 'preparation_started_at.json'
    if not started.exists(): write(started, {'at': p.now()})
    # Freeze all pilot deliverables before inspecting them. Never call pilot status.
    bh = file_hashes(BASE)
    if (OUT / 'baseline_hashes.json').exists():
        if p.read(OUT / 'baseline_hashes.json') != bh:
            raise ValueError('pilot changed since interrupted preparation')
    else:
        write(OUT / 'baseline_hashes.json', bh)
    bm = p.read(BASE / 'manifest.json')
    if p.sha(BASE / 'manifest.json') != p.read(BASE / 'manifest.sha256.json')['sha256']:
        raise ValueError('pilot manifest invalid')
    if p.sha(BASE / 'source_hashes.json') != bm['source_hashes_sha256']:
        raise ValueError('pilot source hash list invalid')
    baseline_source_changes = changed(p.read(BASE / 'source_hashes.json'))
    pool = sorted(p.read(BASE / 'candidates.json'), key=lambda x: x['id'])
    ids = [r['id'] for r in pool]
    if len(ids) != 966 or len(set(ids)) != len(ids):
        raise ValueError('expected exactly 966 unique frozen pilot-pool IDs')
    current_rows = list(map(json.loads, SOURCE.read_text().splitlines())) if SOURCE.exists() else []
    current = {r['id']: r for r in current_rows if r['label'] == 'valid_success'}
    duplicates = [i for i, n in Counter(r['id'] for r in current_rows).items() if n > 1]
    diffs = {'source_missing': not SOURCE.exists(), 'source_hash_changed': not SOURCE.exists() or p.sha(SOURCE) != bm['source_sha256'],
             'missing_from_current_valid_success': sorted(set(ids) - set(current)),
             'extra_current_valid_success_excluded': sorted(set(current) - set(ids)),
             'duplicate_current_ids': duplicates, 'baseline_evidence_changed': baseline_source_changes,
             'candidate_changes': []}
    blocked = {}
    old = {r['id']: r for r in map(json.loads, (BASE / 'reviews.jsonl').read_text().splitlines())}
    if set(old) != {r['id'] for r in bm['selected']} or len(old) != 50:
        raise ValueError('pilot final records incomplete/duplicate')
    impact = impact_scan(bm['selected'], old)
    write(OUT / 'rule_impact.json', impact)
    reuse = sorted(r['id'] for r in impact if r['action'] == 'reuse')
    targeted = sorted(r['id'] for r in impact if r['action'] == 'targeted')
    fresh = sorted(set(ids) - set(old))
    repeat = sorted(random.Random(p.SEED).sample(fresh, math.ceil(len(fresh) * .05)))
    # List frozen before the first primary review, regardless of unavailable items.
    write(OUT / 'blind_secondary_ids.json', repeat)
    rules = p.read(BASE / 'rules.json')
    rules.update(rule_revision=REVISION, scope='Full frozen 966 valid_success candidates; audit only.')
    write(OUT / 'rules.json', rules)
    (OUT / 'prompt.txt').write_text((BASE / 'prompt.txt').read_text() + CLARIFICATIONS)
    (OUT / 'rule_changes.md').write_text('# sft-review-v2.1\n\n' + CLARIFICATIONS + '\n')
    config = {'model': 'gpt-6.1-terra', 'reasoning_effort': 'high', 'fork_turns': 'none',
              'requested_concurrency': 12, 'actual_concurrency': 3, 'max_attempts': 3,
              'schema_version': 'sft-review-v2', 'rule_revision': REVISION,
              'seed': p.SEED, 'scope': 'audit only; no SFT export/training/business replay'}
    write(OUT / 'model_config.json', config)
    write(OUT / 'candidates.json', pool)
    material_hashes, sources = {}, {str(SOURCE): p.sha(SOURCE)} if SOURCE.exists() else {}
    for row in pool:
        i = row['id']; episode = Path(row['episode']); folder = OUT / 'materials' / i
        try:
            c = current.get(i)
            why = []
            if not c: why.append('missing valid_success source row')
            elif any(c.get(k) != row.get(k) for k in ('scenario', 'task_id', 'episode', 'result')):
                why.append('candidate source fields changed')
            if i in duplicates: why.append('duplicate source id')
            if not Path(row['result']).exists() or p.sha(row['result']) != row['result_sha256']:
                why.append('frozen result missing/changed')
            impacted = [s for s in baseline_source_changes if str(episode.parent) + '/' in s]
            if impacted: why.append('pilot frozen episode evidence changed')
            if why:
                diffs['candidate_changes'].append({'id': i, 'reasons': why, 'changed_evidence': impacted})
                blocked[i] = '; '.join(why)
                continue
            pack, _ = p.make_pack(episode)
            pack.update(id=i, task=p.read(episode / 'task.json')['instruction'],
                        protocol_id=p.read(episode / 'trajectory.json')['protocol_id'])
            for turn in pack['turns']:
                inp, output = p.read(turn['source_input']), p.read(turn['source_output'])
                turn['input_metadata'] = {k: v for k, v in inp.items() if k != 'messages'}
                turn['output_metadata'] = {k: v for k, v in output.items() if k not in ('content', 'finish_reason')}
            write(folder / 'pack.json', pack)
            write(folder / 'checks.json', p.checks(episode, pack, p.read(row['result'])))
            write(folder / 'rules.json', rules)
            write(folder / 'private.json', {'notice': '审核专用 AUDITOR ONLY. Never prior teacher grounding.',
                                           'initial': p.snapshot(episode / 'initial.db'),
                                           'final': p.snapshot(episode / 'final.db')})
            (folder / 'review.txt').write_text(p.render(pack, folder / 'pack.json'))
            ji = episode.parent / 'judge_input.json'
            original = p.read(ji)
            write(folder / 'requirements.json', {'task': original['original']['task'],
                    'requirements': original['normalized_rubric'],
                    'note': 'Reference requirements only; user instruction takes precedence. No judge answers.'})
            material_hashes[i] = file_hashes(folder)
            # Includes independent evaluation artifacts, kept out of reviewer materials.
            sources.update(file_hashes(episode.parent))
        except Exception as e:
            blocked[i] = f'preparation failure: {type(e).__name__}: {e}'
    write(OUT / 'source_differences.json', diffs)
    write(OUT / 'source_hashes.json', sources)
    provenance = {}
    for i in reuse:
        if i in blocked: continue
        row = old[i]
        provenance[i] = {'kind': 'reuse', 'notice': '复用旧最终审核，不是本轮新审',
                         'source': str(BASE / 'reviews.jsonl'), 'source_sha256': p.sha(BASE / 'reviews.jsonl'),
                         'record_line': list(old).index(i) + 1, 'record_sha256': p.digest(row),
                         'rule_revision': 'sft-review-v2', 'prompt_sha256': bm['config']['prompt_sha256'],
                         'evidence_refs': row['evidence_refs'], 'material_hashes': bm['materials'][i]}
        # Provenance wrapper, never copied into a new reviewer lane/cache receipt.
        write(OUT / 'reuse' / (i + '.json'), {'provenance': provenance[i], 'record': row})
    write(OUT / 'reuse_provenance.json', provenance)
    write(OUT / 'worklists.json', {'reuse': reuse, 'primary': fresh, 'targeted': targeted, 'secondary': repeat})
    frozen_names = ['prompt.txt', 'rules.json', 'rule_changes.md', 'model_config.json', 'candidates.json',
                    'blind_secondary_ids.json', 'rule_impact.json', 'reuse_provenance.json', 'worklists.json',
                    'baseline_hashes.json', 'source_hashes.json', 'source_differences.json']
    frozen = {str(OUT / name): p.sha(OUT / name) for name in frozen_names}
    frozen.update(file_hashes(OUT / 'reuse'))
    frozen[str(Path(__file__).resolve())] = p.sha(__file__)
    frozen[str(Path(p.__file__).resolve())] = p.sha(p.__file__)
    m = {'created_at': p.read(started)['at'], 'source': str(SOURCE), 'baseline': str(BASE), 'selected': pool,
         'config': config, 'rule_revision': REVISION, 'materials': material_hashes,
         'frozen_files': frozen, 'plans': {'primary': fresh, 'targeted': targeted, 'secondary': repeat},
         'reuse_ids': reuse, 'second_review_ids': repeat, 'preparation_incomplete': blocked}
    write(OUT / 'manifest.json', m)
    write(OUT / 'manifest.sha256.json', {'sha256': p.sha(OUT / 'manifest.json')})
    for lane in LANES: (OUT / 'reviews' / lane).mkdir(parents=True, exist_ok=True)
    verify()
    print(json.dumps({'candidates': len(pool), 'new': len(fresh), 'reuse': len(reuse),
                      'targeted': len(targeted), 'secondary': len(repeat), 'blocked': blocked}, ensure_ascii=False))


def key(m, i, lane):
    return p.digest({'materials': m['materials'][i], 'prompt': m['frozen_files'][str(OUT / 'prompt.txt')],
                     'rules': m['frozen_files'][str(OUT / 'rules.json')], 'config': m['config'], 'phase': lane})


def validate(record, i):
    folder = OUT / 'materials' / i
    if p.read(folder / 'rules.json')['schema_version'] != 'sft-review-v2':
        raise ValueError('strict schema version required')
    if any(type(t) is not int for t in record['coverage_turn_ids']):
        raise ValueError('invalid coverage turn type')
    if not isinstance(record['issues'], list): raise ValueError('issues must be a list')
    for reference in record['evidence_refs'] + [ref for issue in record['issues'] for ref in issue['evidence_refs']]:
        name, fragment = reference.split('#', 1)
        path = Path(name).resolve()
        if path.parent != folder.resolve() or path.name not in ('pack.json', 'private.json', 'checks.json', 'requirements.json', 'rules.json'):
            raise ValueError('reference outside assigned material')
        value = p.read(path)
        if not fragment.startswith('/'): raise ValueError('empty pointer')
        for part in fragment[1:].split('/'):
            part = part.replace('~1', '/').replace('~0', '~')
            if isinstance(value, list):
                if not part.isdigit() or str(int(part)) != part: raise ValueError('invalid array pointer')
                value = value[int(part)]
            else: value = value[part]
    p.validate(record, i, folder)
    if record['decision'] == 'reject' and record.get('reject_category') not in (
            'confirmed_quality_issue', 'material_corruption', 'insufficient_evidence'):
        raise ValueError('reject category required')
    # An issue must cite this teacher output or its exact visible input mapping,
    # in addition to any task/observation/background evidence, for each mask.
    pack = p.read(folder / 'pack.json')
    for t in record['exclude_loss_turn_ids']:
        index = next(n for n, turn in enumerate(pack['turns']) if turn['turn_id'] == t)
        prefix = str(folder / 'pack.json') + f'#/turns/{index}/'
        if not any(issue['certainty'] == 'confirmed' and t in issue['turn_ids'] and
                   any(ref in (prefix + 'output', prefix + 'input_refs') for ref in issue['evidence_refs'])
                   for issue in record['issues']):
            raise ValueError('mask needs exact turn evidence: ' + str(t))
    return record


def result_path(lane, i):
    if lane not in LANES: raise ValueError('invalid lane')
    return OUT / 'reviews' / lane / (i + '.json')


def done(m, lane, i, strict=False):
    path = result_path(lane, i); receipt = path.with_suffix('.receipt.json')
    if not path.exists() or not receipt.exists(): return False
    r = p.read(receipt)
    if r['cache_key'] != key(m, i, lane) or r['result_sha256'] != p.sha(path): return False
    if strict: validate(p.read(path), i)
    return True


def job_path(lane, i):
    return OUT / 'jobs' / lane / (i + '.json')


def claim(i, lane, agent='pending'):
    m = manifest(i)
    if i not in m['plans'].get(lane, []) and lane != 'adjudication': raise ValueError('unplanned job')
    if i in m['preparation_incomplete']: raise ValueError('unverifiable material')
    if done(m, lane, i): raise ValueError('already completed')
    path = job_path(lane, i)
    history = p.read(path) if path.exists() else []
    if history and history[-1]['status'] == 'running': raise ValueError('already running')
    if len(history) >= m['config']['max_attempts']: raise ValueError('attempt limit reached')
    attempt = len(history) + 1
    output = OUT / 'attempts' / lane / i / f'{attempt}.json'
    history.append({'agent': agent, 'status': 'running', 'started_at': p.now(),
                    'attempt': attempt, 'cache_key': key(m, i, lane), 'output': str(output)})
    write(path, history)
    output.parent.mkdir(parents=True, exist_ok=True)
    append_event({'event': 'claim', 'id': i, 'lane': lane, 'agent': agent, 'attempt': attempt})
    return {'id': i, 'lane': lane, 'output': str(output),
            'prompt': str(OUT / 'prompt.txt'), 'material': str(OUT / 'materials' / i),
            'episode': next(r['episode'] for r in m['selected'] if r['id'] == i)}


def bind(i, lane, agent):
    path = job_path(lane, i); h = p.read(path)
    if h[-1]['status'] != 'running': raise ValueError('not running')
    h[-1]['agent'] = agent; write(path, h)
    append_event({'event': 'bind', 'id': i, 'lane': lane, 'agent': agent})


def accept(i, lane):
    m = manifest(i); path = job_path(lane, i); h = p.read(path)
    if done(m, lane, i): return {'id': i, 'status': 'already_done'}
    if h[-1]['status'] != 'running': raise ValueError('no running attempt')
    output = Path(h[-1]['output'])
    record = validate(p.read(output), i)
    if lane == 'adjudication' and not record.get('adjudication_reason'):
        raise ValueError('adjudication requires evidence resolution')
    if h[-1]['cache_key'] != key(m, i, lane): raise ValueError('stale job')
    result = result_path(lane, i)
    record.update(schema_version='sft-review-v2', rule_revision=REVISION)
    write(result, record)
    write(result.with_suffix('.receipt.json'), {'cache_key': key(m, i, lane), 'result_sha256': p.sha(result),
                                               'attempt_output': str(output), 'attempt_sha256': p.sha(output),
                                               'agent': h[-1]['agent'], 'accepted_at': p.now()})
    h[-1].update(status='done', finished_at=p.now()); write(path, h)
    append_event({'event': 'accepted', 'id': i, 'lane': lane, 'decision': record['decision']})
    return {'id': i, 'decision': record['decision'], 'lane': lane}


def fail(i, lane, error):
    path = job_path(lane, i); h = p.read(path)
    if h[-1]['status'] != 'running': raise ValueError('no running attempt')
    h[-1].update(status='failed', error=error, finished_at=p.now()); write(path, h)
    append_event({'event': 'processing_failure', 'id': i, 'lane': lane, 'error': error})


def status():
    m = manifest(); output = {}
    for lane, ids in m['plans'].items():
        completed, running, pending, failed, invalid = [], [], [], [], {}
        for i in ids:
            if i in m['preparation_incomplete']: continue
            try:
                if done(m, lane, i): completed.append(i); continue
            except Exception as e: invalid[i] = str(e)
            jp = job_path(lane, i); h = p.read(jp) if jp.exists() else []
            if h and h[-1]['status'] == 'running': running.append(i)
            elif len(h) >= m['config']['max_attempts']: failed.append(i)
            else: pending.append(i)
        output[lane] = {'done': completed, 'running': running, 'pending': pending,
                        'exhausted': failed, 'invalid': invalid}
    return {'lanes': output, 'reuse': len(m['reuse_ids']), 'preparation_incomplete': m['preparation_incomplete']}


def next_jobs(count):
    s = status(); jobs = []
    # Interleave blind secondary with primary; all planned IDs pre-frozen.
    for lane in ('targeted', 'secondary', 'primary'):
        for i in s['lanes'][lane]['pending']:
            jobs.append({'id': i, 'lane': lane})
            if len(jobs) >= count: return jobs
    return jobs


def verify():
    m = manifest(full=True)
    sources = p.read(OUT / 'source_hashes.json'); baseline = p.read(OUT / 'baseline_hashes.json')
    source_changes, baseline_changes = changed(sources), changed(baseline)
    rounds = 0; errors = []
    for i in m['materials']:
        pack = p.read(OUT / 'materials' / i / 'pack.json')
        for turn in pack['turns']:
            inp = dict(turn['input_metadata'], messages=[pack['messages'][r] for r in turn['input_refs']])
            output = dict(turn['output_metadata'], content=turn['output'], finish_reason=turn['finish_reason'])
            if inp != p.read(turn['source_input']) or output != p.read(turn['source_output']):
                errors.append({'id': i, 'turn': turn['turn_id'], 'error': 'nonlossless_roundtrip'})
            rounds += 1
    result = {'checked_at': p.now(), 'source_files_checked': len(sources), 'source_changes': source_changes,
              'baseline_files_checked': len(baseline), 'baseline_changes': baseline_changes,
              'exact_roundtrip_turns': rounds, 'roundtrip_errors': errors,
              'passed': not (source_changes or baseline_changes or errors)}
    write(OUT / 'source_verification.json', result)
    return result


def signature(record):
    return record['decision'], sorted(record['exclude_loss_turn_ids'])


def finalize(full_verify=False):
    m = manifest(); state = status(); old = {r['id']: r for r in map(json.loads, (BASE / 'reviews.jsonl').read_text().splitlines())}
    final, missing, conflicts, changes = [], [], [], []
    for row in m['selected']:
        i = row['id']
        if i in m['preparation_incomplete']:
            missing.append({'id': i, 'phase': 'preparation', 'error': m['preparation_incomplete'][i]}); continue
        if i in m['reuse_ids']:
            wrapper = p.read(OUT / 'reuse' / (i + '.json')); r = wrapper['record']
            p.validate(r, i, BASE / 'materials' / i)
            provenance = wrapper['provenance']; kind = 'reuse'
        else:
            kind = 'targeted' if i in m['plans']['targeted'] else 'primary'
            if i not in state['lanes'][kind]['done']:
                missing.append({'id': i, 'phase': kind, 'error': 'processing_incomplete',
                                'status': next((k for k in ('running', 'exhausted', 'pending') if i in state['lanes'][kind][k]), 'invalid')}); continue
            r = p.read(result_path(kind, i)); provenance = {'kind': kind, 'source': str(result_path(kind, i)),
                                                          'rule_revision': REVISION, 'source_sha256': p.sha(result_path(kind, i))}
            if full_verify: validate(r, i)
        if i in m['plans']['secondary']:
            if i not in state['lanes']['secondary']['done']:
                missing.append({'id': i, 'phase': 'secondary', 'error': 'processing_incomplete'}); continue
            second = p.read(result_path('secondary', i))
            if full_verify: validate(second, i)
            if signature(r) != signature(second):
                conflicts.append({'id': i, 'primary': r['decision'], 'secondary': second['decision'],
                                  'primary_exclude': r['exclude_loss_turn_ids'], 'secondary_exclude': second['exclude_loss_turn_ids']})
                if not done(m, 'adjudication', i):
                    missing.append({'id': i, 'phase': 'adjudication', 'error': 'unresolved_disagreement'}); continue
                r = p.read(result_path('adjudication', i))
                provenance.update(adjudication_source=str(result_path('adjudication', i)),
                                  adjudication_sha256=p.sha(result_path('adjudication', i)))
        # Evidence adjudications also cover old-sample changes and spot audits,
        # independently of whether an ID belongs to the blind-secondary sample.
        if done(m, 'adjudication', i):
            r = p.read(result_path('adjudication', i))
            if not r.get('adjudication_reason'): raise ValueError('undocumented adjudication')
            if full_verify: validate(r, i)
            provenance.update(adjudication_source=str(result_path('adjudication', i)),
                              adjudication_sha256=p.sha(result_path('adjudication', i)))
        if i in old and signature(r) != signature(old[i]):
            changes.append({'id': i, 'before': old[i], 'after': r})
        final.append(dict(r, review_origin=kind, provenance=provenance))
    # Each frozen candidate has exactly one final decision OR one unfinished state.
    assert len({r['id'] for r in final}) == len(final)
    assert set(r['id'] for r in final).isdisjoint(r['id'] for r in missing)
    assert {r['id'] for r in final + missing} == {r['id'] for r in m['selected']}
    for name, value in [('processing_status.json', state), ('incomplete.json', missing),
                        ('disagreements.json', conflicts), ('old_sample_changes.json', changes)]: write(OUT / name, value)
    (OUT / 'reviews.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in final))
    for label in ('keep', 'reject'): write(OUT / (label + '.json'), [r['id'] for r in final if r['decision'] == label])
    write(OUT / 'supervision_annotations.json', {r['id']: r['exclude_loss_turn_ids'] for r in final})
    write(OUT / 'all_candidates_status.json', sorted(
        [{'id': r['id'], 'status': r['decision'], 'exclude_loss_turn_ids': r['exclude_loss_turn_ids']} for r in final] +
        [dict(r, status='incomplete') for r in missing], key=lambda x: x['id']))
    highrisk = {'R4_side_effects', 'R2_grounding'}
    human_ids = {r['id'] for r in conflicts + changes} | {
        r['id'] for r in final if r['exclude_loss_turn_ids'] or any(
            issue['rule_id'] in highrisk or issue['certainty'] == 'uncertain' for issue in r['issues'])}
    human_ids |= set(m['plans']['targeted'])
    for label in ('keep', 'reject'):
        human_ids.update(r['id'] for r in sorted((r for r in final if r['decision'] == label), key=lambda x: x['id'])[:5])
    hp = OUT / 'human_review.json'; previous = {r['id']: r for r in p.read(hp)} if hp.exists() else {}
    human = []
    for r in final:
        if r['id'] not in human_ids: continue
        item = dict(id=r['id'], model_decision=r['decision'], model_reason=r['reason'],
                    issues=r['issues'], evidence_refs=r['evidence_refs'], exclude_loss_turn_ids=r['exclude_loss_turn_ids'],
                    material=str(OUT / 'materials' / r['id']), human_decision='', human_exclude_loss_turn_ids=None, human_notes='')
        if r['id'] in previous:
            item.update({k: v for k, v in previous[r['id']].items() if k.startswith('human_')})
        human.append(item)
    # Preserve manually inserted rows and fields, and never overwrite an existing Markdown worksheet.
    human.extend(r for i, r in previous.items() if i not in {r['id'] for r in human})
    write(hp, human)
    lines = ['# 人工抽查材料（人工标签未填写；模型复审不是人工验收）', '']
    for r in human:
        lines += [f"## {r['id']} · {r['model_decision']}", r['model_reason'],
                  f"材料：{r['material']}", '不监督：' + str(r['exclude_loss_turn_ids']),
                  '人工结论：' + r['human_decision'], '人工备注：' + r['human_notes'], '']
    (OUT / 'human_review_generated.md').write_text('\n'.join(lines))
    if not (OUT / 'human_review.md').exists(): (OUT / 'human_review.md').write_text('\n'.join(lines))
    failures = []
    for path in (OUT / 'jobs').glob('*/*.json'):
        failures += [dict(item, job=str(path)) for item in p.read(path) if item['status'] == 'failed']
    verification = verify() if full_verify else (p.read(OUT / 'source_verification.json') if (OUT / 'source_verification.json').exists() else None)
    counts = dict(Counter(r['decision'] for r in final))
    summary = {'updated_at': p.now(), 'created_at': m['created_at'], 'candidates': len(m['selected']),
               'finalized': len(final), 'counts': counts, 'incomplete': len(missing),
               'reuse': sum(r['review_origin'] == 'reuse' for r in final),
               'primary_completed': len(state['lanes']['primary']['done']),
               'targeted_completed': len(state['lanes']['targeted']['done']),
               'secondary_completed': len(state['lanes']['secondary']['done']),
               'processing_failures': len(failures), 'failures': failures,
               'disagreements': len(conflicts), 'old_sample_changes': len(changes),
               'issue_types': dict(Counter(i['tag'] for r in final for i in r['issues'])),
               'reject_categories': dict(Counter(r.get('reject_category', 'insufficient_evidence' if any(i['certainty'] == 'uncertain' and i['severity'] == 'major' for i in r['issues']) else 'confirmed_quality_issue') for r in final if r['decision'] == 'reject')),
               'uncertain_issues': sum(i['certainty'] == 'uncertain' for r in final for i in r['issues']),
               'keep_excluded_turns': sum(len(r['exclude_loss_turn_ids']) for r in final if r['decision'] == 'keep'),
               'human_rows': len(human), 'human_validation': 'pending', 'tokens': None, 'cost_usd': None,
               'usage_note': 'Actual subagent token/cost data unavailable; unknown.',
               'source_verification': verification}
    summary['wall_seconds'] = (datetime.fromisoformat(summary['updated_at']) - datetime.fromisoformat(m['created_at'])).total_seconds()
    write(OUT / 'summary.json', summary)
    passed = bool(full_verify and verification['passed'] and not missing)
    write(OUT / 'acceptance.json', {'checked_at': p.now(), 'passed': passed,
          'complete': not missing, 'candidate_accounting_passed': True, 'strict_schema': 'sft-review-v2',
          'rule_revision': REVISION, 'no_processing_failure_counted_as_reject': True,
          'human_validation': 'pending', 'verification': verification,
          'test_results': p.read(OUT / 'test_results.json') if (OUT / 'test_results.json').exists() else None})
    report = ['# 全量SFT轨迹审核', '', f"冻结{len(m['selected'])}条；最终{len(final)}条；未完成{len(missing)}条。",
              f"keep {counts.get('keep', 0)}；reject {counts.get('reject', 0)}。",
              f"新审{summary['primary_completed']}/{len(m['plans']['primary'])}；复用{summary['reuse']}；定向重审{summary['targeted_completed']}/{len(m['plans']['targeted'])}；盲复审{summary['secondary_completed']}/{len(m['plans']['secondary'])}。",
              f"处理失败{len(failures)}次（不计数据reject）；复审分歧{len(conflicts)}条；旧样本结果或mask变化{len(changes)}条。",
              f"证据不确定问题{summary['uncertain_issues']}个；keep中不监督{summary['keep_excluded_turns']}轮。",
              f"实际耗时{summary['wall_seconds']/60:.1f}分钟；模型gpt-6.1-terra / high，实际审核并发上限3（预留主agent）。",
              'token及费用未知。保留率、模型一致率、程序校验通过率都不是审核准确率。',
              '人工验收待填写。未导出SFT、未训练、未重放业务操作。', '',
              '问题类型：' + json.dumps(summary['issue_types'], ensure_ascii=False),
              '拒绝分类：' + json.dumps(summary['reject_categories'], ensure_ascii=False), '',
              '续跑命令和调度约束见 resume.md；逐项状态见 all_candidates_status.json；证据冻结核验见 source_verification.json。']
    (OUT / 'report.md').write_text('\n\n'.join(report) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('source_verification', 'failures', 'issue_types')}, ensure_ascii=False))


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'status', 'next', 'claim', 'bind', 'validate', 'accept', 'fail', 'verify', 'finalize'))
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--id'); parser.add_argument('--lane', choices=LANES)
    parser.add_argument('--agent', default='pending'); parser.add_argument('--record'); parser.add_argument('--error')
    parser.add_argument('--count', type=int, default=3); parser.add_argument('--full-verify', action='store_true')
    args = parser.parse_args(); OUT = args.output.resolve()
    if OUT == BASE.resolve() or OUT.is_relative_to(BASE.resolve()): raise ValueError('baseline must stay read only')
    OUT.mkdir(parents=True, exist_ok=True)
    # Scheduler mutations serialized; reviewer validation does not take the lock.
    with (OUT / '.scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.command == 'prepare': prepare()
        elif args.command == 'status': print(json.dumps(status(), ensure_ascii=False))
        elif args.command == 'next': print(json.dumps(next_jobs(args.count)))
        elif args.command == 'claim': print(json.dumps(claim(args.id, args.lane, args.agent), ensure_ascii=False))
        elif args.command == 'bind': bind(args.id, args.lane, args.agent)
        elif args.command == 'validate': validate(p.read(args.record), args.id); print('valid')
        elif args.command == 'accept': print(json.dumps(accept(args.id, args.lane)))
        elif args.command == 'fail': fail(args.id, args.lane, args.error)
        elif args.command == 'verify': print(json.dumps(verify(), ensure_ascii=False))
        elif args.command == 'finalize': finalize(args.full_verify)


if __name__ == '__main__': main()
