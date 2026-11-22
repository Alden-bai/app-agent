"""Re-evaluate existing SFT weights and base under one corrected stopping policy."""
import argparse
import fcntl
import os
from pathlib import Path

from harness_awm.contracts import ROOT, RunConfig
from harness_awm.io import file_hash, now, read_json, write_json
from harness_awm.tracking import SwanLabTracker
from .actor import STOP_POLICY
from .selection import checked_plan, sources, compact, choose_best, run_evaluation


def prepare(root, source):
    from .baseline import isolation_probe
    old = read_json(source / 'plan.json')
    if file_hash(source / 'plan.json') != read_json(source / 'plan.sha256.json')['sha256']:
        raise ValueError('source plan hash mismatch')
    if root.exists():
        raise ValueError('new evaluation directory required')
    candidates = read_json(source / 'candidates.json')
    expected = {f"{trial['name']}-epoch{epoch}" for trial in old['trials'] for epoch in range(1, old['epochs'] + 1)}
    if set(candidates) != expected or 'base' in candidates:
        raise ValueError('incomplete source SFT candidates')
    # Bind to the already frozen per-candidate validation manifest as well.
    for name, checkpoint in candidates.items():
        manifest_path = source / 'validation' / name / 'manifest.json'
        if file_hash(manifest_path) != read_json(manifest_path.with_name('manifest.sha256.json'))['sha256']:
            raise ValueError('source candidate manifest changed')
        if read_json(manifest_path)['checkpoint'] != checkpoint:
            raise ValueError('candidate identity mismatch')
    root.mkdir(parents=True)
    plan = {**old, 'created_at': now(), 'source_run': str(source),
        'source_plan_sha256': file_hash(source / 'plan.json'),
        'source_candidates_sha256': file_hash(source / 'candidates.json'),
        'schema_version': 'awm-sft-reevaluation-v1', 'stop_policy': STOP_POLICY,
        'effective_eos_token_ids': [248044, 248046], 'source_hashes': sources(),
        'sft_candidates': candidates, 'base_candidate': old['checkpoint'],
        'training_updates': False, 'old_metrics_reused': False,
        'reason': 'Chat end-of-turn token was absent from generation stop set; re-evaluate base and all SFT candidates consistently.'}
    if not isolation_probe(RunConfig.model_validate(plan['config']), root / 'isolation_probe')['passed']:
        raise RuntimeError('isolation probe failed')
    write_json(root / 'plan.json', plan)
    write_json(root / 'plan.sha256.json', {'sha256': file_hash(root / 'plan.json')})
    write_json(root / 'status.json', {'stage': 'prepared', 'updated_at': now(), 'training_updates': False})


def require_drained(source):
    """Never overlap the old GPU workers or interrupt in-flight Judge requests."""
    import sqlite3
    queue = source / 'validation/queue.sqlite'
    with sqlite3.connect(queue.resolve().as_uri() + '?mode=ro', uri=True) as db:
        counts = dict(db.execute('SELECT state,count(*) FROM jobs GROUP BY state'))
    if counts.get('pending', 0) or counts.get('claimed', 0):
        raise ValueError('source workers have not drained')
    for pid in read_json(source / 'validation/launch.json')['worker_pids']:
        proc = Path(f'/proc/{pid}')
        if proc.exists() and (proc / 'stat').read_text().split()[2] != 'Z':
            raise ValueError('source worker still alive')


def execute(root):
    plan = checked_plan(root)
    require_drained(Path(plan['source_run']))
    # This diagnostic must demonstrate actual early stopping on a real SFT
    # checkpoint before spending Judge budget on the full corrected protocol.
    smoke = read_json(root / 'stopping_smoke.json')
    if smoke.get('stop_policy') != STOP_POLICY or smoke.get('passed') is not True:
        raise ValueError('corrected stopping smoke has not passed')
    with (root / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'launch.json').exists():
            raise ValueError('evaluation already launched')
        write_json(root / 'launch.json', {'pid': os.getpid(), 'started_at': now()})
        tracker = SwanLabTracker(root, mode='online', config={
            'stage': 'sft-evaluation-corrected-eos', 'stop_policy': STOP_POLICY,
            'old_metrics_reused': False, 'training_updates': False,
            'selection_rule': plan['selection_rule'], 'eval_workers': plan['eval_workers']})
        write_json(root / 'tracking.json', {'url': tracker.url})
        failed = True
        try:
            candidates = plan['sft_candidates']
            reports, step = run_evaluation(root, plan, {'base': plan['base_candidate'], **candidates},
                                           'validation', tracker, 0)
            # Base provides a matched reference but is not an SFT candidate.
            sft_reports = {k: reports[k] for k in candidates}
            best = choose_best(sft_reports, plan['minimum_coverage'])
            selection = {'selected': best, 'checkpoint': candidates[best], 'selected_at': now(),
                'stop_policy': STOP_POLICY, 'selection_rule': plan['selection_rule'],
                'validation': {k: compact(r) for k, r in reports.items()},
                'test_used_for_selection': False, 'base_excluded_from_sft_selection': True}
            write_json(root / 'selection.json', selection)
            write_json(root / 'selection.sha256.json', {'sha256': file_hash(root / 'selection.json')})
            test, step = run_evaluation(root, plan, {'base': plan['base_candidate'], best: candidates[best]},
                                       'test', tracker, step)
            summary = {'selected': best, 'checkpoint': candidates[best], 'stop_policy': STOP_POLICY,
                'baseline': {'validation': compact(reports['base']), 'test': compact(test['base'])},
                'validation': compact(reports[best]), 'test': compact(test[best]),
                'test_used_for_selection': False, 'old_metrics_reused': False,
                'completed_at': now(), 'swanlab_url': tracker.url}
            write_json(root / 'summary.json', summary)
            tracker.log({'summary': summary}, step=step)
            write_json(root / 'status.json', {'stage': 'finished', 'selected': best, 'updated_at': now()})
            failed = False
        except BaseException as exc:
            write_json(root / 'status.json', {'stage': 'failed', 'error': f'{type(exc).__name__}: {exc}', 'updated_at': now()})
            raise
        finally:
            tracker.finish(failed=failed)


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'run'))
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--source', type=Path)
    a = p.parse_args()
    if a.action == 'prepare':
        if a.source is None:
            p.error('--source required')
        prepare(a.output.resolve(), a.source.resolve())
    else:
        execute(a.output.resolve())


if __name__ == '__main__':
    main()
