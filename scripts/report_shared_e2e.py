"""Report measured throughput and outcome differences without mixing production scores."""
import argparse
from collections import Counter
from pathlib import Path
import sqlite3
import sys
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness_awm.io import read_json, write_json, now, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--upload', action='store_true')
    a = parser.parse_args(); summary = read_json(a.run / 'summary.json')
    detail = {}
    for mode in ('baseline', 'shared'):
        stage = a.run / mode
        timings = [read_json(p) for p in (stage / 'timings').glob('*.json')]
        by_candidate = {}
        for candidate in sorted({r['job']['candidate'] for r in timings}):
            rows = [r for r in timings if r['job']['candidate'] == candidate]
            by_candidate[candidate] = {'attempts': len(rows),
                'scored': sum(r['evaluation']['scored'] for r in rows),
                'confirmed_successes': sum(r['evaluation']['confirmed_successes'] for r in rows),
                'median_episode_seconds': median(r['end'] - r['start'] for r in rows)}
        counters = Counter(); judge_errors = Counter(); generated_tokens = 0; trajectory_times = []
        with sqlite3.connect('file:' + str(stage / 'queue.sqlite') + '?mode=ro', uri=True) as db:
            outputs = [Path(r[0]) for r in db.execute('SELECT output FROM jobs')]
        for output in outputs:
            folder = output / 'episodes/t00000/a0'; result = read_json(folder / 'episode_result.json')
            trajectory_times.append((folder / 'trajectory.json').stat().st_mtime)
            counters['turns'] += result['turns']; counters['tool_calls'] += result['tool_calls']
            counters['model_seconds_sum'] += result.get('execution_metrics', {}).get('model_seconds', 0)
            for p in (folder / 'rewards_v2').glob('*.json'):
                for err in read_json(p).get('errors', []): judge_errors[str(err)] += 1
            for p in (folder / 'model_turns').glob('*/output.json'):
                generated_tokens += read_json(p).get('usage', {}).get('completion_tokens', 0)
        s = summary[mode]
        detail[mode] = {'by_candidate': by_candidate, 'judge_errors': dict(judge_errors),
            'execution_totals': dict(counters), 'generated_tokens': generated_tokens,
            'last_trajectory_wall_seconds': max(trajectory_times) - read_json(stage / 'go.json')['epoch'],
            'scored_per_minute': s['scored'] * 60 / s['wall_seconds'],
            'coverage': s['scored'] / s['jobs']}
    batches = []
    for p in (a.run / 'shared').glob('service_*.jsonl'):
        import json
        batches.extend(json.loads(line) for line in p.read_text().splitlines() if line.strip())
    measured = [r for r in batches if r['event'] == 'batch']
    detail['batching'] = {'histogram': dict(Counter(r['batch'] for r in measured)),
        'mean_batch': sum(r['batch'] for r in measured) / len(measured) if measured else None,
        'longest_batch_seconds': max((r['seconds'] for r in measured), default=None),
        'oom_splits': sum(r['event'] == 'oom_split' for r in batches),
        'service_peak_allocated_GiB': max((r['peak_allocated_GiB'] for r in measured), default=None)}
    detail['effective_scored_speedup'] = detail['shared']['scored_per_minute'] / detail['baseline']['scored_per_minute']
    detail['at'] = now(); detail['summary_sha256'] = file_hash(a.run / 'summary.json')
    detail['external_load_note'] = 'Judge-only recovery (4 clients) was active during shared measurement. GPU was exclusive.'
    write_json(a.run / 'analysis.json', detail)
    if a.upload:
        from harness_awm.tracking import SwanLabTracker
        tracker = SwanLabTracker(a.run / 'report', mode='online', config={
            'stage': 'shared-batch-vs-18-process-e2e', 'validation_tasks': 10,
            'checkpoints': 5, 'jobs_per_mode': 50, 'baseline_model_processes': 18,
            'shared_model_processes': 5, 'shared_task_processes': 32,
            'benchmark_scores_excluded_from_production': True, 'reference_kernels_preserved': True})
        try:
            tracker.log({'baseline': summary['baseline'], 'shared': summary['shared'],
                'wall_speedup': summary['wall_speedup'], 'time_saved_fraction': summary['wall_time_saved_fraction'],
                'effective_scored_speedup': detail['effective_scored_speedup'],
                'estimated_hours_saved': summary['estimated_hours_saved'],
                'same_output_jobs': summary['same_output_jobs'], 'same_status_jobs': summary['same_status_jobs'],
                'detail': detail}, step=0)
            write_json(a.run / 'tracking.json', {'url': tracker.url})
        finally: tracker.finish()
    print({'wall_speedup': summary['wall_speedup'], 'effective_scored_speedup': detail['effective_scored_speedup'],
        'hours_saved': summary['estimated_hours_saved'], 'batching': detail['batching']})


if __name__ == '__main__': main()
