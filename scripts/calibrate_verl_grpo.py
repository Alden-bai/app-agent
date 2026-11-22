"""Run independent train-task controls and bind admissions to the tested isolation."""
import argparse
import asyncio
import json
from pathlib import Path
import sqlite3

from harness_awm.backend import Budget, ScriptedBackend
from harness_awm.contracts import ROOT, RunConfig
from harness_awm.evaluation import evaluate_episode
from harness_awm.io import db_fingerprint, digest, file_hash, read_json, verify_seal, write_json
from harness_awm.judge_inputs import sources_from_manifest
from harness_awm.judge_runtime import config_hash
from harness_awm.reward_policy import AdmissionManifest, VERSION
from harness_awm.reward_values import COMPOSITE_VERSION
from harness_awm.runner import EpisodeRunner
from harness_awm.training.baseline import isolation_probe
from harness_awm.training.tasks import load_tasks, select_tasks

TASKS = ['e_commerce_marketplace_10:0', 'fleet_management_6:1', 'membership_management_2:0']
KINDS = ('complete', 'positive_replay', 'partial', 'unexecuted')


def state(path, sql):
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
        rows = db.execute(sql).fetchall()
    if len(rows) != 1:
        raise ValueError('ambiguous control target')
    return [bool(value) for value in rows[0]]


def actions(spec, kind):
    calls = [] if kind == 'unexecuted' else (
        spec.get('partial', spec['calls'][:1]) if kind == 'partial' else spec['calls'])
    sequence = []
    seen = set()
    for name, arguments in calls:
        if name not in seen:
            sequence.append({'type': 'get_tool_schema', 'name': name})
            seen.add(name)
        sequence.append({'type': 'call_tool', 'name': name, 'arguments': arguments})
    positive = kind in {'complete', 'positive_replay'}
    final = {'type': 'finish', 'status': 'completed' if positive else 'incomplete',
             'answer': spec.get('answer', 'Requested profile changes executed.') if positive else 'Changes are incomplete.'}
    if not positive:
        final['reason'] = 'Independent control intentionally leaves obligations unfinished.'
    sequence.append(final)
    if kind == 'unexecuted':
        sequence.append(dict(final))  # AWM permits a no-tool finish after its one correction.
    return sequence


async def assess_control(folder, task, config, budget, semaphore):
    """One bounded retry for an invalid quality output, never for a valid score."""
    result = await evaluate_episode(folder, task, config, budget, semaphore)
    if (result.get('evaluation_valid') is True and result.get('business_label') != 'fail'
            and (result.get('trajectory_quality') or {}).get('status') == 'error'):
        previous = result
        result = await evaluate_episode(folder, task, config, budget, semaphore)
        if (result.get('business_label'), result.get('task_status')) != (
                previous.get('business_label'), previous.get('task_status')):
            raise ValueError('cached_business_judgment_changed_during_quality_repair')
        write_json(Path(folder) / 'quality_retry.json', {
            'reason': 'invalid_quality_output', 'previous_assessment_hash': digest(previous),
            'assessment_hash': digest(result), 'business_execution_repeated': False})
    return result


async def calibrate(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tasks, _, sources = load_tasks(args.tasks, 'grpo_train')
    tasks = select_tasks(tasks, getattr(args, 'task', None) or TASKS)
    raw = read_json(args.config)
    raw.update(judge_inputs_manifest=sources, reward_admissions=str(output / 'reward_admissions.json'),
               judge_rate_db=raw.get('judge_rate_db') or str(output / 'rate.sqlite'))
    config = RunConfig.model_validate(raw)
    if config.purpose != 'development' or config.isolation not in {'landlock', 'strict', 'private-root'}:
        raise ValueError('calibration requires isolated development workers')
    probe = isolation_probe(config, output / 'isolation')
    if not probe['passed']:
        raise ValueError(f'isolation unavailable: {probe}')
    from harness_awm.judge_rate_limit import install
    install(config, output / 'rate.sqlite')
    specs = read_json(getattr(args, 'controls', None) or ROOT / 'configs/training/grpo-calibration-controls.json')
    write_json(output / 'config.json', config.model_dump())
    write_json(output / 'expectations.json', {'specs': specs, 'tasks': [t.model_dump() for t in tasks],
        'judge_config_hash': config_hash(config), 'expected': {k: 'pass' if k in KINDS[:2] else 'fail' for k in KINDS}})
    budget = Budget(output / 'budget.sqlite', config.max_physical_requests, config.max_total_tokens)
    scripted = RunConfig.model_validate({**config.model_dump(), 'actor': 'scripted'})
    sem = asyncio.Semaphore(config.judge_concurrency)
    runner = EpisodeRunner(scripted, budget, sem)
    entries, reports = {}, []
    for task in tasks:
        spec, controls, trajectory_controls = specs.get(f'{task.scenario}:{task.native_task_id}', specs.get(task.scenario)), [], []
        if spec is None:
            raise ValueError('missing independent control specification: ' + task.task_instance_id)
        for kind in KINDS:
            folder = output / task.scenario / kind
            await runner.run(task, folder, f'calibration/{task.scenario}/{kind}',
                             ScriptedBackend(actions(spec, kind)), evaluate_after=False)
            verify_seal(folder)
            initial, final = [state(folder / f'{name}.db', spec['sql']) for name in ('initial', 'final')]
            positive = kind in KINDS[:2]
            assert all(final) if positive else not all(final), 'control has unexpected business state'
            if kind == 'partial':
                assert any(b and not a for a, b in zip(initial, final)), 'partial control did not change state'
            if kind == 'unexecuted':
                assert db_fingerprint(folder / 'initial.db') == db_fingerprint(folder / 'final.db')
            if task.scenario == 'membership_management_2':
                def tags(name):
                    with sqlite3.connect(f'file:{folder / name}?mode=ro', uri=True) as db:
                        return set(db.execute('SELECT user_id,tag FROM user_tags').fetchall())
                assert tags('initial.db') <= tags('final.db'), 'control removed existing tags'
            expected = 'pass' if positive else 'fail'
            write_json(folder / 'independent_assertion.json', {'initial': initial, 'final': final,
                'expected': expected, 'sql': spec['sql'], 'before_judge': True})
            assessment = await assess_control(folder, task, scripted, budget, sem)
            result = {'kind': kind, 'path': str(folder), 'expected': expected,
                'actual': assessment.get('business_label'), 'evaluation_valid': assessment.get('evaluation_valid'),
                'matches': assessment.get('evaluation_valid') is True and assessment.get('business_label') == expected,
                'test_only': read_json(folder / 'episode_result.json')['test_only'],
                'assessment_hash': digest(assessment)}
            write_json(folder / 'calibration_result.json', result)
            controls.append(result)
            if kind == 'complete' and config.reward_policy == COMPOSITE_VERSION:
                quality = assessment.get('trajectory_quality') or {}
                dims = (quality.get('assessment') or {}).get('dimensions', [])
                trajectory_controls.append({'kind': 'clean', 'path': str(folder), 'test_only': True,
                    'matches': quality.get('status') == 'valid' and len(dims) == 3 and all(d['grade'] == 0 for d in dims),
                    'assessment_hash': digest(quality), 'grades': {d['dimension']: d['grade'] for d in dims}})
            print(json.dumps({'task': task.scenario, **result, 'budget': budget.summary()}), flush=True)
        if config.reward_policy == COMPOSITE_VERSION:
            folder = output / task.scenario / 'redundant_quality_control'
            name = spec['calls'][0][0]
            # Three identical discoveries (including actions()'s schema read) demonstrate
            # redundancy without reaching the harness's four-repeat stop boundary.
            redundant = [{'type': 'get_tool_schema', 'name': name} for _ in range(2)] + actions(spec, 'complete')
            await runner.run(task, folder, f'calibration/{task.scenario}/redundant',
                             ScriptedBackend(redundant), evaluate_after=False)
            assert all(state(folder / 'final.db', spec['sql'])), 'redundant control must still succeed'
            assessment = await assess_control(folder, task, scripted, budget, sem)
            quality = assessment.get('trajectory_quality') or {}
            dims = (quality.get('assessment') or {}).get('dimensions', [])
            trajectory_controls.append({'kind': 'redundant', 'path': str(folder), 'test_only': True,
                'matches': assessment.get('evaluation_valid') is True and assessment.get('business_label') == 'pass'
                    and quality.get('status') == 'valid' and len(dims) == 3 and any(d['grade'] > 0 for d in dims),
                'assessment_hash': digest(quality), 'grades': {d['dimension']: d['grade'] for d in dims}})
            write_json(folder / 'quality_calibration.json', trajectory_controls[-1])
        passed = len(controls) == 4 and all(c['matches'] and c['test_only'] for c in controls)
        if config.reward_policy == COMPOSITE_VERSION:
            passed = passed and len(trajectory_controls) == 2 and all(c['matches'] for c in trajectory_controls)
        proof = output / task.scenario / 'calibration.json'
        report = {'passed': passed, 'task_instance_id': task.task_instance_id, 'isolation': config.isolation,
                  'judge_config_hash': config_hash(config), 'controls': controls,
                  'trajectory_controls': trajectory_controls}
        write_json(proof, report)
        reports.append(report)
        if passed:
            source = sources_from_manifest(task, sources)
            entries[task.task_instance_id] = dict(status='admitted', calibration_ref=str(proof),
                calibration_sha256=file_hash(proof), purpose='development', isolation=config.isolation, policy_version=config.reward_policy,
                task_instance_id=task.task_instance_id, original_hash=digest(source['original']),
                rubric_hash=digest(source['rubric']), initial_state_hash=task.initial_state_hash,
                environment_code_hash=task.environment_code_hash, tool_schema_hash=task.tool_schema_hash,
                judge_config_hash=config_hash(config))
    manifest = AdmissionManifest.model_validate({'schema_version': 'awm-reward-admissions-v2', 'tasks': entries})
    write_json(output / 'reward_admissions.json', manifest.model_dump())
    report = {'passed': len(entries) == len(tasks), 'admitted': len(entries), 'tasks': reports, 'budget': budget.summary()}
    write_json(output / 'summary.json', report)
    if not report['passed']:
        raise ValueError('calibration incomplete; inspect preserved controls')


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--task', action='append', help='Exact train task; repeat to expand the calibrated pool')
    p.add_argument('--controls', type=Path, help='Independent actions and SQL assertions for selected tasks')
    asyncio.run(calibrate(p.parse_args()))


if __name__ == '__main__':
    main()
