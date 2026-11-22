"""Bind every frozen train task to an explicit, already validated shared Judge suite."""
import argparse
from pathlib import Path

from harness_awm.contracts import ROOT, RunConfig
from harness_awm.io import digest, file_hash, read_json, write_json
from harness_awm.judge_inputs import load_json
from harness_awm.judge_runtime import config_hash
from harness_awm.judge_v2 import normalize_rubric
from harness_awm.reward_policy import AdmissionManifest
from harness_awm.training.grpo import preflight
from harness_awm.training.tasks import load_tasks


def prepare(args):
    tasks, manifest, sources = load_tasks(args.tasks, 'grpo_train')
    if len(tasks) != args.expected_tasks:
        raise ValueError('unexpected full training task count')
    config = RunConfig.model_validate(read_json(args.config)).model_copy(update={
        'judge_inputs_manifest': sources, 'reward_admissions': str(args.calibrated.resolve())})
    if config.purpose != 'development' or config.isolation not in {'landlock', 'strict', 'private-root'}:
        raise ValueError('shared calibration requires isolated development workers')
    prior = AdmissionManifest.model_validate(read_json(args.calibrated))
    if prior.schema_version != 'awm-reward-admissions-v2':
        raise ValueError('requires actual per-task control calibrations, not another shared suite')
    controlled = [t for t in tasks if t.task_instance_id in prior.tasks]
    if len(controlled) < 2 or len(controlled) != len(prior.tasks):
        raise ValueError('control tasks must be distinct members of this frozen training split')
    preflight(controlled, config, 4)
    judge_hash = config_hash(config)
    index = load_json(sources)['tasks']
    root = Path(sources).parent
    bindings = {}
    for task in tasks:
        values = {}
        for name in ('original', 'rubric'):
            ref = index[task.task_instance_id][name]
            path = (root / ref['path']).resolve()
            if not path.is_relative_to(root) or file_hash(path) != ref['sha256']:
                raise ValueError('source asset mismatch:' + task.task_instance_id)
            values[name] = load_json(path)
        original, rubric = values['original'], values['rubric']
        if original['task_id'] != f'{task.scenario}:{task.native_task_id}' or original['task'] != task.instruction:
            raise ValueError('source instruction mismatch:' + task.task_instance_id)
        normalize_rubric(original, rubric)
        bindings[task.task_instance_id] = dict(task_instance_id=task.task_instance_id,
            original_hash=digest(original), rubric_hash=digest(rubric), initial_state_hash=task.initial_state_hash,
            environment_code_hash=task.environment_code_hash, tool_schema_hash=task.tool_schema_hash,
            judge_config_hash=judge_hash)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    suite = output / 'shared-calibration.json'
    write_json(suite, dict(schema_version='awm-shared-judge-calibration-v1',
        calibration_scope='shared-suite', per_task_calibrated=False, purpose='development', isolation=config.isolation,
        split='grpo_train', policy_version=config.reward_policy, judge_config_hash=judge_hash,
        task_manifest=manifest, task_count=len(tasks), task_bindings={k:digest(v) for k,v in bindings.items()},
        calibrated_controls=[e.model_dump() for e in prior.tasks.values()],
        interpretation='Controls validate the shared Judge pipeline; each task has source checks, not behavioral calibration.'))
    entries = {key:dict(**binding, status='admitted', policy_version=config.reward_policy,
        purpose='development', isolation=config.isolation, calibration_scope='shared-suite', per_task_calibrated=False,
        calibration_ref=str(suite), calibration_sha256=file_hash(suite)) for key,binding in bindings.items()}
    admission = output / 'reward_admissions.json'
    write_json(admission, AdmissionManifest.model_validate(dict(schema_version='awm-reward-admissions-v3',tasks=entries)).model_dump())
    config = config.model_copy(update={'reward_admissions': str(admission)})
    write_json(output / 'config.json', config.model_dump())
    report = preflight(tasks, config, 4)
    report.update(task_count=len(tasks), scenarios=len({t.scenario for t in tasks}),
                  calibrated_control_tasks=len(controlled), per_task_calibrated=False, task_manifest=manifest)
    write_json(output / 'audit.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--calibrated',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-tasks',type=int,default=2746)
    prepare(p.parse_args())
