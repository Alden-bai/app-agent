"""Validate and relocate the frozen inference skill bank; no GPU or Judge calls."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness_awm.contracts import SkillConfig
from harness_awm.io import file_hash, read_json, write_json
from harness_awm.skillbank import SkillBank
from harness_awm.skillbank_build import training_allowlist
from harness_awm.training.tasks import load_tasks


def prepare(bundle, tasks, output):
    bundle, tasks, output = Path(bundle).resolve(), Path(tasks).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    raw = read_json(bundle / 'inference_skills.json')
    # The distributable template is relative to its own folder, never the caller's cwd.
    bank_path = Path(raw['bank_path'])
    if not bank_path.is_absolute():
        bank_path = bundle / bank_path
    config = SkillConfig.model_validate({**raw, 'bank_path': str(bank_path.resolve())})
    bank = SkillBank(config)
    if bank.data['runtime_manifest_sha256'] != file_hash(tasks / 'manifest.json'):
        raise ValueError('skill_runtime_manifest_mismatch')
    allowed = training_allowlist(tasks)
    source_hashes = {s['sha256'] for s in bank.data['sources']}
    for skill in bank.skills:
        for evidence in skill['evidence']:
            task = allowed.get(evidence['task_instance_id'])
            if task is None or evidence['archive_sha256'] not in source_hashes:
                raise ValueError('skill_evidence_outside_frozen_training_sources')
            if skill['kind'] == 'workflow' and (skill['scenario'] != task['scenario'] or
                    skill['tool_schema_hash'] != task['tool_schema_hash']):
                raise ValueError('skill_evidence_environment_mismatch')
    validation, _, _ = load_tasks(tasks, 'validation')
    matched = workflows = 0
    for task in validation:
        catalog = read_json(Path(task.environment_ref).with_name('tools.json'))
        selected = bank.retrieve(task, catalog['tools'] if isinstance(catalog, dict) else catalog)
        matched += bool(selected)
        workflows += any(s['kind'] == 'workflow' for s in selected)
    report = dict(bank_sha256=bank.sha256, skills=len(bank.skills),
        workflows=sum(s['kind'] == 'workflow' for s in bank.skills),
        failure_lessons=sum(s['kind'] == 'failure_lesson' for s in bank.skills),
        source_split='grpo_train', training_evidence_checked=True,
        runtime_manifest_sha256=bank.data['runtime_manifest_sha256'],
        validation_tasks=len(validation), validation_tasks_with_skills=matched,
        validation_tasks_with_workflows=workflows, evaluation_run=False, training_updates=False,
        raw_archives_included=False)
    write_json(output, config.model_dump())
    write_json(output.with_suffix('.verification.json'), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT / 'delivery/skillrl')
    parser.add_argument('--tasks', type=Path, default=ROOT / 'data/training_runtime_20261106_v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.bundle, args.tasks, args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
