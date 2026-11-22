"""Dataset admission with shared Judge controls, not per-task calibration."""
from pathlib import Path
from .io import digest, file_hash, read_json


def verify_shared_suite(entry, binding, *, verified=None):
    from .reward_policy import DevelopmentAdmission, verify_development_calibration
    path = Path(entry.calibration_ref)
    if file_hash(path) != entry.calibration_sha256:
        raise ValueError('reward_calibration_changed')
    key = (str(path.resolve()), entry.calibration_sha256)
    proof = verified[key] if verified is not None and key in verified else read_json(path)
    if (proof.get('schema_version') != 'awm-shared-judge-calibration-v1'
            or proof.get('calibration_scope') != 'shared-suite'
            or proof.get('per_task_calibrated') is not False
            or proof.get('judge_config_hash') != entry.judge_config_hash
            or proof.get('policy_version') != entry.policy_version
            or proof.get('purpose') != 'development' or proof.get('isolation') != entry.isolation
            or proof.get('split') != 'grpo_train'
            or proof.get('task_bindings', {}).get(entry.task_instance_id) != digest(binding)):
        raise ValueError('shared_calibration_binding_mismatch')
    controls = proof.get('calibrated_controls', [])
    if len(controls) < 2 or len({c.get('task_instance_id') for c in controls}) != len(controls):
        raise ValueError('shared_calibration_requires_distinct_controls')
    if verified is not None and key in verified:
        return
    for raw in controls:
        control = DevelopmentAdmission.model_validate(raw)
        if (control.status != 'admitted' or control.judge_config_hash != entry.judge_config_hash
                or control.policy_version != entry.policy_version
                or control.purpose != entry.purpose or control.isolation != entry.isolation):
            raise ValueError('shared_calibration_control_binding_mismatch')
        verify_development_calibration(control, control.task_instance_id)
    if verified is not None:
        verified[key] = proof
