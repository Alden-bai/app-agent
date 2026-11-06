"""Fail-closed admission and sample disposition for environment-screened runs.

Call admission before starting a future episode. Call sample_disposition before
assigning reward. A business failure alone never diagnoses an environment fault.
"""
from pathlib import Path
from .io import digest, file_hash, read_json


def admission(admission_path, task_id, initial_path, environment_path):
    record=read_json(admission_path)
    if not record.get('complete'):
        raise ValueError('environment_screen_incomplete')
    for key,hashkey in [('screen_manifest','screen_manifest_sha256'),('tasks_ref','tasks_sha256')]:
        if file_hash(record[key])!=record[hashkey]:
            raise ValueError('environment_screen_hash_mismatch')
    import json
    with Path(record['tasks_ref']).open() as stream:
        matches=[row for line in stream if (row:=json.loads(line))['task_id']==task_id]
    if len(matches)!=1:
        raise ValueError('task_missing_or_ambiguous')
    row=matches[0]
    if file_hash(initial_path)!=row['initial_sha256'] or digest(read_json(environment_path))!=row['environment_sha256']:
        raise ValueError('screened_environment_version_mismatch')
    return dict(allowed=row['sampling_eligible'],environment_status=row['environment_status'],
                rubric_accepted=row['rubric_accepted'],task_id=task_id,
                limits='Finite screening; runtime fault detection remains required.')


def sample_disposition(*, faults=(), outcome_unknown=False, evaluation_valid=False,
                       business_result=None, recovered=False):
    """faults: kind, confirmed, affects_outcome, evidence_refs.

The caller must supply actual fault evidence/attribution; raw 4xx/isError and an
actor's own claim are insufficient. Recovered technical faults remain tainted
under this conservative sampling policy, rather than silently counting success.
"""
    technical={'database','tool','environment','harness','model_api'}
    proven=[f for f in faults if f.get('kind') in technical and f.get('confirmed') is True
            and f.get('evidence_refs') and f.get('affects_outcome') is True]
    if proven:
        return dict(disposition='discard_sample',assign_reward=False,label='technical_fault',
                    preserve_evidence=True,retry_write=False)
    suspicious=[f for f in faults if f.get('kind') in technical|{'unknown','judge','rubric'}]
    if outcome_unknown or suspicious or not evaluation_valid or business_result not in {'pass','fail'}:
        return dict(disposition='quarantine',assign_reward=False,label='requires_review',
                    preserve_evidence=True,retry_write=False)
    return dict(disposition='keep',assign_reward=True,label='valid_success' if business_result=='pass' else 'valid_model_failure',
                preserve_evidence=True,retry_write=False)
