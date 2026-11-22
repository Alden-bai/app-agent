import copy
from pathlib import Path

import pytest

from harness_awm.io import digest, file_hash, read_json, seal, write_json
from harness_awm.judge_runtime import config_hash
from harness_awm.reward_policy import AdmissionManifest, SharedCalibrationAdmission, admission_for
from harness_awm.reward_values import COMPOSITE_VERSION
from harness_awm.training.grpo import preflight
from test_training_grpo import setup_sources


@pytest.fixture(params=['landlock', 'strict'])
def shared(tmp_path, request):
    backend = request.param
    task, config, original, rubric = setup_sources(tmp_path)
    config = config.model_copy(update={'purpose':'development','isolation':backend,'reward_policy':COMPOSITE_VERSION})
    binding=dict(task_instance_id=task.task_instance_id,original_hash=digest(original),rubric_hash=digest(rubric),
                 initial_state_hash=task.initial_state_hash,environment_code_hash=task.environment_code_hash,
                 tool_schema_hash=task.tool_schema_hash,judge_config_hash=config_hash(config))
    controls=[]
    for index in range(2):
        folder=tmp_path/f'control-{index}'
        write_json(folder/'task.json',{'synthetic_test_only':True,'id':index})
        seal(folder)
        proof=folder/'calibration.json'
        write_json(proof,dict(passed=True,task_instance_id=f'control-{index}',isolation=backend,judge_config_hash=config_hash(config),
            controls=[dict(matches=True,test_only=True,path=str(folder)) for _ in range(4)],
            trajectory_controls=[dict(kind=kind,matches=True,test_only=True,path=str(folder)) for kind in ['clean','redundant']]))
        controls.append(dict(**{**binding,'task_instance_id':f'control-{index}'},status='admitted',purpose='development',isolation=backend,
            policy_version=COMPOSITE_VERSION,calibration_ref=str(proof),calibration_sha256=file_hash(proof)))
    suite=tmp_path/'suite.json'
    proof=dict(schema_version='awm-shared-judge-calibration-v1',calibration_scope='shared-suite',per_task_calibrated=False,
               purpose='development',isolation=backend,split='grpo_train',judge_config_hash=config_hash(config),
               policy_version=COMPOSITE_VERSION,task_bindings={task.task_instance_id:digest(binding)},calibrated_controls=controls)
    write_json(suite,proof)
    entry=dict(**binding,status='admitted',purpose='development',isolation=backend,policy_version=COMPOSITE_VERSION,
               calibration_ref=str(suite),calibration_sha256=file_hash(suite),calibration_scope='shared-suite',per_task_calibrated=False)
    manifest=dict(schema_version='awm-reward-admissions-v3',tasks={task.task_instance_id:entry})
    write_json(config.reward_admissions,manifest)
    return task,config,original,rubric,proof,manifest,suite


def test_explicit_shared_admission_and_full_preflight(shared):
    task,c,o,r,_,manifest,_=shared
    entry=admission_for(task,c,o,r)
    assert isinstance(entry,SharedCalibrationAdmission) and entry.per_task_calibrated is False
    assert preflight([task],c)['checked_tasks']==1
    other = 'landlock' if c.isolation == 'strict' else 'strict'
    with pytest.raises(ValueError, match='mismatch'):
        admission_for(task, c.model_copy(update={'isolation': other}), o, r)
    manifest['schema_version']='awm-reward-admissions-v2'
    with pytest.raises(ValueError,match='schema_mismatch'):AdmissionManifest.model_validate(manifest)


@pytest.mark.parametrize('change,match',[
    ({'split':'test'},'train_split'),({'split':'dev'},'train_split'),
    ({'tool_schema_hash':'changed'},'binding_mismatch')])
def test_shared_rejects_other_split_and_changed_world(shared,change,match):
    task,c,o,r,*_=shared
    with pytest.raises(ValueError,match=match):admission_for(task.model_copy(update=change),c,o,r)


def test_shared_revocation_and_changed_judge(shared):
    task,c,o,r,_,m,_=shared
    changed=c.model_copy(update={'judge_api':c.judge_api.model_copy(update={'model':'changed'})})
    with pytest.raises(ValueError,match='binding_mismatch'):admission_for(task,changed,o,r)
    m['tasks'][task.task_instance_id]['status']='revoked';write_json(c.reward_admissions,m)
    with pytest.raises(ValueError,match='revoked'):admission_for(task,c,o,r)


@pytest.mark.parametrize('change,match',[
    ('controls','distinct_controls'),('scope','binding_mismatch'),('task','binding_mismatch'),
    ('quality','trajectory_reward_calibration'),('evidence','evidence_hash_mismatch')])
def test_shared_revalidates_proofs(shared,change,match):
    task,c,o,r,proof,m,suite=shared
    if change=='controls':proof['calibrated_controls']=proof['calibrated_controls'][:1]
    if change=='scope':proof['per_task_calibrated']=True
    if change=='task':proof['task_bindings'][task.task_instance_id]='wrong'
    if change in ['quality','evidence']:
        control=proof['calibrated_controls'][0];p=Path(control['calibration_ref'])
        if change=='quality':
            d=read_json(p);d['trajectory_controls'][0]['matches']=False;write_json(p,d)
            control['calibration_sha256']=file_hash(p)
        else:write_json(p.parent/'task.json',{'changed':True})
    write_json(suite,proof);m['tasks'][task.task_instance_id]['calibration_sha256']=file_hash(suite);write_json(c.reward_admissions,m)
    with pytest.raises(ValueError,match=match):admission_for(task,c,o,r)


def test_shared_suite_hash_and_source_hash_remain_enforced(shared):
    task,c,o,r,_,_,suite=shared
    write_json(suite,{'changed':True})
    with pytest.raises(ValueError,match='calibration_changed'):admission_for(task,c,o,r)
    with pytest.raises(ValueError,match='binding_mismatch'):admission_for(task,c,{**o,'task':'changed'},r)
