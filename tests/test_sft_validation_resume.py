from pathlib import Path
import pytest
from harness_awm.io import write_json,read_json
from scripts.prepare_sft_validation_resume import audit_source_additions


def test_resume_source_audit_allows_only_new_training_modules():
    old={'harness_awm/training/actor.py':'original'}
    assert audit_source_additions(old,{**old,'harness_awm/training/verl/sft.py':'new'})==['harness_awm/training/verl/sft.py']
    with pytest.raises(ValueError):audit_source_additions(old,{**old,'harness_awm/training/actor.py':'changed'})
    with pytest.raises(ValueError):audit_source_additions(old,{**old,'harness_awm/new_evaluator.py':'new'})
    with pytest.raises(ValueError):audit_source_additions(old,{})


@pytest.mark.parametrize('ambiguous',[False,True])
def test_resume_repairs_before_selection_and_never_starts_test(tmp_path,monkeypatch,ambiguous):
    from scripts import resume_sft_validation as driver
    root=tmp_path/'repairs';root.mkdir();control=tmp_path/'control';control.mkdir()
    monkeypatch.setattr(driver,'REPAIRS',root);monkeypatch.setattr(driver,'CONTROL',control)
    monkeypatch.setattr(driver,'ARCHIVE',tmp_path/'archive');monkeypatch.setattr(driver,'SOURCE',tmp_path/'source')
    plan={'sft_candidates':['a','b'],'minimum_coverage':.9,'selection_rule':'validation only'}
    write_json(root/'policy.json',{'files':{}})
    monkeypatch.setattr(driver.selection,'checked_plan',lambda _:plan)
    events=[]
    class Tracker:
        url='offline-test'
        def __init__(self,*args,**kwargs):pass
        def log(self,*args,**kwargs):pass
        def finish(self,**kwargs):pass
    monkeypatch.setattr(driver,'SwanLabTracker',Tracker)
    monkeypatch.setattr(driver.recovery,'monitor_repairs',lambda source,split,*args,**kwargs: events.append(('repair',split)) or [])
    monkeypatch.setattr(driver.recovery,'prepare_validation',lambda *args:{'base':{},'a':{},'b':{}})
    monkeypatch.setattr(driver,'run_stage',lambda root,plan,split,*args:events.append(('model',split)))
    monkeypatch.setattr(driver.recovery,'overlay_repairs',lambda *args:None)
    monkeypatch.setattr(driver.recovery,'collect_repairs',lambda:[])
    monkeypatch.setattr(driver.recovery,'emit',lambda *args:None)
    def reports(*args):
        return {name:{'evaluation_split':'validation','evaluation':{'planned':100,'result_records':100,
            'confirmed_successes':succeeded,'scored':scored,'score_coverage':scored/100,'eval_reward':{'sum':succeeded}}}
            for name,succeeded,scored in [('base',0,100),('a',50,100),('b',45,90 if ambiguous else 100)]}
    monkeypatch.setattr(driver.recovery,'final_reports',reports)
    if ambiguous:
        with pytest.raises(RuntimeError,match='could change winner'):driver.judge()
        assert not (root/'selection.json').exists()
    else:
        driver.judge()
        assert read_json(root/'selection.json')['selected']=='a'
        assert read_json(root/'status.json')['test_started'] is False
    assert events==[('repair','validation'),('model','validation'),('repair','validation')]
