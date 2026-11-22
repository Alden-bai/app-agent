import copy
import pytest
from harness_awm.training.rewards import population_advantages
from harness_awm.trajectory_quality import quality_payload


def test_small_defects_are_distinct_without_unit_variance():
    assert population_advantages([1]*7+[.975], std_floor=.1)[-1] == pytest.approx(-.21875, abs=1e-5)
    assert population_advantages([1]*7+[.95], std_floor=.1)[-1] == pytest.approx(-.4375, abs=1e-5)
    for values in ([1,0,0,1], [1]*8, [0]*8):
        assert population_advantages(values, std_floor=.1) == population_advantages(values)


@pytest.mark.parametrize('floor', [-1, True, float('nan'), float('inf')])
def test_bad_floor_rejected(floor):
    with pytest.raises(ValueError):
        population_advantages([1,0], std_floor=floor)


def test_wire_roundtrip_preserves_every_action_and_visible_message():
    messages = {'M1': {'role':'user','content':'visible evidence'},
                'M2': {'role':'assistant','content':'first action'},
                'M3': {'role':'user','content':'later evidence'}}
    context = {'messages': messages, 'turns': [
        {'turn':'0001','input_message_ids':['M1'],
         'latest_observation':{'message_id':'M1', **messages['M1']},'output':'first action'},
        {'turn':'0002','input_message_ids':['M1','M2','M3'],
         'latest_observation':{'message_id':'M3', **messages['M3']},'output':'final action'}]}
    before = copy.deepcopy(context)
    wire = quality_payload(context,'task',{'business_label':'pass','task_status':'confirmed_pass'})
    assert wire['messages'] == messages
    for old, new in zip(context['turns'],wire['turns']):
        assert old['input_message_ids'] == new['input_message_ids']
        assert old['latest_observation']['content'] == wire['messages'][new['latest_observation_id']]['content']
        action = wire['messages'][new['output_message_id']]['content'] if 'output_message_id' in new else new['output']
        assert old['output'] == action
    assert context == before
    assert 'M3' not in wire['turns'][0]['input_message_ids']
