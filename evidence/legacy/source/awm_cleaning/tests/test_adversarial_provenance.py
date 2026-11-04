import pytest

from awm_cleaning.provenance import ProvenanceError, resolve
from awm_cleaning.certification import sql_review_valid


@pytest.mark.parametrize('value,task,registry', [
    ({'$public': 42, 'citation': 'trust me'}, 'do the task', {}),
    ({'$public': True, 'citation': 'valid'}, 'do the task', {'valid': 1}),
    ({'$literal': 1, 'quote': '1200000'}, 'Budget is 1200000', {}),
    ({'$literal': True, 'quote': 'pending'}, 'set pending', {}),
    ({'$literal': None, 'quote': 'pending'}, 'set pending', {}),
    ({'$literal': {'hidden_id': 42}, 'quote': 'pending'}, 'set pending', {}),
    ({'$result': True, 'path': []}, 'task', {}),
])
def test_claimed_provenance_cannot_launder_hidden_values(value, task, registry):
    with pytest.raises(ProvenanceError):
        resolve(value, task, [{'id': 3}, {'id': 4}], [], public_contracts=registry)


def test_fully_checked_label_cannot_override_failed_actual_check():
    review = {'status': 'fully_checked',
              'aspects': {k: True for k in ('object', 'join', 'filter', 'aggregate', 'null', 'omission')},
              'checks': [{'name': 'complete_export', 'passed': False}]}
    assert sql_review_valid(review) is False
