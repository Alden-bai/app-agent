"""Diagnostic rewards must stay bounded and not masquerade as business rewards."""
import pytest

from harness_awm.training.trl_smoke import diagnostic_reward


@pytest.mark.parametrize("completion", ["not JSON", "[]", "null", '{"name":"tool"}', '{"type":7}'])
def test_invalid_action_has_zero_diagnostic_reward(completion):
    assert diagnostic_reward(completion, '{"type":"search_tools","query":"profile"}') == 0


def test_reference_reward_ignores_json_key_order_and_whitespace():
    assert diagnostic_reward('{"query": "profile", "type": "search_tools"}',
                             '{"type":"search_tools","query":"profile"}') == 1


def test_reference_similarity_is_not_a_binary_success_label():
    reference = '{"type":"search_tools","query":"profile"}'
    value = diagnostic_reward('{"type":"search_tools","query":"user profile"}', reference)
    assert 0.5 < value < 1
    assert diagnostic_reward('{"type":"finish","content":"profile"}', reference) < value
