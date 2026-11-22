"""Critical loss-mask, frozen-source, checkpoint and reward boundary contracts."""
import json
from pathlib import Path

import pytest

from harness_awm.training.data import encode_turn, ResponseCollator, resolve_ref, verify_split, sha
from harness_awm.training.rewards import admitted_group, InvalidRewardGroup, population_advantages
from harness_awm.export import tokenize_turn
from harness_awm.training.doctor import supports_capability
from harness_awm.reward_values import VERSION


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}</end>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return list(text.encode())


def test_gpu_wheel_capability_includes_compatible_minor_architecture():
    assert supports_capability((8, 9), ["sm_80", "sm_86", "sm_90"])
    assert supports_capability((12, 0), ["sm_86", "sm_120"])
    assert not supports_capability((12, 0), ["sm_86", "sm_90"])
    assert not supports_capability((8, 6), ["sm_89"])


def test_only_current_response_and_padding_receive_correct_labels():
    history = [{"role": "user", "content": "task"}, {"role": "assistant", "content": "excluded old action"},
               {"role": "user", "content": "tool result"}]
    a = encode_turn(CharacterTokenizer(), history, "current action")
    b = encode_turn(CharacterTokenizer(), [{"role": "user", "content": "task"}], "done")
    batch = ResponseCollator(0)([a, b])
    supervised = bytes(t for t in a["labels"] if t != -100).decode()
    assert supervised == "current action</end>"
    assert "excluded old action" in bytes(a["input_ids"]).decode()
    assert batch["labels"][1, len(b["labels"]):].eq(-100).all()
    assert batch["attention_mask"][1, len(b["labels"]):].eq(0).all()
    with pytest.raises(ValueError):
        ResponseCollator(0)([{"input_ids": [1, 2], "labels": [-100, -100]}])


def test_legacy_export_supports_transformers5_mapping():
    class MappingTokenizer(CharacterTokenizer):
        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": super().apply_chat_template(*args, **kwargs)}
    messages = [{"role": "user", "content": "task"}]
    assert tokenize_turn(MappingTokenizer(), messages, "done") == tokenize_turn(CharacterTokenizer(), messages, "done")


def test_template_prefix_change_fails():
    class BadTemplate(CharacterTokenizer):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
            return [1, 2] if add_generation_prompt else [3, 4, 5]
    with pytest.raises(ValueError, match="prefix"):
        encode_turn(BadTemplate(), [{"role": "user", "content": "task"}], "done")


def test_relocation_and_escape(tmp_path):
    assert resolve_ref('/old/repo/runs/a.json', '/old/repo', tmp_path) == tmp_path / 'runs/a.json'
    with pytest.raises(ValueError):
        resolve_ref('/old/other/secret', '/old/repo', tmp_path)
    with pytest.raises(ValueError):
        resolve_ref('/old/repo/../secret', '/old/repo', tmp_path)


def test_split_tampering_and_cross_split_overlap(tmp_path):
    names = ['sft_trajectories', 'grpo_train', 'validation', 'test']
    def write(overlap=False):
        files = {}
        for i, name in enumerate(names):
            path = tmp_path / (name + '.jsonl')
            path.write_text(json.dumps({'task_id': '0' if overlap else str(i),
                                        'split': 'sft_train' if i == 0 else name}) + '\n')
            files[path.name] = sha(path)
        (tmp_path / 'manifest.json').write_text(json.dumps({'schema_version': 'awm-training-split-v1', 'files': files}))
    write()
    assert verify_split(tmp_path)[1]['test'] == 1
    (tmp_path / 'test.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_split(tmp_path)
    write(overlap=True)
    with pytest.raises(ValueError, match='overlap'):
        verify_split(tmp_path)


def test_null_or_unadmitted_reward_holds_entire_group():
    valid = {'reward_valid': True, 'test_only': False, 'execution_integrity': 'valid',
             'terminal_state_status': 'trusted'}
    group = [{**valid, 'reward': r} for r in [1, 0, 1, 0]]
    assert admitted_group(group) == [1., 0., 1., 0.]
    assert population_advantages([1, 0, 1, 0]) == pytest.approx([1, -1, 1, -1], abs=3e-6)
    assert population_advantages([0, 0, 0, 0]) == [0, 0, 0, 0]
    for change in [{'reward': None}, {'reward': float('nan')}, {'reward_valid': False},
                   {'test_only': True}, {'terminal_state_status': 'unknown'}, {'reward': True},
                   {'reward': 0.5}, {'reward': 0.3}]:
        with pytest.raises(InvalidRewardGroup):
            admitted_group([*group[:3], {**group[3], **change}])


def test_discount_group_uses_fractional_rewards_without_rounding():
    valid = {'reward_valid': True, 'test_only': False, 'execution_integrity': 'valid',
             'terminal_state_status': 'trusted', 'reward_policy': VERSION,
             'business_label': 'unknown', 'task_status': 'provisional_pass',
             'evaluation_valid': True, 'evaluation_status': 'valid'}
    group = [{**valid, 'reward': r} for r in [0.3, 0, 0, 0]]
    assert admitted_group(group) == [0.3, 0., 0., 0.]
    advantages = population_advantages(admitted_group(group))
    assert advantages == pytest.approx([1.732037, -0.577346, -0.577346, -0.577346], abs=1e-5)
    for change in [{'reward_policy': 'judge-v2-terminal-binary-v1'}, {'task_status': 'confirmed_pass'},
                   {'business_label': 'fail'}, {'evaluation_status': 'evaluation_error'}, {'evaluation_valid': False}]:
        with pytest.raises(InvalidRewardGroup):
            admitted_group([{**group[0], **change}, *group[1:]])
