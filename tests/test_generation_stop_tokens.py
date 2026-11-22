import asyncio
from types import SimpleNamespace

import pytest

from harness_awm.training.actor import TrainingActorBackend, stop_token_ids, STOP_POLICY
from harness_awm.io import read_json, write_json


@pytest.mark.parametrize('end_token', [1, 2])
def test_actor_stops_at_chat_or_model_eos_without_role_continuation(tmp_path, end_token):
    torch = pytest.importorskip('torch')

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(is_encoder_decoder=False)
            self.generation_config = SimpleNamespace(eos_token_id=2)
            self.calls = 0

        def prepare_inputs_for_generation(self, ids, **kwargs):
            return {'input_ids': ids}

        def forward(self, input_ids):
            token = [6, end_token, 7][min(self.calls, 2)]
            self.calls += 1
            logits = torch.full((1, 1, 10), -10.)
            logits[0, 0, token] = 10
            return SimpleNamespace(logits=logits, past_key_values=object())

    class Tokenizer:
        eos_token_id = 1
        def apply_chat_template(self, *args, **kwargs): return [3]
        def decode(self, tokens, **kwargs): return ' '.join(str(t) for t in tokens if t not in {1, 2})

    model = Policy()
    actor = TrainingActorBackend(model, Tokenizer(), policy_version='synthetic', max_output_tokens=8, greedy=True)
    result = asyncio.run(actor.generate([{'role': 'user', 'content': 'test'}], tmp_path, 1))
    assert result['finish_reason'] == 'stop'
    assert result['usage']['completion_tokens'] == 2
    assert model.calls == 2 and result['content'] == '6'
    record = read_json(tmp_path / 'policy_turn.json')
    assert record['completion_ids'] == [6, end_token]
    assert record['eos_token_ids'] == [1, 2] and record['stop_policy'] == STOP_POLICY


def test_stop_ids_merge_lists_and_missing_generation_eos():
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[2, 1, 2]))
    assert stop_token_ids(model, SimpleNamespace(eos_token_id=1)) == [1, 2]
    model.generation_config.eos_token_id = None
    assert stop_token_ids(model, SimpleNamespace(eos_token_id=1)) == [1]
    with pytest.raises(ValueError):
        stop_token_ids(model, SimpleNamespace(eos_token_id=None))


def test_corrected_comparison_repeats_base_but_never_selects_it(tmp_path, monkeypatch):
    from harness_awm.training import reevaluate
    from test_sft_selection import report
    candidates = {name: {'path': name, 'sha256': name} for name in ['a1', 'a2', 'b1', 'b2']}
    plan = {'source_run': 'old', 'sft_candidates': candidates, 'base_candidate': {'path': 'base'},
            'selection_rule': 'validation', 'minimum_coverage': .9, 'eval_workers': 20}
    monkeypatch.setattr(reevaluate, 'checked_plan', lambda root: plan)
    monkeypatch.setattr(reevaluate, 'require_drained', lambda source: None)
    class Tracker:
        url = 'https://example.invalid/run'
        def __init__(self, *args, **kwargs): pass
        def log(self, *args, **kwargs): pass
        def finish(self, *args, **kwargs): pass
    monkeypatch.setattr(reevaluate, 'SwanLabTracker', Tracker)
    write_json(tmp_path / 'stopping_smoke.json', {'passed': True, 'stop_policy': STOP_POLICY})
    calls = []
    def evaluate(root, plan, requested, split, tracker, step):
        calls.append(split)
        if split == 'validation':
            assert set(requested) == {'base', *candidates}
            return {name: report(99 if name == 'base' else 50 if name == 'a2' else 20) for name in requested}, step + 1
        assert read_json(root / 'selection.json')['selected'] == 'a2'
        assert set(requested) == {'base', 'a2'}
        return {name: report(30, split='test') for name in requested}, step + 1
    monkeypatch.setattr(reevaluate, 'run_evaluation', evaluate)
    reevaluate.execute(tmp_path)
    assert calls == ['validation', 'test']
    assert read_json(tmp_path / 'summary.json')['old_metrics_reused'] is False
