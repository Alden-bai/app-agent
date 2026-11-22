import concurrent.futures
import copy
import pytest

from harness_awm.training.selection import choose_best, init_queue, claim, finish


def report(success, scored=95, planned=100, split='validation'):
    return {'evaluation_split': split, 'evaluation': {
        'planned': planned, 'result_records': planned, 'scored': scored,
        'score_coverage': scored / planned, 'confirmed_successes': success,
        'eval_reward': {'sum': success + 0.3}}}


def test_selection_uses_fixed_denominator_and_rejects_test():
    assert choose_best({'a': report(40, 90), 'b': report(42, 100)}) == 'b'
    assert choose_best({'a': report(40, 99), 'b': report(40, 95)}) == 'a'
    with pytest.raises(ValueError, match='validation'):
        choose_best({'a': report(99, split='test')})
    with pytest.raises(ValueError, match='task counts'):
        choose_best({'a': report(20), 'b': report(20, planned=99)})
    with pytest.raises(ValueError, match='coverage'):
        choose_best({'a': report(20, scored=80)})
    r = report(20)
    r['evaluation']['result_records'] = 99
    with pytest.raises(ValueError, match='incomplete'):
        choose_best({'a': r})


def test_queue_claims_once_and_can_help_other_models(tmp_path):
    queue = tmp_path / 'queue.sqlite'
    init_queue(queue, [('a', str(i), str(tmp_path / f'a{i}')) for i in range(20)] +
                      [('b', str(i), str(tmp_path / f'b{i}')) for i in range(20)])
    def work(owner):
        found = []
        while job := claim(queue, owner, 'b'):
            found.append(job['id'])
            finish(queue, job['id'], owner, 'done')
        return found
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        jobs = sum(pool.map(work, range(8)), [])
    assert len(jobs) == len(set(jobs)) == 40
    assert claim(queue, 9) is None
    with pytest.raises(ValueError, match='ownership'):
        finish(queue, jobs[0], 9, 'done')


def test_microbatch_gradients_match_unsplit_loss():
    torch = pytest.importorskip('torch')
    from harness_awm.training.adaptive import microbatches
    from harness_awm.training.data import ResponseCollator
    torch.manual_seed(7)
    rows = [{'input_ids': [1, 2, 3, 4, 5], 'labels': [-100, -100, 3, 4, 5]},
            {'input_ids': [2, 3, 4], 'labels': [-100, -100, 4]},
            {'input_ids': [3, 4, 5, 6], 'labels': [-100, -100, 5, 6]}]
    inputs = ResponseCollator(0)(rows)
    model = torch.nn.Sequential(torch.nn.Embedding(8, 4), torch.nn.Linear(4, 8))
    other = copy.deepcopy(model)
    denominator = (inputs['labels'][:, 1:] != -100).sum()
    def loss(net, batch):
        logits = net(batch['input_ids'])[:, :-1]
        return torch.nn.functional.cross_entropy(logits.reshape(-1, 8), batch['labels'][:, 1:].reshape(-1),
                                                 reduction='sum', ignore_index=-100) / denominator
    full = loss(model, inputs)
    full.backward()
    chunks = list(microbatches(inputs, 8))
    assert sum(len(x['input_ids']) for x in chunks) == 3
    assert all(x['input_ids'].numel() <= 8 for x in chunks)
    for chunk in chunks:
        loss(other, chunk).backward()
    for a, b in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(a.grad, b.grad)
    with pytest.raises(ValueError, match='complete sample'):
        list(microbatches(inputs, 4))


def test_actual_sft_trainer_split_matches_full_update(tmp_path):
    torch = pytest.importorskip('torch')
    from datasets import Dataset
    from trl import SFTConfig
    from test_training_grpo import tiny_policy
    from harness_awm.training.adaptive import trainer_class
    from harness_awm.training.data import ResponseCollator
    rows = [{'input_ids': [3, 4, 5, 6, 1], 'labels': [-100, -100, 5, 6, 1]},
            {'input_ids': [3, 6, 1], 'labels': [-100, 6, 1]},
            {'input_ids': [4, 5, 6, 1], 'labels': [-100, -100, -100, 1]}]
    models = []
    for budget in (0, 8):
        model, tokenizer = tiny_policy('qwen3.5')
        args = SFTConfig(output_dir=str(tmp_path / str(budget)), use_cpu=True, bf16=False,
            max_steps=1, per_device_train_batch_size=3, learning_rate=.01,
            max_grad_norm=0, optim='sgd', warmup_steps=0, save_strategy='no',
            report_to='none', dataset_kwargs={'skip_prepare_dataset': True},
            train_sampling_strategy='sequential', gradient_checkpointing=False)
        trainer = trainer_class()(model=model, args=args, train_dataset=Dataset.from_list(rows),
            processing_class=tokenizer, data_collator=ResponseCollator(0), microbatch_tokens=budget)
        trainer.train()
        models.append(model)
    for a, b in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)


def test_pipeline_tests_only_frozen_winner(tmp_path, monkeypatch):
    from harness_awm.training import selection
    from harness_awm.io import write_json, read_json
    base = tmp_path / 'base'
    for split in ('validation', 'test'):
        write_json(base / split / 'metrics.json', report(1, split=split))
    root = tmp_path / 'sweep'
    root.mkdir()
    checkpoint = {'path': 'base-model', 'sha256': 'base'}
    plan = {'checkpoint': checkpoint, 'baseline': str(base), 'epochs': 2, 'data': 'data',
            'trials': [{'name': 'a', 'learning_rate': 1e-5}, {'name': 'b', 'learning_rate': 2e-5}],
            'batch_size': 64, 'gradient_accumulation': 1, 'microbatch_tokens': 131072,
            'kernel_deps': 'unused', 'no_gradient_checkpointing': False,
            'selection_rule': 'validation', 'minimum_coverage': .9}
    monkeypatch.setattr(selection, 'checked_plan', lambda root: plan)
    class Tracker:
        url = 'https://example.invalid/run'
        def __init__(self, *args, **kwargs): pass
        def log(self, *args, **kwargs): pass
        def finish(self, *args, **kwargs): pass
    monkeypatch.setattr(selection, 'SwanLabTracker', Tracker)
    monkeypatch.setattr('harness_awm.training.local.checkpoint_identity',
                        lambda path: checkpoint if path == 'base-model' else {'path': str(path), 'sha256': str(path)})
    def train(command, **kwargs):
        from pathlib import Path
        output = Path(command[command.index('--output') + 1])
        for epoch in (1, 2):
            write_json(output / f'checkpoint-{epoch}' / 'trainer_state.json', {'epoch': epoch})
    monkeypatch.setattr(selection.subprocess, 'run', train)
    calls = []
    def evaluate(root, plan, candidates, split, tracker, step):
        calls.append((split, list(candidates)))
        if split == 'validation':
            assert len(candidates) == 4
            return {k: report(50 if k == 'b-epoch1' else 20) for k in candidates}, step + 1
        frozen = read_json(root / 'selection.json')
        assert frozen['selected'] == 'b-epoch1'
        assert list(candidates) == ['b-epoch1']
        assert not frozen['test_used_for_selection']
        return {'b-epoch1': report(30, split='test')}, step + 1
    monkeypatch.setattr(selection, 'run_evaluation', evaluate)
    selection.execute(root)
    assert [c[0] for c in calls] == ['validation', 'test']
    assert read_json(root / 'summary.json')['selected'] == 'b-epoch1'
    assert read_json(root / 'status.json')['stage'] == 'finished'
