"""Independently reload the final GRPO export and verify update evidence."""
import argparse
import json
from pathlib import Path

from harness_awm.io import read_json, write_json
from harness_awm.training.local import checkpoint_identity, load_checkpoint


def main():
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--inspect-incomplete', action='store_true',
                   help='check a saved checkpoint even when the update target was missed; never accept that run')
    args = p.parse_args()
    plan, summary = [read_json(args.run / name) for name in ('plan.json', 'summary.json')]
    target = plan.get('target_updates')
    target_met = (len(summary['groups']) >= plan['groups'] if target is None
                  else summary['optimizer_steps'] >= target)
    if summary['failed'] or (not target_met and not args.inspect_incomplete):
        raise ValueError('configured training target was not met')
    updates = [group for group in summary['groups'] if group['status'] == 'updated']
    assert len(updates) == summary['optimizer_steps']
    for step, group in enumerate(updates, 1):
        assert group['optimizer_step'] == step
        assert all(row['finite'] and row['changed_samples'] > 0
                   and row['optimizer_state_steps'] == [step] for row in group['parameters'])
    checkpoint_name = (f"group_step_{len(summary['groups'])}" if plan.get('save_unit') == 'groups'
                       else f"global_step_{summary['optimizer_steps']}")
    checkpoint = Path(plan['checkpoint_dir']) / checkpoint_name / 'actor'
    if plan.get('save_unit') == 'groups':
        saved = read_json(checkpoint.parent / 'training_state.json')
        assert saved['groups'] == len(summary['groups'])
        assert saved['optimizer_steps'] == summary['optimizer_steps']
    required = ('model_world_size_1_rank_0.pt', 'optim_world_size_1_rank_0.pt', 'extra_state_world_size_1_rank_0.pt')
    for name in required:
        if not (checkpoint / name).is_file():
            raise ValueError('missing full training state: ' + name)
    model_path = checkpoint / 'huggingface'
    identity = checkpoint_identity(model_path)
    model, tokenizer = load_checkpoint(model_path, dtype='float32')
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all():
            raise ValueError('nonfinite exported parameter')
    # Compare the reloaded weights with the final training worker's same fixed samples.
    import hashlib
    pieces = []
    for parameter in model.parameters():
        flat = parameter.detach().reshape(-1)
        pieces.append(flat[::max(1, flat.numel() // 1024)][:1024].float().cpu())
    sample_hash = hashlib.sha256(torch.cat(pieces).numpy().tobytes()).hexdigest()
    expected = updates[-1]['parameters'] if updates else read_json(args.run / 'initial_parameters.json')
    expected_hash = expected[0]['sample_sha256']
    if sample_hash != expected_hash:
        raise ValueError('HF export does not match final FP32 training weights')
    messages = [{'role': 'user', 'content': 'Return only this JSON: {"ok":true}'}]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                       enable_thinking=False, return_tensors='pt')
    if hasattr(ids, 'input_ids'):
        ids = ids.input_ids
    ids = ids.to('cuda')
    with torch.no_grad():
        generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
            max_new_tokens=64, eos_token_id=plan['eos_ids'], pad_token_id=tokenizer.pad_token_id)
    completion = generated[0, ids.shape[-1]:].tolist()
    if not completion or completion[-1] not in plan['eos_ids']:
        raise ValueError('reloaded model did not produce an EOS-terminated response')
    report = {'passed': target_met, 'checkpoint_reload_passed': True, 'target_met': target_met,
              'optimizer_steps': summary['optimizer_steps'], 'checkpoint': identity,
              'full_training_state': list(required), 'parameter_sample_sha256': sample_hash,
              'completion_ids': completion, 'completion': tokenizer.decode(completion, skip_special_tokens=True),
              'eos_ids': plan['eos_ids'], 'judge_requests': 0, 'effectiveness_claim': False}
    destination = args.run / ('acceptance.json' if target_met else 'checkpoint-inspection.json')
    if destination.exists():
        raise ValueError('acceptance already exists')
    write_json(destination, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
