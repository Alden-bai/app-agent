"""Restart a completed cosine stage without discarding optimizer moments."""
from functools import partial
import math
from pathlib import Path


def stage_factor(step, *, start, end, warmup):
    relative = step - start
    if relative < 0:
        raise ValueError('checkpoint precedes learning-rate stage')
    if relative < warmup:
        return relative / max(1, warmup)
    progress = min(1., (relative - warmup) / (end - start - warmup))
    return .5 * (1. + math.cos(math.pi * progress))


def continuation_callback(start, end, warmup):
    from transformers import TrainerCallback
    from harness_awm.io import write_json

    if not 0 <= start < end or not 0 <= warmup < end - start:
        raise ValueError('invalid continuation learning-rate stage')

    class ContinuationCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, optimizer=None, lr_scheduler=None, **kwargs):
            if not start <= state.global_step < end or state.max_steps > end:
                raise ValueError('resume state outside configured learning-rate stage')
            if not optimizer.state:
                raise ValueError('continuation requires restored optimizer moments')
            for parameter, moments in optimizer.state.items():
                for key in ('exp_avg', 'exp_avg_sq'):
                    if key not in moments or moments[key].shape != parameter.shape:
                        raise ValueError('optimizer moments do not match model parameters')
            scheduler = getattr(lr_scheduler, 'scheduler', lr_scheduler)
            if not hasattr(scheduler, 'lr_lambdas'):
                raise TypeError('continuation requires a LambdaLR scheduler')
            factor = partial(stage_factor, start=start, end=end, warmup=warmup)
            scheduler.base_lrs = [args.learning_rate] * len(optimizer.param_groups)
            scheduler.lr_lambdas = [factor] * len(optimizer.param_groups)
            scheduler.last_epoch = state.global_step
            rates = [args.learning_rate * factor(state.global_step)] * len(optimizer.param_groups)
            scheduler._last_lr = rates
            for group, rate in zip(optimizer.param_groups, rates):
                group['initial_lr'] = args.learning_rate
                group['lr'] = rate
            write_json(Path(args.output_dir) / 'continuation_state.json', {
                'resumed_global_step': state.global_step, 'stage_start': start,
                'stage_end': end, 'warmup_steps': warmup, 'peak_learning_rate': args.learning_rate,
                'initial_learning_rates': rates, 'optimizer_states': len(optimizer.state),
                'optimizer_step_min': min(float(s['step']) for s in optimizer.state.values()),
                'optimizer_step_max': max(float(s['step']) for s in optimizer.state.values()),
            })

    return ContinuationCallback()


def load_resume_weights(model, checkpoint):
    """Undo Qwen's exported language_model prefix, then require complete weights."""
    from safetensors.torch import load_file
    source = load_file(str(Path(checkpoint) / 'model.safetensors'), device='cpu')
    expected = model.state_dict()
    mapped = {}
    renamed = 0
    for name, tensor in source.items():
        target = name
        if target not in expected and target.startswith('model.language_model.'):
            target = 'model.' + target.removeprefix('model.language_model.')
        if target in mapped:
            raise ValueError(f'duplicate checkpoint key after conversion: {target}')
        mapped[target] = tensor
        renamed += target != name
    # Safetensors omits the tied LM head; restore only a verified storage alias.
    if 'lm_head.weight' not in mapped and 'model.embed_tokens.weight' in mapped:
        if model.get_input_embeddings().weight is model.get_output_embeddings().weight:
            mapped['lm_head.weight'] = mapped['model.embed_tokens.weight']
    model.load_state_dict(mapped, strict=True)
    return {'source_tensor_count': len(source), 'loaded_tensor_count': len(mapped),
            'renamed_tensor_count': renamed, 'strict': True}
