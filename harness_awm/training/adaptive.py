"""Bound padded tokens per forward while preserving the optimizer batch loss."""


def microbatches(inputs, token_budget):
    """Keep every row/target; trim only right padding and split on row boundaries."""
    lengths = inputs['attention_mask'].sum(dim=1).tolist()
    start = 0
    while start < len(lengths):
        end, width = start, 0
        while end < len(lengths):
            next_width = max(width, lengths[end])
            if end > start and next_width * (end + 1 - start) > token_budget:
                break
            if next_width > token_budget:
                raise ValueError('token budget cannot fit a complete sample')
            width = next_width
            end += 1
        yield {k: v[start:end, :width].contiguous() for k, v in inputs.items()}
        start = end


def trainer_class():
    from trl import SFTTrainer

    class TokenBudgetSFTTrainer(SFTTrainer):
        def __init__(self, *args, microbatch_tokens=0, **kwargs):
            self.microbatch_tokens = microbatch_tokens
            super().__init__(*args, **kwargs)

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            if not getattr(self.args, 'strict_resume_weights', False):
                return super()._load_from_checkpoint(resume_from_checkpoint, model=model)
            from .continuation import load_resume_weights
            from harness_awm.io import write_json
            from pathlib import Path
            audit = load_resume_weights(self.model if model is None else model, resume_from_checkpoint)
            write_json(Path(self.args.output_dir) / 'resume_weights.json', audit)

        def training_step(self, model, inputs, num_items_in_batch=None):
            if not self.microbatch_tokens:
                return super().training_step(model, inputs, num_items_in_batch)
            if num_items_in_batch is None:
                raise ValueError('adaptive microbatches require full optimizer-batch target normalization')
            losses = []
            for batch in microbatches(inputs, self.microbatch_tokens):
                # Each loss uses the SAME full accumulated target count. Summed
                # gradients equal one unsplit response-token-mean loss.
                losses.append(super().training_step(model, batch, num_items_in_batch))
            return sum(losses)

    return TokenBudgetSFTTrainer
