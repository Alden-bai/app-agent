"""verl loss callback preserving AWM's clipped objective and trajectory weighting."""


def policy_loss(model_output, data, dp_group=None, *, clip=0.2):
    import torch
    from verl.utils.metric import AggregationType, Metric
    from verl.workers.utils.padding import no_padding_2_padding
    from ..grpo import clipped_objective

    current = no_padding_2_padding(model_output["log_probs"], data)
    corrected = "learner_old_log_probs" in data
    keys = ["response_mask", "old_log_probs", "advantages"]
    if corrected:
        keys += ["learner_old_log_probs", "rollout_is_weights"]
    fields = data.select(*keys).to_padded_tensor()
    mask = fields["response_mask"].bool()
    # Padding gets benign finite operands, even if a kernel emits NaN there.
    current = torch.where(mask, current, 0.0)
    old = torch.where(mask, fields["learner_old_log_probs" if corrected else "old_log_probs"], 0.0)
    advantage = torch.where(mask, fields["advantages"], 0.0)
    token_losses = clipped_objective(current, old, advantage, clip=clip)
    if corrected:
        token_losses = token_losses * torch.where(mask, fields["rollout_is_weights"], 0.0).detach()
    loss = (token_losses * mask).sum() * data["dp_size"] / data["global_batch_size"]
    if not torch.isfinite(loss):
        raise ValueError("nonfinite_GRPO_loss")
    return loss, {"awm/pg_loss": Metric(value=loss.detach(), aggregation=AggregationType.SUM)}

