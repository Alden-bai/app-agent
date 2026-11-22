"""Framework-independent trajectory validation and verl batch construction."""
import math
from pathlib import Path

from ...io import digest, read_json, verify_seal, write_json
from ...judge_inputs import sources_from_manifest
from ...reward_policy import admission_for
from ..rewards import InvalidRewardGroup, admitted_samples, null_penalty_eligible, population_advantages


def read_group(folder, task, config, policy_version, group_size, tokenizer, *, null_reward_policy="exclude"):
    """Recheck admission and sealed evidence immediately before a policy update."""
    folder = Path(folder)
    manifest = read_json(folder / "group.json")
    if (manifest["policy_version"] != policy_version or manifest["config_hash"] != digest(config.model_dump())
            or manifest["task"] != task.model_dump() or len(manifest["candidates"]) != group_size):
        raise ValueError("stale_or_changed_verl_group")
    try:
        admission = admission_for(task, config, **sources_from_manifest(task, config.judge_inputs_manifest))
    except (ValueError, OSError) as exc:
        raise InvalidRewardGroup("reward_admission_changed_or_missing") from exc
    admission_id = digest(admission.model_dump())
    results, trajectories = [], []
    for index, candidate in enumerate(manifest["candidates"]):
        if candidate["path"] != f"candidate-{index:03d}":
            raise ValueError("invalid_candidate_path")
        episode = folder / candidate["path"]
        sealed = verify_seal(episode)
        result = read_json(episode / "episode_result.json")
        if digest(result) != candidate["result_hash"] or result["task_instance_id"] != task.task_instance_id:
            raise ValueError("changed_episode_result")
        if result.get("reward_admission_id") != admission_id:
            if result.get("reward_admission_id") is not None:
                raise InvalidRewardGroup("reward_admission_changed_or_missing")
            # A failed episode may never have reached the Judge that attaches this ID.
            # Task admission was checked above; keep its original failure reason for the penalty log.
            if not null_penalty_eligible(result, null_reward_policy):
                result = {**result, "reward_valid": False, "reward_exclusion":
                          result.get("reward_exclusion") or "reward_admission_changed_or_missing"}
        # Explicit null-as-zero training keeps sampled tokens, but not a fabricated Judge verdict.
        results.append(result)
        trajectories.append([])
        if result.get("reward_valid") is not True and not null_penalty_eligible(result, null_reward_policy):
            continue
        turns = []
        for path in sorted((episode / "model_turns").glob("*/policy_turn.json")):
            if str(path.relative_to(episode)) not in sealed:
                raise ValueError("unsealed_policy_turn")
            turn = read_json(path)
            validate_turn(turn, policy_version)
            if (turn["messages_hash"] != digest(read_json(path.parent / "input.json")["messages"])
                    or tokenizer.decode(turn["completion_ids"], skip_special_tokens=True)
                    != read_json(path.parent / "output.json")["content"]):
                raise ValueError("changed_turn_context_or_output")
            turns.append(turn)
        if not turns or len(turns) != len(list((episode / "model_turns").glob("*/output.json"))):
            result.update(reward_valid=False, reward_exclusion="missing_policy_turns")
        trajectories[-1] = turns
    indices, rewards, excluded = admitted_samples(results, group_size, null_reward_policy=null_reward_policy)
    # There is no gradient without a complete, verified sampled action record.
    usable = [(i, reward) for i, reward in zip(indices, rewards)
              if trajectories[i] and results[i].get("reward_exclusion") != "missing_policy_turns"]
    excluded.extend({"index": i, "reason": "missing_policy_turns"}
                    for i in indices if not trajectories[i] or results[i].get("reward_exclusion") == "missing_policy_turns")
    indices, rewards = [i for i, _ in usable], [reward for _, reward in usable]
    penalties = [{"index": i, "original_reward": None, "training_reward": 0.0,
                  "reason": results[i].get("reward_exclusion") or results[i].get("evaluation_status")
                            or results[i].get("termination_reason")}
                 for i in indices if null_penalty_eligible(results[i], null_reward_policy)]
    write_json(folder / "reward_selection.json", {"included_indices": indices, "excluded": excluded,
        "original_group_size": group_size, "effective_group_size": len(indices),
        "null_reward_policy": null_reward_policy, "null_penalties": penalties})
    if len(indices) < 2:
        raise InvalidRewardGroup("fewer_than_two_admitted_candidates")
    return [trajectories[i] for i in indices], rewards, population_advantages(rewards, std_floor=0.10)


def validate_turn(turn, policy_version):
    prompt, completion, old = turn["prompt_ids"], turn["completion_ids"], turn["old_logprobs"]
    if (turn["policy_version"] != policy_version or turn["test_only"] is not False
            or not prompt or not completion or len(completion) != len(old)
            or any(type(t) is not int or t < 0 for t in prompt + completion)
            or any(type(x) not in (int, float) or not math.isfinite(x) or x > 0 for x in old)
            or turn["action_mask"] != [0] * len(prompt) + [1] * len(completion)
            or turn["sampling"] != {"temperature": 1.0, "top_k": 0, "top_p": 1.0}):
        raise ValueError("invalid_or_stale_policy_turn")


def turn_batch(trajectories, advantages, pad_token_id, world_size=1):
    """One real sampled context per row; no concatenation across context compaction.

With seq-mean-token-sum aggregation, B/(K*T_i) on each advantage
gives exactly (1/K) sum_i (1/T_i) sum_t PPO(token_it).
Rows padded to a DP multiple have zero loss masks, including their EOS.
"""
    import torch
    if (not trajectories or len(trajectories) != len(advantages) or world_size < 1
            or any(not math.isfinite(a) for a in advantages)):
        raise ValueError("invalid_turn_batch")
    rows = []
    for turns, advantage in zip(trajectories, advantages):
        if not turns:
            raise ValueError("empty_trajectory")
        total = sum(len(t["completion_ids"]) for t in turns)
        if not total:
            raise ValueError("empty_trajectory")
        for turn in turns:
            rows.append((turn, advantage / (len(trajectories) * total), True))
    count = math.ceil(len(rows) / world_size) * world_size
    # Duplicate a valid context only as a fully masked DP padding row.
    rows += [(rows[0][0], 0.0, False)] * (count - len(rows))
    plen = max(len(t["prompt_ids"]) for t, _, _ in rows)
    rlen = max(len(t["completion_ids"]) for t, _, _ in rows)
    prompts = torch.full((count, plen), pad_token_id, dtype=torch.long)
    responses = torch.full((count, rlen), pad_token_id, dtype=torch.long)
    attention = torch.zeros((count, plen + rlen), dtype=torch.long)
    mask = torch.zeros((count, rlen), dtype=torch.long)
    old = torch.zeros((count, rlen), dtype=torch.float32)
    adv = torch.zeros_like(old)
    for i, (turn, weight, real) in enumerate(rows):
        p, c = turn["prompt_ids"], turn["completion_ids"]
        prompts[i, -len(p):] = torch.tensor(p)
        responses[i, :len(c)] = torch.tensor(c)
        attention[i, plen-len(p):plen+len(c)] = 1
        if real:
            mask[i, :len(c)] = 1
            old[i, :len(c)] = torch.tensor(turn["old_logprobs"])
            adv[i, :len(c)] = weight * count
    return {"prompts": prompts, "responses": responses,
            "input_ids": torch.cat([prompts, responses], dim=-1), "attention_mask": attention,
            "position_ids": (attention.cumsum(-1) - 1).clamp_min(0),
            "response_mask": mask, "old_log_probs": old, "advantages": adv}


def check_logprobs(batch, recomputed, atol=5e-3, rtol=1e-3):
    import torch
    mask = batch["response_mask"].bool()
    old, current = batch["old_log_probs"][mask], recomputed[mask]
    if not torch.isfinite(current).all() or not torch.allclose(current, old, atol=atol, rtol=rtol):
        delta = float((current - old).abs().max())
        raise ValueError(f"sampling_policy_logprob_mismatch: max_abs_delta={delta}")
    return float((current - old).abs().max())


def rollout_correction(batch, recomputed, max_ratio=2.0):
    """Explicit token IS for same-version inference/learner numerical differences.

    Preserve sampled old_log_probs. PPO is anchored at the frozen learner policy,
    and detached pi_learner/pi_sample weights correct the sampling distribution.
    This is token-level (biased, low-variance) correction, not exact trajectory IS.
    A probability ratio outside [1/max_ratio, max_ratio] still aborts the update.
    """
    import torch
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_weights

    mask = batch['response_mask'].bool()
    sampled = batch['old_log_probs']
    if (not math.isfinite(max_ratio) or max_ratio <= 1 or not mask.any()
            or recomputed.shape != sampled.shape
            or not torch.isfinite(recomputed[mask]).all()
            or not torch.isfinite(sampled[mask]).all()):
        raise ValueError('invalid_rollout_correction_inputs')
    learner = torch.where(mask, recomputed.detach(), 0.0)
    ratio_log = torch.where(mask, learner - sampled, 0.0)
    delta = float(ratio_log[mask].abs().max())
    if delta > math.log(max_ratio):
        raise ValueError(f'rollout_correction_ratio_out_of_bounds: max_abs_delta={delta}')
    weights, metrics = compute_rollout_correction_weights(
        ratio_log, mask.to(sampled.dtype), rollout_is='token',
        rollout_is_threshold=max_ratio, rollout_is_batch_normalize=False)
    return {'learner_old_log_probs': learner, 'rollout_is_weights': weights}, {
        'mode': 'token-is-v1', 'max_logprob_abs_delta': delta,
        'mean_logprob_abs_delta': float(ratio_log[mask].abs().mean()),
        'ratio_min': float(ratio_log[mask].exp().min()),
        'ratio_max': float(ratio_log[mask].exp().max()), 'metrics': metrics}
