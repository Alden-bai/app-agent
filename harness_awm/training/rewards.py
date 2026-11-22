"""Terminal rewards and explicit training-only null penalties; Judge labels stay intact."""
import math

from ..reward_values import COMPOSITE_VERSION, PROVISIONAL_REWARD, TERMINAL_REWARDS, VERSION


class InvalidRewardGroup(ValueError):
    """No usable group remains under the selected training reward policy."""


def null_penalty_eligible(result, null_reward_policy):
    """Training-only fallback; never rewrite the sealed Judge assessment."""
    if null_reward_policy not in {"exclude", "zero"}:
        raise ValueError("unknown_null_reward_policy")
    return (null_reward_policy == "zero" and "reward" in result and result["reward"] is None
            and result.get("test_only") is False and result.get("termination_reason") != "cancelled")


def admitted_samples(results, group_size, *, null_reward_policy="exclude"):
    """Select usable candidates before computing statistics or constructing losses."""
    if len(results) != group_size or group_size < 2:
        raise InvalidRewardGroup("incomplete candidate group")
    indices, rewards, excluded = [], [], []
    for index, result in enumerate(results):
        try:
            reward = 0.0 if null_penalty_eligible(result, null_reward_policy) else admitted_reward(result)
        except InvalidRewardGroup as exc:
            excluded.append({"index": index, "reason": str(exc)})
        else:
            indices.append(index)
            rewards.append(reward)
    policies = {results[i].get("reward_policy") for i in indices
                if not null_penalty_eligible(results[i], null_reward_policy)}
    if len(policies) > 1:
        raise InvalidRewardGroup("mixed_reward_policies")
    return indices, rewards, excluded


def admitted_reward(result):
    value = result.get("reward")
    if (result.get("reward_valid") is not True or result.get("test_only") is not False
            or result.get("execution_integrity") != "valid"
            or result.get("terminal_state_status") != "trusted"
            or type(value) not in (int, float) or not math.isfinite(value)
            or (result.get("reward_policy") != COMPOSITE_VERSION and value not in TERMINAL_REWARDS.values())):
        raise InvalidRewardGroup("unadmitted_or_unsettled_reward:" + str(
            result.get("reward_exclusion") or result.get("evaluation_status")))
    if result.get("reward_policy") == COMPOSITE_VERSION:
        from ..composite_reward import reward_from_components
        try:
            expected = reward_from_components(result.get("reward_components"),
                result.get("business_label"), result.get("task_status"))
            if (value != expected or result.get("evaluation_valid") is not True
                    or result.get("evaluation_status") != "valid"):
                raise ValueError("inconsistent_composite_reward")
        except (ValueError, TypeError, KeyError) as exc:
            raise InvalidRewardGroup("invalid composite reward: " + str(exc)) from exc
    elif value == PROVISIONAL_REWARD and (
        result.get("reward_policy") != VERSION
        or result.get("business_label") != "unknown"
        or result.get("task_status") != "provisional_pass"
        or result.get("evaluation_valid") is not True
        or result.get("evaluation_status") != "valid"
    ):
        raise InvalidRewardGroup("discount requires a valid provisional pass under the current policy")
    return float(value)


def admitted_group(results, group_size=4):
    if len(results) != group_size or group_size < 2:
        raise InvalidRewardGroup("incomplete candidate group")
    policies = {r.get("reward_policy") for r in results}
    if COMPOSITE_VERSION in policies and len(policies) != 1:
        raise InvalidRewardGroup("mixed_reward_policies")
    return [admitted_reward(result) for result in results]


def population_advantages(rewards, epsilon=1e-6, std_floor=0.0):
    """Design-v1 reference, not a claim that TRL's default uses this estimator."""
    if len(rewards) < 2 or not math.isfinite(epsilon) or epsilon <= 0 or any(not math.isfinite(x) for x in rewards):
        raise ValueError("invalid advantage inputs")
    if type(std_floor) not in (int, float) or not math.isfinite(std_floor) or std_floor < 0:
        raise ValueError("invalid advantage standard deviation floor")
    if all(r == rewards[0] for r in rewards):
        return [0.0] * len(rewards)
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    return [(r - mean) / (max(std, std_floor) + epsilon) for r in rewards]
