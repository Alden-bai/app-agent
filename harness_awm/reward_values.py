"""Shared values for the versioned terminal reward contract."""

VERSION = "judge-v2-terminal-discount-v1"
COMPOSITE_VERSION = "judge-v2-cst-v1"
LAMBDA_S = 0.05
LAMBDA_T = 0.15
PROVISIONAL_REWARD = 0.3
TERMINAL_REWARDS = {
    ("pass", "confirmed_pass"): 1,
    ("unknown", "provisional_pass"): PROVISIONAL_REWARD,
    ("fail", "failed"): 0,
}


def validate_reward_value(value, policy=None):
    # Reject bools and strings before Pydantic can coerce them into numbers.
    if value is not None and (
        type(value) not in (int, float) or not (
            0 <= value <= 1 if policy == COMPOSITE_VERSION else value in TERMINAL_REWARDS.values())
    ):
        raise ValueError("reward_must_be_0_0.3_1_or_null")
    return value
