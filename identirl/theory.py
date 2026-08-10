"""Closed-form oracle value gaps for validating benchmark labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .envs import Condition


@dataclass(frozen=True)
class ValueGaps:
    reward_gap: float
    observation_gap: float
    memory_gap: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def discounted_horizon(horizon: int, gamma: float) -> float:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if gamma == 1.0:
        return float(horizon)
    return (1.0 - gamma**horizon) / (1.0 - gamma)


def factorial_value_gaps(
    condition: Condition | str,
    *,
    state_zero_probability: float = 0.75,
    horizon: int = 5,
    gamma: float = 1.0,
) -> ValueGaps:
    """Return exact gaps for :class:`DiagnosticPOMDP`."""
    condition = Condition(condition)
    q = state_zero_probability
    scale = discounted_horizon(horizon, gamma)
    observation_gap = scale * (1.0 - q) if condition.observation_corruption else 0.0
    if not condition.reward_corruption:
        reward_gap = 0.0
    elif condition.observation_corruption:
        reward_gap = scale * (2.0 * q - 1.0)
    else:
        reward_gap = scale
    # All useful history is either current or intrinsically absent.
    return ValueGaps(reward_gap, observation_gap, 0.0)


def matched_value_gaps(
    mechanism: str,
    *,
    state_zero_probability: float = 0.75,
    horizon: int = 5,
    gamma: float = 1.0,
) -> ValueGaps:
    scale = discounted_horizon(horizon, gamma)
    if mechanism == "observation":
        return ValueGaps(0.0, scale * (1.0 - state_zero_probability), 0.0)
    if mechanism == "reward":
        return ValueGaps(scale, 0.0, 0.0)
    raise ValueError("mechanism must be 'observation' or 'reward'")


def validate_factorial_labels(
    *, threshold: float = 1e-9, state_zero_probability: float = 0.75
) -> dict[str, tuple[bool, bool]]:
    observed: dict[str, tuple[bool, bool]] = {}
    for condition in Condition:
        gaps = factorial_value_gaps(
            condition, state_zero_probability=state_zero_probability
        )
        observed[condition.value] = (
            gaps.reward_gap > threshold,
            gaps.observation_gap > threshold,
        )
    expected = {
        "neither": (False, False),
        "reward": (True, False),
        "observation": (False, True),
        "both": (True, True),
    }
    if observed != expected:
        raise AssertionError(f"factorial labels invalid: {observed}")
    return observed
