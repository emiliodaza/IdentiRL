"""Gymnasium environments with separate proxy and intended rewards.

The public Gymnasium reward is always the proxy reward. Intended rewards and
latent states live under ``info["privileged"]`` so the runner can route them to
a physically separate privileged log.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import gymnasium as gym
from gymnasium import spaces


class Condition(str, Enum):
    NEITHER = "neither"
    REWARD = "reward"
    OBSERVATION = "observation"
    BOTH = "both"

    @property
    def reward_corruption(self) -> bool:
        return self in (Condition.REWARD, Condition.BOTH)

    @property
    def observation_corruption(self) -> bool:
        return self in (Condition.OBSERVATION, Condition.BOTH)


class DiagnosticPOMDP(gym.Env[int, int]):
    """Factorial two-state, two-action benchmark.

    The latent state is fixed for an episode. The intended action equals the
    state. Reward corruption inverts that target; observation corruption maps
    both states to one sentinel. Thus both failures are independently toggled.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        condition: Condition | str = Condition.NEITHER,
        *,
        horizon: int = 5,
        state_zero_probability: float = 0.75,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.condition = Condition(condition)
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if not 0.5 < state_zero_probability < 1.0:
            raise ValueError("state_zero_probability must be in (0.5, 1)")
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")
        self.horizon = horizon
        self.q = state_zero_probability
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(2)
        # 0/1 reveal state; 2 is the deliberately uninformative observation.
        self.observation_space = spaces.Discrete(3)
        self._state = 0
        self._step = 0

    def _observation(self) -> int:
        return 2 if self.condition.observation_corruption else self._state

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed)
        self._state = int(self.np_random.random() >= self.q)
        self._step = 0
        return self._observation(), {"condition": self.condition.value}

    def rewards_for(self, action: int) -> tuple[float, float]:
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}")
        intended = float(action == self._state)
        proxy_target = 1 - self._state if self.condition.reward_corruption else self._state
        return float(action == proxy_target), intended

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        proxy, intended = self.rewards_for(action)
        state_at_decision = self._state
        self._step += 1
        terminated = self._step >= self.horizon
        info = {
            "condition": self.condition.value,
            "privileged": {
                "latent_state": state_at_decision,
                "intended_reward": intended,
            },
        }
        return self._observation(), proxy, terminated, False, info

    def render(self) -> str:
        return (
            f"DiagnosticPOMDP(condition={self.condition.value}, state={self._state}, "
            f"observation={self._observation()}, step={self._step}/{self.horizon})"
        )


class PassiveAmbiguityEnv(gym.Env[int, int]):
    """The exact passive-equivalence pair from the specification.

    ``mechanism='observation'`` has hidden state and aligned reward.
    ``mechanism='reward'`` has a state-independent intended objective and a
    stochastic, misspecified proxy. Conditional on any action, observable
    proxy-reward distributions match exactly between mechanisms.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        mechanism: str,
        *,
        horizon: int = 5,
        state_zero_probability: float = 0.75,
    ) -> None:
        super().__init__()
        if mechanism not in {"observation", "reward"}:
            raise ValueError("mechanism must be 'observation' or 'reward'")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if not 0.5 < state_zero_probability < 1.0:
            raise ValueError("state_zero_probability must be in (0.5, 1)")
        self.mechanism = mechanism
        self.horizon = horizon
        self.q = state_zero_probability
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(1)
        self._state = 0
        self._step = 0

    def _sample_state(self) -> int:
        return int(self.np_random.random() >= self.q)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed)
        self._step = 0
        self._state = self._sample_state()
        return 0, {"mechanism": self.mechanism}

    def rewards_for(self, action: int) -> tuple[float, float]:
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}")
        if self.mechanism == "observation":
            reward = float(action == self._state)
            return reward, reward
        proxy_probability = self.q if action == 0 else 1.0 - self.q
        proxy = float(self.np_random.random() < proxy_probability)
        return proxy, float(action == 1)

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        state_at_decision = self._state
        proxy, intended = self.rewards_for(action)
        self._step += 1
        terminated = self._step >= self.horizon
        info = {
            "mechanism": self.mechanism,
            "privileged": {
                "latent_state": state_at_decision,
                "intended_reward": intended,
            },
        }
        if not terminated:
            self._state = self._sample_state()
        return 0, proxy, terminated, False, info

    def render(self) -> str:
        return f"PassiveAmbiguityEnv(mechanism={self.mechanism}, step={self._step})"
