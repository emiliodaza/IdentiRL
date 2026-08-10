"""Dependency-light tabular Q-learning and tabular PPO learners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ActionSample:
    action: int
    probabilities: np.ndarray
    value: float


class Learner(Protocol):
    def act(self, observation: int) -> ActionSample: ...

    def observe(
        self,
        observation: int,
        action: int,
        reward: float,
        next_observation: int,
        done: bool,
        action_probability: float,
        value: float,
    ) -> float: ...

    def finish(self) -> None: ...


class TabularQLearner:
    def __init__(
        self,
        observation_count: int,
        action_count: int,
        *,
        seed: int,
        learning_rate: float = 0.15,
        gamma: float = 0.95,
        epsilon: float = 0.2,
    ) -> None:
        self.q = np.zeros((observation_count, action_count), dtype=float)
        self.action_count = action_count
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def action_probabilities(self, observation: int) -> np.ndarray:
        values = self.q[observation]
        best = np.flatnonzero(np.isclose(values, values.max()))
        probabilities = np.full(self.action_count, self.epsilon / self.action_count)
        probabilities[best] += (1.0 - self.epsilon) / len(best)
        return probabilities

    def act(self, observation: int) -> ActionSample:
        probabilities = self.action_probabilities(observation)
        action = int(self.rng.choice(self.action_count, p=probabilities))
        return ActionSample(action, probabilities, float(self.q[observation].max()))

    def observe(
        self,
        observation: int,
        action: int,
        reward: float,
        next_observation: int,
        done: bool,
        action_probability: float,
        value: float,
    ) -> float:
        continuation = 0.0 if done else float(self.q[next_observation].max())
        target = reward + self.gamma * continuation
        td_error = target - self.q[observation, action]
        self.q[observation, action] += self.learning_rate * td_error
        return float(td_error)

    def finish(self) -> None:
        return None


@dataclass
class _PPOTransition:
    observation: int
    action: int
    reward: float
    next_observation: int
    done: bool
    old_probability: float


class TabularPPOLearner:
    """Clipped PPO with categorical tabular policy and value function.

    This retains PPO's old-policy importance ratio and clipped surrogate while
    avoiding a heavyweight neural-network dependency in the tiny benchmark.
    """

    def __init__(
        self,
        observation_count: int,
        action_count: int,
        *,
        seed: int,
        gamma: float = 0.95,
        policy_learning_rate: float = 0.08,
        value_learning_rate: float = 0.12,
        clip_ratio: float = 0.2,
        update_epochs: int = 6,
        rollout_steps: int = 32,
    ) -> None:
        self.logits = np.zeros((observation_count, action_count), dtype=float)
        self.values = np.zeros(observation_count, dtype=float)
        self.action_count = action_count
        self.gamma = gamma
        self.policy_learning_rate = policy_learning_rate
        self.value_learning_rate = value_learning_rate
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.rollout_steps = rollout_steps
        self.rng = np.random.default_rng(seed)
        self._rollout: list[_PPOTransition] = []

    def action_probabilities(self, observation: int) -> np.ndarray:
        logits = self.logits[observation] - self.logits[observation].max()
        exp = np.exp(logits)
        return exp / exp.sum()

    def act(self, observation: int) -> ActionSample:
        probabilities = self.action_probabilities(observation)
        action = int(self.rng.choice(self.action_count, p=probabilities))
        return ActionSample(action, probabilities, float(self.values[observation]))

    def observe(
        self,
        observation: int,
        action: int,
        reward: float,
        next_observation: int,
        done: bool,
        action_probability: float,
        value: float,
    ) -> float:
        next_value = 0.0 if done else self.values[next_observation]
        td_error = reward + self.gamma * next_value - value
        self._rollout.append(
            _PPOTransition(
                observation,
                action,
                reward,
                next_observation,
                done,
                max(float(action_probability), 1e-12),
            )
        )
        if len(self._rollout) >= self.rollout_steps:
            self._update()
        return float(td_error)

    def _returns_and_advantages(self) -> tuple[np.ndarray, np.ndarray]:
        returns = np.zeros(len(self._rollout), dtype=float)
        running = 0.0
        for index in range(len(self._rollout) - 1, -1, -1):
            transition = self._rollout[index]
            running = transition.reward + self.gamma * running * (not transition.done)
            returns[index] = running
        baseline = np.array(
            [self.values[item.observation] for item in self._rollout], dtype=float
        )
        advantages = returns - baseline
        if len(advantages) > 1 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def _update(self) -> None:
        if not self._rollout:
            return
        returns, advantages = self._returns_and_advantages()
        for _ in range(self.update_epochs):
            order = self.rng.permutation(len(self._rollout))
            for index in order:
                transition = self._rollout[int(index)]
                observation = transition.observation
                probabilities = self.action_probabilities(observation)
                new_probability = max(probabilities[transition.action], 1e-12)
                ratio = new_probability / transition.old_probability
                advantage = advantages[index]
                clipped_ratio = np.clip(
                    ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                )
                unclipped_objective = ratio * advantage
                clipped_objective = clipped_ratio * advantage
                # The clipped branch is locally constant outside the interval.
                if unclipped_objective <= clipped_objective or np.isclose(
                    ratio, clipped_ratio
                ):
                    score = -probabilities
                    score[transition.action] += 1.0
                    self.logits[observation] += (
                        self.policy_learning_rate * ratio * advantage * score
                    )
                value_error = returns[index] - self.values[observation]
                self.values[observation] += self.value_learning_rate * value_error
        self._rollout.clear()

    def finish(self) -> None:
        self._update()


def make_learner(
    name: str, observation_count: int, action_count: int, *, seed: int
) -> Learner:
    if name == "tabular":
        return TabularQLearner(observation_count, action_count, seed=seed)
    if name == "ppo":
        return TabularPPOLearner(observation_count, action_count, seed=seed)
    raise ValueError(f"unknown learner {name!r}; choose 'tabular' or 'ppo'")
