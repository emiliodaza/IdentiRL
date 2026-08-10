"""Short privileged-state and intended-reward diagnostic probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .data import run_identifier


@dataclass(frozen=True)
class InterventionCost:
    probe_episodes: int
    state_queries: int
    reward_queries: int
    episode_price: float = 1.0
    state_query_price: float = 4.0
    reward_query_price: float = 4.0

    @property
    def total(self) -> float:
        return (
            self.episode_price * self.probe_episodes
            + self.state_query_price * self.state_queries
            + self.reward_query_price * self.reward_queries
        )


def intervention_cost(kind: str, probe_episodes: int) -> InterventionCost:
    if kind == "passive":
        return InterventionCost(probe_episodes, 0, 0)
    if kind == "state":
        return InterventionCost(probe_episodes, probe_episodes, 0)
    if kind == "reward":
        return InterventionCost(probe_episodes, 0, probe_episodes)
    if kind == "combined":
        return InterventionCost(probe_episodes, probe_episodes, probe_episodes)
    raise ValueError("kind must be passive, state, reward, or combined")


def collect_probe(
    env_factory: Callable[[], Any],
    *,
    label: str,
    benchmark: str,
    learner: str,
    seed: int,
    kind: str,
    probe_episodes: int,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Collect balanced-action probe episodes without modifying training."""
    if kind not in {"state", "reward", "combined"}:
        raise ValueError("kind must be state, reward, or combined")
    if probe_episodes < 1:
        raise ValueError("probe_episodes must be positive")
    env = env_factory()
    rng = np.random.default_rng(seed + 90_000)
    actions = np.resize(np.asarray([0, 1], dtype=int), probe_episodes)
    rng.shuffle(actions)
    run_id = run_id or run_identifier(benchmark, learner, label, seed)
    rows: list[dict[str, Any]] = []
    for probe_episode in range(probe_episodes):
        observation, _ = env.reset(seed=seed + 100_000 + probe_episode)
        # Randomized overlap is guaranteed by a seeded, balanced action schedule.
        action = int(actions[probe_episode])
        _, proxy_reward, _, _, info = env.step(action)
        private = info["privileged"]
        row: dict[str, Any] = {
            "run_id": run_id,
            "benchmark": benchmark,
            "learner": learner,
            "seed": seed,
            "kind": kind,
            "probe_episode": probe_episode,
            "observation": int(observation),
            "action": action,
            "proxy_reward": float(proxy_reward),
        }
        if kind in {"state", "combined"}:
            row["latent_state"] = int(private["latent_state"])
        if kind in {"reward", "combined"}:
            row["intended_reward"] = float(private["intended_reward"])
        rows.append(row)
    return rows
