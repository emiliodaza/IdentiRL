"""Training, schema validation, and CSV persistence.

The passive schema is allow-listed. This makes it impossible for privileged
fields to enter a passive file accidentally through an ``info`` dictionary.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .learners import make_learner


PASSIVE_FIELDS = (
    "run_id",
    "benchmark",
    "learner",
    "seed",
    "global_step",
    "episode",
    "episode_step",
    "observation",
    "action",
    "proxy_reward",
    "action_probability_0",
    "action_probability_1",
    "policy_entropy",
    "value_estimate",
    "td_error",
    "episode_end",
)

PRIVILEGED_FIELDS = (
    "run_id",
    "benchmark",
    "label",
    "learner",
    "seed",
    "global_step",
    "latent_state",
    "intended_reward",
)


def policy_entropy(probabilities: np.ndarray) -> float:
    safe = np.clip(probabilities, 1e-12, 1.0)
    return float(-(safe * np.log(safe)).sum())


def run_identifier(benchmark: str, learner: str, label: str, seed: int) -> str:
    """Return a stable opaque identifier that does not reveal the class name."""
    payload = f"identirl-v1|{benchmark}|{learner}|{label}|{seed}".encode()
    return f"run-{hashlib.sha256(payload).hexdigest()[:16]}"


def assert_passive_schema(rows: Iterable[Mapping[str, Any]]) -> None:
    forbidden = {"latent_state", "state", "intended_reward", "true_reward"}
    for index, row in enumerate(rows):
        extras = set(row) - set(PASSIVE_FIELDS)
        leaked = set(row) & forbidden
        if extras or leaked:
            raise ValueError(
                f"passive row {index} violates schema; extras={sorted(extras)}, "
                f"privileged={sorted(leaked)}"
            )


def train_run(
    env: Any,
    *,
    label: str,
    benchmark: str,
    learner_name: str,
    seed: int,
    total_steps: int,
    run_id: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    learner = make_learner(
        learner_name, env.observation_space.n, env.action_space.n, seed=seed + 10_000
    )
    run_id = run_id or run_identifier(benchmark, learner_name, label, seed)
    passive: list[dict[str, Any]] = []
    privileged: list[dict[str, Any]] = []
    observation, _ = env.reset(seed=seed)
    episode = 0
    episode_step = 0
    for global_step in range(total_steps):
        sample = learner.act(int(observation))
        next_observation, reward, terminated, truncated, info = env.step(sample.action)
        done = bool(terminated or truncated)
        td_error = learner.observe(
            int(observation),
            sample.action,
            float(reward),
            int(next_observation),
            done,
            float(sample.probabilities[sample.action]),
            sample.value,
        )
        passive.append(
            {
                "run_id": run_id,
                "benchmark": benchmark,
                "learner": learner_name,
                "seed": seed,
                "global_step": global_step,
                "episode": episode,
                "episode_step": episode_step,
                "observation": int(observation),
                "action": sample.action,
                "proxy_reward": float(reward),
                "action_probability_0": float(sample.probabilities[0]),
                "action_probability_1": float(sample.probabilities[1]),
                "policy_entropy": policy_entropy(sample.probabilities),
                "value_estimate": sample.value,
                "td_error": td_error,
                "episode_end": int(done),
            }
        )
        private = info["privileged"]
        privileged.append(
            {
                "run_id": run_id,
                "benchmark": benchmark,
                "label": label,
                "learner": learner_name,
                "seed": seed,
                "global_step": global_step,
                "latent_state": int(private["latent_state"]),
                "intended_reward": float(private["intended_reward"]),
            }
        )
        if done and global_step + 1 < total_steps:
            observation, _ = env.reset()
            episode += 1
            episode_step = 0
        else:
            observation = next_observation
            episode_step += 1
    learner.finish()
    assert_passive_schema(passive)
    return passive, privileged


def write_csv(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
