"""Leakage-safe feature extraction and diagnostic evaluation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import PASSIVE_FIELDS, assert_passive_schema


PREFIX_FRACTIONS = (0.05, 0.10, 0.20)
CLASS_ORDER = ("neither", "reward", "observation", "both")


def group_by_run(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["run_id"])].append(row)
    for run_rows in grouped.values():
        run_rows.sort(key=lambda item: int(item.get("global_step", item.get("probe_episode", 0))))
    return dict(grouped)


def prefix_rows(rows: Sequence[dict[str, Any]], fraction: float) -> list[dict[str, Any]]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    count = max(1, int(np.ceil(len(rows) * fraction)))
    return list(rows[:count])


def _float(rows: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def _categorical_entropy(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum())


def history_dependence_score(rows: Sequence[dict[str, Any]]) -> float:
    """Held-out MSE gain from adding the previous observable transition."""
    samples: list[tuple[list[float], list[float], float]] = []
    for previous, current in zip(rows, rows[1:]):
        if int(previous["episode_end"]):
            continue
        base = [
            1.0,
            float(current["observation"]),
            float(current["action"]),
        ]
        history = base + [
            float(previous["observation"]),
            float(previous["action"]),
            float(previous["proxy_reward"]),
        ]
        samples.append((base, history, float(current["proxy_reward"])))
    if len(samples) < 8:
        return 0.0
    split = max(4, len(samples) // 2)
    if split >= len(samples):
        return 0.0

    def held_out_mse(which: int) -> float:
        train_x = np.asarray([sample[which] for sample in samples[:split]])
        train_y = np.asarray([sample[2] for sample in samples[:split]])
        test_x = np.asarray([sample[which] for sample in samples[split:]])
        test_y = np.asarray([sample[2] for sample in samples[split:]])
        regularizer = 1e-3 * np.eye(train_x.shape[1])
        weights = np.linalg.solve(train_x.T @ train_x + regularizer, train_x.T @ train_y)
        return float(np.mean((test_x @ weights - test_y) ** 2))

    return held_out_mse(0) - held_out_mse(1)


def reward_curve_features(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    reward = _float(rows, "proxy_reward")
    midpoint = max(1, len(reward) // 2)
    return {
        "reward_mean": float(reward.mean()),
        "reward_std": float(reward.std()),
        "reward_slope": _slope(reward),
        "reward_change": float(reward[midpoint:].mean() - reward[:midpoint].mean()),
    }


def standard_training_features(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    features = reward_curve_features(rows)
    actions = _float(rows, "action")
    observations = _float(rows, "observation")
    td_error = _float(rows, "td_error")
    episode_ends = _float(rows, "episode_end")
    features.update(
        {
            "policy_entropy_mean": float(_float(rows, "policy_entropy").mean()),
            "action_diversity": _categorical_entropy(actions),
            "observation_coverage": float(len(np.unique(observations))),
            "observation_entropy": _categorical_entropy(observations),
            "td_error_mean": float(td_error.mean()),
            "td_error_abs_mean": float(np.abs(td_error).mean()),
            "td_error_std": float(td_error.std()),
            "episode_length": float(len(rows) / max(1.0, episode_ends.sum())),
            "history_dependence": history_dependence_score(rows),
        }
    )
    return features


def probe_features(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    kind = str(rows[0]["kind"])
    action = _float(rows, "action")
    proxy = _float(rows, "proxy_reward")
    features: dict[str, float] = {}
    if kind in {"state", "combined"}:
        state = _float(rows, "latent_state")
        state_match = (action == state).astype(float)
        features.update(
            {
                "state_proxy_agreement": float(np.mean(proxy == state_match)),
                "state_action_match_proxy": float(proxy[state_match == 1].mean())
                if np.any(state_match == 1)
                else 0.0,
                "state_action_mismatch_proxy": float(proxy[state_match == 0].mean())
                if np.any(state_match == 0)
                else 0.0,
                "state_observation_agreement": float(
                    np.mean(state == _float(rows, "observation"))
                ),
            }
        )
    if kind in {"reward", "combined"}:
        intended = _float(rows, "intended_reward")
        features.update(
            {
                "reward_audit_disagreement": float(np.mean(proxy != intended)),
                "audited_intended_mean": float(intended.mean()),
                "audited_proxy_mean": float(proxy.mean()),
                "intended_action0": float(intended[action == 0].mean())
                if np.any(action == 0)
                else 0.0,
                "intended_action1": float(intended[action == 1].mean())
                if np.any(action == 1)
                else 0.0,
            }
        )
    return features


def _matrix(feature_dicts: Sequence[dict[str, float]]) -> tuple[np.ndarray, list[str]]:
    keys = sorted({key for features in feature_dicts for key in features})
    return np.asarray([[features.get(key, 0.0) for key in keys] for features in feature_dicts]), keys


def _scores(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]) -> dict[str, Any]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_recall": {
            label: float(value)
            for label, value in zip(
                labels,
                recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
            )
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def leave_one_seed_out(
    features: Sequence[dict[str, float]],
    labels: Sequence[str],
    seeds: Sequence[int],
    *,
    class_order: Sequence[str],
    random_baseline: bool = False,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Split entire runs by paired seed, never individual transitions."""
    x, feature_names = _matrix(features)
    y = np.asarray(labels)
    seed_array = np.asarray(seeds)
    predictions = np.empty(len(y), dtype=object)
    rng = np.random.default_rng(random_seed)
    for held_out_seed in np.unique(seed_array):
        test = seed_array == held_out_seed
        train = ~test
        if random_baseline:
            predictions[test] = rng.choice(class_order, size=int(test.sum()))
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2_000, random_state=random_seed),
        )
        model.fit(x[train], y[train])
        predictions[test] = model.predict(x[test])
    result = _scores(y, predictions, class_order)
    unique_seeds = np.unique(seed_array)
    bootstrap_balanced: list[float] = []
    bootstrap_f1: list[float] = []
    for _ in range(300):
        sampled = rng.choice(unique_seeds, size=len(unique_seeds), replace=True)
        indices = np.concatenate([np.flatnonzero(seed_array == seed) for seed in sampled])
        bootstrap_balanced.append(float(balanced_accuracy_score(y[indices], predictions[indices])))
        bootstrap_f1.append(
            float(
                f1_score(
                    y[indices],
                    predictions[indices],
                    labels=class_order,
                    average="macro",
                    zero_division=0,
                )
            )
        )
    result["balanced_accuracy_ci95"] = [
        float(value) for value in np.quantile(bootstrap_balanced, [0.025, 0.975])
    ]
    result["macro_f1_ci95"] = [
        float(value) for value in np.quantile(bootstrap_f1, [0.025, 0.975])
    ]
    result["feature_names"] = feature_names
    result["n_runs"] = len(y)
    return result


def build_run_features(
    passive_rows: list[dict[str, Any]],
    *,
    labels_by_run: dict[str, str],
    prefix_fraction: float,
    feature_set: str,
    probe_rows: list[dict[str, Any]] | None = None,
    probe_count: int | None = None,
) -> tuple[list[dict[str, float]], list[str], list[int]]:
    assert_passive_schema(passive_rows)
    passive_groups = group_by_run(passive_rows)
    probe_groups = group_by_run(probe_rows or [])
    features: list[dict[str, float]] = []
    labels: list[str] = []
    seeds: list[int] = []
    for run_id in sorted(passive_groups):
        rows = prefix_rows(passive_groups[run_id], prefix_fraction)
        if feature_set == "reward":
            current = reward_curve_features(rows)
        elif feature_set == "standard":
            current = standard_training_features(rows)
        else:
            raise ValueError("feature_set must be 'reward' or 'standard'")
        if probe_rows is not None:
            selected = probe_groups.get(run_id, [])
            if probe_count is not None:
                selected = selected[:probe_count]
            current.update(probe_features(selected))
        features.append(current)
        if run_id not in labels_by_run:
            raise ValueError(f"missing ground-truth label for {run_id}")
        labels.append(str(labels_by_run[run_id]))
        seeds.append(int(rows[0]["seed"]))
    return features, labels, seeds
