"""End-to-end paired-seed experiment runner and report generation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import run_identifier, train_run, write_csv
from .diagnostics import (
    CLASS_ORDER,
    PREFIX_FRACTIONS,
    build_run_features,
    leave_one_seed_out,
)
from .envs import Condition, DiagnosticPOMDP, PassiveAmbiguityEnv
from .interventions import collect_probe, intervention_cost
from .theory import factorial_value_gaps, matched_value_gaps, validate_factorial_labels


def _filter(rows: list[dict[str, Any]], **criteria: Any) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if all(str(row.get(key)) == str(value) for key, value in criteria.items())
    ]


def _metric_row(
    *,
    learner: str,
    benchmark: str,
    protocol: str,
    prefix: float,
    probe_count: int,
    cost: float,
    scores: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "learner": learner,
        "benchmark": benchmark,
        "protocol": protocol,
        "prefix_fraction": prefix,
        "probe_count": probe_count,
        "intervention_cost": cost,
        "balanced_accuracy": scores["balanced_accuracy"],
        "balanced_accuracy_ci_low": scores["balanced_accuracy_ci95"][0],
        "balanced_accuracy_ci_high": scores["balanced_accuracy_ci95"][1],
        "macro_f1": scores["macro_f1"],
        "macro_f1_ci_low": scores["macro_f1_ci95"][0],
        "macro_f1_ci_high": scores["macro_f1_ci95"][1],
        "n_runs": scores["n_runs"],
        "confusion_matrix": json.dumps(scores["confusion_matrix"]),
        "feature_names": json.dumps(scores["feature_names"]),
    }
    for label, recall in scores["per_class_recall"].items():
        row[f"recall_{label}"] = recall
    return row


def evaluate_diagnostics(
    passive_rows: list[dict[str, Any]],
    probes: dict[str, list[dict[str, Any]]],
    *,
    labels_by_run: dict[str, str],
    learners: list[str],
    probe_counts: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    flat: list[dict[str, Any]] = []
    detailed: list[dict[str, Any]] = []
    for learner in learners:
        for benchmark, class_order in (
            ("factorial", CLASS_ORDER),
            ("matched", ("observation", "reward")),
        ):
            passive = _filter(passive_rows, learner=learner, benchmark=benchmark)
            for prefix in PREFIX_FRACTIONS:
                for protocol, feature_set, random_baseline in (
                    ("random", "standard", True),
                    ("reward_curve", "reward", False),
                    ("training_statistics", "standard", False),
                ):
                    features, labels, seeds = build_run_features(
                        passive,
                        labels_by_run=labels_by_run,
                        prefix_fraction=prefix,
                        feature_set=feature_set,
                    )
                    scores = leave_one_seed_out(
                        features,
                        labels,
                        seeds,
                        class_order=class_order,
                        random_baseline=random_baseline,
                        random_seed=11,
                    )
                    row = _metric_row(
                        learner=learner,
                        benchmark=benchmark,
                        protocol=protocol,
                        prefix=prefix,
                        probe_count=0,
                        cost=0.0,
                        scores=scores,
                    )
                    flat.append(row)
                    detailed.append({**row, "scores": scores})
                for kind in ("state", "reward", "combined"):
                    probe_rows = _filter(probes[kind], learner=learner, benchmark=benchmark)
                    for probe_count in probe_counts:
                        features, labels, seeds = build_run_features(
                            passive,
                            labels_by_run=labels_by_run,
                            prefix_fraction=prefix,
                            feature_set="standard",
                            probe_rows=probe_rows,
                            probe_count=probe_count,
                        )
                        scores = leave_one_seed_out(
                            features,
                            labels,
                            seeds,
                            class_order=class_order,
                            random_seed=11,
                        )
                        cost = intervention_cost(kind, probe_count).total
                        row = _metric_row(
                            learner=learner,
                            benchmark=benchmark,
                            protocol=kind,
                            prefix=prefix,
                            probe_count=probe_count,
                            cost=cost,
                            scores=scores,
                        )
                        flat.append(row)
                        detailed.append({**row, "scores": scores})
    return flat, detailed


def _plot_prefix(metrics: list[dict[str, Any]], output: Path, max_probe: int) -> None:
    for learner in sorted({str(row["learner"]) for row in metrics}):
        for benchmark in ("factorial", "matched"):
            fig, axis = plt.subplots(figsize=(7.0, 4.2))
            for protocol in ("random", "reward_curve", "training_statistics", "state", "reward", "combined"):
                selected = [
                    row
                    for row in metrics
                    if row["learner"] == learner
                    and row["benchmark"] == benchmark
                    and row["protocol"] == protocol
                    and (row["probe_count"] in (0, max_probe))
                ]
                selected.sort(key=lambda row: row["prefix_fraction"])
                if selected:
                    axis.plot(
                        [100 * float(row["prefix_fraction"]) for row in selected],
                        [float(row["balanced_accuracy"]) for row in selected],
                        marker="o",
                        label=protocol.replace("_", " "),
                    )
            chance = 0.25 if benchmark == "factorial" else 0.5
            axis.axhline(chance, color="black", linestyle="--", linewidth=1, label="chance")
            axis.set(xlabel="Training prefix (%)", ylabel="Balanced accuracy", ylim=(0, 1.05))
            axis.set_title(f"{benchmark.title()} diagnosis — {learner}")
            axis.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            fig.savefig(output / f"accuracy_prefix_{benchmark}_{learner}.png", dpi=180)
            plt.close(fig)


def _plot_cost(metrics: list[dict[str, Any]], output: Path) -> None:
    for learner in sorted({str(row["learner"]) for row in metrics}):
        for benchmark in ("factorial", "matched"):
            fig, axis = plt.subplots(figsize=(7.0, 4.2))
            baseline = next(
                row
                for row in metrics
                if row["learner"] == learner
                and row["benchmark"] == benchmark
                and row["protocol"] == "training_statistics"
                and np.isclose(row["prefix_fraction"], 0.2)
            )
            axis.scatter([0], [baseline["balanced_accuracy"]], label="passive", s=55)
            for protocol in ("state", "reward", "combined"):
                selected = [
                    row
                    for row in metrics
                    if row["learner"] == learner
                    and row["benchmark"] == benchmark
                    and row["protocol"] == protocol
                    and np.isclose(row["prefix_fraction"], 0.2)
                ]
                selected.sort(key=lambda row: row["intervention_cost"])
                axis.plot(
                    [float(row["intervention_cost"]) for row in selected],
                    [float(row["balanced_accuracy"]) for row in selected],
                    marker="o",
                    label=protocol,
                )
            axis.set(xlabel="Intervention cost C(d)", ylabel="Balanced accuracy", ylim=(0, 1.05))
            axis.set_title(f"Accuracy–cost curve — {benchmark}, {learner}")
            axis.legend()
            fig.tight_layout()
            fig.savefig(output / f"accuracy_cost_{benchmark}_{learner}.png", dpi=180)
            plt.close(fig)


def _plot_confusions(metrics: list[dict[str, Any]], output: Path, max_probe: int) -> None:
    for learner in sorted({str(row["learner"]) for row in metrics}):
        for benchmark, labels in (
            ("factorial", CLASS_ORDER),
            ("matched", ("observation", "reward")),
        ):
            selected = [
                row
                for row in metrics
                if row["learner"] == learner
                and row["benchmark"] == benchmark
                and np.isclose(row["prefix_fraction"], 0.2)
                and (
                    row["protocol"] == "training_statistics"
                    or (row["protocol"] == "combined" and row["probe_count"] == max_probe)
                )
            ]
            fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
            for axis, row in zip(axes, selected):
                matrix = np.asarray(json.loads(row["confusion_matrix"]))
                image = axis.imshow(matrix, cmap="Blues", vmin=0)
                for (i, j), value in np.ndenumerate(matrix):
                    axis.text(j, i, str(value), ha="center", va="center")
                axis.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
                axis.set_yticks(range(len(labels)), labels=labels)
                axis.set_title(str(row["protocol"]).replace("_", " "))
                axis.set(xlabel="Predicted", ylabel="True")
                fig.colorbar(image, ax=axis, fraction=0.046)
            fig.suptitle(f"Confusion matrices — {benchmark}, {learner}")
            fig.tight_layout()
            fig.savefig(output / f"confusion_{benchmark}_{learner}.png", dpi=180)
            plt.close(fig)


def _empirical_equivalence(
    rows: list[dict[str, Any]], labels_by_run: dict[str, str]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for learner in sorted({str(row["learner"]) for row in rows}):
        learner_rows = _filter(rows, learner=learner, benchmark="matched")
        for action in (0, 1):
            rates: dict[str, float] = {}
            for label in ("observation", "reward"):
                subset = [
                    float(row["proxy_reward"])
                    for row in learner_rows
                    if labels_by_run[str(row["run_id"])] == label
                    and int(row["action"]) == action
                ]
                rates[label] = float(np.mean(subset))
            result.append(
                {
                    "learner": learner,
                    "action": action,
                    "observation_mechanism_rate": rates["observation"],
                    "reward_mechanism_rate": rates["reward"],
                    "absolute_difference": abs(rates["observation"] - rates["reward"]),
                }
            )
    return result


def _minimum_cost_table(
    metrics: list[dict[str, Any]], *, target_accuracy: float = 0.90
) -> list[dict[str, Any]]:
    """Choose the cheapest protocol whose 95% lower bound meets the target."""
    result: list[dict[str, Any]] = []
    learners = sorted({str(row["learner"]) for row in metrics})
    for learner in learners:
        for benchmark in ("factorial", "matched"):
            for prefix in PREFIX_FRACTIONS:
                candidates = [
                    row
                    for row in metrics
                    if row["learner"] == learner
                    and row["benchmark"] == benchmark
                    and np.isclose(row["prefix_fraction"], prefix)
                    and row["protocol"] in {"training_statistics", "state", "reward", "combined"}
                    and float(row["balanced_accuracy_ci_low"]) >= target_accuracy
                ]
                candidates.sort(
                    key=lambda row: (
                        float(row["intervention_cost"]),
                        int(row["probe_count"]),
                        str(row["protocol"]),
                    )
                )
                if candidates:
                    best = candidates[0]
                    result.append(
                        {
                            "learner": learner,
                            "benchmark": benchmark,
                            "prefix_fraction": prefix,
                            "target_accuracy": target_accuracy,
                            "status": "identified",
                            "protocol": best["protocol"],
                            "probe_count": best["probe_count"],
                            "minimum_cost": best["intervention_cost"],
                            "balanced_accuracy": best["balanced_accuracy"],
                            "ci_lower_bound": best["balanced_accuracy_ci_low"],
                        }
                    )
                else:
                    result.append(
                        {
                            "learner": learner,
                            "benchmark": benchmark,
                            "prefix_fraction": prefix,
                            "target_accuracy": target_accuracy,
                            "status": "not_reached",
                            "protocol": "",
                            "probe_count": "",
                            "minimum_cost": "",
                            "balanced_accuracy": "",
                            "ci_lower_bound": "",
                        }
                    )
    return result


def _write_findings(
    output: Path,
    metrics: list[dict[str, Any]],
    equivalence: list[dict[str, Any]],
    *,
    max_probe: int,
) -> None:
    def score(learner: str, benchmark: str, protocol: str) -> float:
        match = next(
            row
            for row in metrics
            if row["learner"] == learner
            and row["benchmark"] == benchmark
            and row["protocol"] == protocol
            and np.isclose(row["prefix_fraction"], 0.2)
            and (int(row["probe_count"]) in (0, max_probe))
        )
        return float(match["balanced_accuracy"])

    max_difference = max(float(row["absolute_difference"]) for row in equivalence)
    lines = [
        "# Five-seed benchmark findings",
        "",
        "These are pipeline-validation results, not a final high-power scientific claim.",
        "Each value uses leave-one-paired-seed-out evaluation over five seeds.",
        "",
        "## Main checks",
        "",
    ]
    for learner in sorted({str(row["learner"]) for row in metrics}):
        lines.append(
            f"- **{learner}:** at the 20% prefix, standard passive balanced accuracy "
            f"is {score(learner, 'factorial', 'training_statistics'):.2f} on the factorial "
            f"benchmark and {score(learner, 'matched', 'training_statistics'):.2f} on the "
            f"exact ambiguity pair; the combined {max_probe}-episode probe reaches "
            f"{score(learner, 'factorial', 'combined'):.2f} and "
            f"{score(learner, 'matched', 'combined'):.2f}, respectively."
        )
    lines.extend(
        [
            f"- Across actions and learners, the largest empirical difference in matched "
            f"conditional proxy-reward rates is {max_difference:.4f}; population distributions "
            "are equal by construction.",
            "- All conditions use binary proxy rewards with support `[0, 1]`, so reward scale "
            "is not a label shortcut.",
            "",
            "## Shortcuts, confounds, and scope",
            "",
            "- In the factorial benchmark, observation identity and coverage intentionally reveal "
            "whether observation corruption is enabled. Treat this as a controlled positive "
            "benchmark, not evidence that passive logs identify unrestricted causal classes.",
            "- The matched pair removes that shortcut and matches the entire conditional passive "
            "reward distribution. Its reward-only and observation-only classes remain "
            "non-identifiable from passive population data even when finite-sample scores wander "
            "above or below chance.",
            "- A state-only probe can solve this structured family because the proxy/state relation "
            "is deliberately diagnostic. The unrestricted four-class guarantee in the paper still "
            "requires structural separation and can require both probes.",
            "- The environment has zero memory/architecture gap by construction. The implemented "
            "history-dependence statistic is a diagnostic association, not proof of intrinsic "
            "missing information.",
            "- Five seeds produce wide intervals. Choose final run counts from pilot variance and "
            "report the generated bootstrap confidence intervals.",
        ]
    )
    with (output / "findings.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run_experiments(
    output_dir: str | Path,
    *,
    learners: list[str],
    seeds: list[int],
    total_steps: int = 1_000,
    probe_counts: list[int] | None = None,
    horizon: int = 5,
    state_zero_probability: float = 0.75,
) -> dict[str, Any]:
    """Run all four labels and the exact passive-equivalence control."""
    if len(seeds) < 2:
        raise ValueError("at least two seeds are required for held-out evaluation")
    probe_counts = sorted(set(probe_counts or [4, 8, 16, 32, 64]))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    validate_factorial_labels(state_zero_probability=state_zero_probability)

    passive: list[dict[str, Any]] = []
    privileged: list[dict[str, Any]] = []
    probes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    manifest: list[dict[str, Any]] = []
    max_probe = max(probe_counts)

    jobs: list[tuple[str, str, Callable[[], Any]]] = []
    for condition in Condition:
        jobs.append(
            (
                "factorial",
                condition.value,
                lambda condition=condition: DiagnosticPOMDP(
                    condition,
                    horizon=horizon,
                    state_zero_probability=state_zero_probability,
                ),
            )
        )
    for mechanism in ("observation", "reward"):
        jobs.append(
            (
                "matched",
                mechanism,
                lambda mechanism=mechanism: PassiveAmbiguityEnv(
                    mechanism,
                    horizon=horizon,
                    state_zero_probability=state_zero_probability,
                ),
            )
        )

    for learner in learners:
        for benchmark, label, env_factory in jobs:
            for seed in seeds:
                run_id = run_identifier(benchmark, learner, label, seed)
                manifest.append(
                    {
                        "run_id": run_id,
                        "benchmark": benchmark,
                        "label": label,
                        "learner": learner,
                        "seed": seed,
                    }
                )
                run_passive, run_privileged = train_run(
                    env_factory(),
                    label=label,
                    benchmark=benchmark,
                    learner_name=learner,
                    seed=seed,
                    total_steps=total_steps,
                    run_id=run_id,
                )
                passive.extend(run_passive)
                privileged.extend(run_privileged)
                for kind in ("state", "reward", "combined"):
                    probes[kind].extend(
                        collect_probe(
                            env_factory,
                            label=label,
                            benchmark=benchmark,
                            learner=learner,
                            seed=seed,
                            kind=kind,
                            probe_episodes=max_probe,
                            run_id=run_id,
                        )
                    )

    write_csv(output / "passive_training.csv", passive)
    write_csv(output / "privileged_evaluation.csv", privileged)
    write_csv(output / "run_manifest.csv", manifest)
    for kind, rows in probes.items():
        write_csv(output / f"probe_{kind}.csv", rows)

    oracle_rows: list[dict[str, Any]] = []
    for condition in Condition:
        oracle_rows.append(
            {
                "benchmark": "factorial",
                "label": condition.value,
                **factorial_value_gaps(
                    condition,
                    state_zero_probability=state_zero_probability,
                    horizon=horizon,
                ).as_dict(),
                "proxy_reward_min": 0.0,
                "proxy_reward_max": 1.0,
            }
        )
    for mechanism in ("observation", "reward"):
        oracle_rows.append(
            {
                "benchmark": "matched",
                "label": mechanism,
                **matched_value_gaps(
                    mechanism,
                    state_zero_probability=state_zero_probability,
                    horizon=horizon,
                ).as_dict(),
                "proxy_reward_min": 0.0,
                "proxy_reward_max": 1.0,
            }
        )
    write_csv(output / "oracle_value_gaps.csv", oracle_rows)

    labels_by_run = {str(row["run_id"]): str(row["label"]) for row in manifest}
    metrics, detailed = evaluate_diagnostics(
        passive,
        probes,
        labels_by_run=labels_by_run,
        learners=learners,
        probe_counts=probe_counts,
    )
    write_csv(output / "diagnostic_metrics.csv", metrics)
    with (output / "diagnostic_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(detailed, handle, indent=2)
    equivalence = _empirical_equivalence(passive, labels_by_run)
    write_csv(output / "matched_equivalence_check.csv", equivalence)
    minimum_cost = _minimum_cost_table(metrics)
    write_csv(output / "minimum_cost_interventions.csv", minimum_cost)
    _write_findings(output, metrics, equivalence, max_probe=max_probe)

    _plot_prefix(metrics, output, max_probe)
    _plot_cost(metrics, output)
    _plot_confusions(metrics, output, max_probe)

    artifact_names = {path.name for path in output.iterdir()}
    artifact_names.add("run_summary.json")
    summary = {
        "learners": learners,
        "seeds": seeds,
        "total_steps_per_run": total_steps,
        "factorial_runs": len(learners) * len(seeds) * 4,
        "matched_control_runs": len(learners) * len(seeds) * 2,
        "passive_rows": len(passive),
        "probe_counts": probe_counts,
        "artifacts": sorted(artifact_names),
    }
    with (output / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
