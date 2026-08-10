"""Command-line interface for validation and full benchmark runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gymnasium.utils.env_checker import check_env

from .envs import Condition, DiagnosticPOMDP, PassiveAmbiguityEnv
from .experiment import run_experiments
from .theory import factorial_value_gaps, matched_value_gaps, validate_factorial_labels


def validate() -> dict[str, object]:
    labels = validate_factorial_labels()
    for condition in Condition:
        check_env(DiagnosticPOMDP(condition), skip_render_check=True)
    for mechanism in ("observation", "reward"):
        check_env(PassiveAmbiguityEnv(mechanism), skip_render_check=True)
    return {
        "gymnasium_checks": "passed",
        "factorial_labels": labels,
        "factorial_gaps": {
            condition.value: factorial_value_gaps(condition).as_dict()
            for condition in Condition
        },
        "matched_gaps": {
            mechanism: matched_value_gaps(mechanism).as_dict()
            for mechanism in ("observation", "reward")
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="identirl",
        description="Reward-misspecification vs observation-insufficiency benchmark",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="check environments and oracle labels")
    run = subparsers.add_parser("run", help="run training, probes, diagnostics, and plots")
    run.add_argument("--output", type=Path, default=Path("artifacts/latest"))
    run.add_argument("--learners", nargs="+", choices=("tabular", "ppo"), default=["tabular", "ppo"])
    run.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    run.add_argument("--steps", type=int, default=1_000)
    run.add_argument("--probe-counts", nargs="+", type=int, default=[4, 8, 16, 32, 64])
    run.add_argument("--horizon", type=int, default=5)
    run.add_argument("--state-zero-probability", type=float, default=0.75)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        result = validate()
    else:
        result = run_experiments(
            args.output,
            learners=args.learners,
            seeds=args.seeds,
            total_steps=args.steps,
            probe_counts=args.probe_counts,
            horizon=args.horizon,
            state_zero_probability=args.state_zero_probability,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
