"""2D episode playback for trained IdentiRL runs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_run(
    artifact_dir: str | Path,
    *,
    learner: str,
    benchmark: str,
    label: str,
    seed: int,
) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, str]]]:
    artifact_dir = Path(artifact_dir)
    manifest = _read(artifact_dir / "run_manifest.csv")
    matches = [
        row
        for row in manifest
        if row["learner"] == learner
        and row["benchmark"] == benchmark
        and row["label"] == label
        and int(row["seed"]) == seed
    ]
    if len(matches) != 1:
        raise ValueError(
            "expected one run for "
            f"learner={learner}, benchmark={benchmark}, label={label}, seed={seed}; "
            f"found {len(matches)}"
        )
    metadata = matches[0]
    run_id = metadata["run_id"]
    passive = [
        row
        for row in _read(artifact_dir / "passive_training.csv")
        if row["run_id"] == run_id
    ]
    privileged = [
        row
        for row in _read(artifact_dir / "privileged_evaluation.csv")
        if row["run_id"] == run_id
    ]
    passive.sort(key=lambda row: int(row["global_step"]))
    privileged.sort(key=lambda row: int(row["global_step"]))
    if not passive or len(passive) != len(privileged):
        raise ValueError("passive and privileged run data are missing or misaligned")
    return metadata, passive, privileged


def _reward_color(value: float) -> str:
    return "#17804f" if value > 0.5 else "#b43b35"


def _draw_frame(
    figure: plt.Figure,
    world_axis: plt.Axes,
    return_axis: plt.Axes,
    frame: int,
    metadata: dict[str, str],
    passive: list[dict[str, str]],
    privileged: list[dict[str, str]],
) -> None:
    world_axis.clear()
    return_axis.clear()
    row = passive[frame]
    private = privileged[frame]
    state = int(private["latent_state"])
    observation = int(row["observation"])
    action = int(row["action"])
    proxy = float(row["proxy_reward"])
    intended = float(private["intended_reward"])

    world_axis.set_xlim(-0.2, 2.2)
    world_axis.set_ylim(-0.15, 1.65)
    world_axis.axis("off")
    for room in (0, 1):
        face = "#e7f0fb" if room == state else "#f7f8fa"
        world_axis.add_patch(
            FancyBboxPatch(
                (room * 1.15, 0.72),
                0.9,
                0.55,
                boxstyle="round,pad=0.03,rounding_size=0.05",
                edgecolor="#425466",
                facecolor=face,
                linewidth=1.5,
            )
        )
        world_axis.text(room * 1.15 + 0.45, 1.18, f"Latent state {room}", ha="center", fontsize=11)
    state_x = state * 1.15 + 0.45
    world_axis.add_patch(Circle((state_x, 0.91), 0.11, color="#2563eb"))
    world_axis.text(state_x, 0.91, "S", ha="center", va="center", color="white", weight="bold")

    agent_x = 1.025
    world_axis.add_patch(Circle((agent_x, 0.30), 0.13, color="#2f3542"))
    world_axis.text(agent_x, 0.30, "agent", ha="center", va="center", color="white", fontsize=8)
    target_x = action * 1.15 + 0.45
    world_axis.add_patch(
        FancyArrowPatch(
            (agent_x, 0.44),
            (target_x, 0.70),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.2,
            color="#7c3aed",
        )
    )
    world_axis.text((agent_x + target_x) / 2, 0.53, f"action {action}", ha="center", color="#6d28d9")

    observation_text = "hidden (?)" if observation == 2 else str(observation)
    world_axis.text(
        1.025,
        1.48,
        f"agent observation: {observation_text}",
        ha="center",
        fontsize=12,
        weight="bold",
    )
    world_axis.text(
        0.10,
        0.03,
        f"proxy reward = {proxy:.0f}",
        color="white",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": _reward_color(proxy), "edgecolor": "none"},
        fontsize=10,
    )
    world_axis.text(
        1.25,
        0.03,
        f"intended reward = {intended:.0f}",
        color="white",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": _reward_color(intended), "edgecolor": "none"},
        fontsize=10,
    )
    world_axis.set_title(
        f"{metadata['label'].title()} condition | episode {row['episode']} | step {row['episode_step']}",
        fontsize=13,
        pad=8,
    )

    x = np.arange(frame + 1)
    proxy_values = np.asarray([float(item["proxy_reward"]) for item in passive[: frame + 1]])
    intended_values = np.asarray(
        [float(item["intended_reward"]) for item in privileged[: frame + 1]]
    )
    return_axis.plot(x, np.cumsum(proxy_values), label="proxy return", color="#7c3aed", linewidth=2)
    return_axis.plot(x, np.cumsum(intended_values), label="intended return", color="#17804f", linewidth=2)
    return_axis.set_xlim(0, max(1, len(passive) - 1))
    return_axis.set_ylim(0, max(2.0, float(frame + 1)))
    return_axis.set_xlabel("training transition")
    return_axis.set_ylabel("cumulative return")
    return_axis.grid(alpha=0.22)
    return_axis.legend(loc="upper left", frameon=False)
    return_axis.set_title(f"{metadata['learner'].upper()} behavior over time", fontsize=13)
    figure.suptitle(
        "IdentiRL: what the learner sees versus what the evaluator knows",
        fontsize=15,
        weight="bold",
    )


def create_animation(
    artifact_dir: str | Path,
    output_path: str | Path,
    *,
    learner: str = "ppo",
    benchmark: str = "factorial",
    label: str = "both",
    seed: int = 0,
    steps: int = 80,
    fps: int = 5,
    snapshot_path: str | Path | None = None,
) -> Path:
    metadata, passive, privileged = select_run(
        artifact_dir,
        learner=learner,
        benchmark=benchmark,
        label=label,
        seed=seed,
    )
    count = min(max(1, steps), len(passive))
    passive = passive[:count]
    privileged = privileged[:count]
    figure, (world_axis, return_axis) = plt.subplots(1, 2, figsize=(10.6, 5.2))
    figure.subplots_adjust(top=0.82, bottom=0.14, left=0.05, right=0.97, wspace=0.22)

    def update(frame: int) -> None:
        _draw_frame(figure, world_axis, return_axis, frame, metadata, passive, privileged)

    movie = animation.FuncAnimation(figure, update, frames=count, interval=1000 / fps, repeat=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    movie.save(output_path, writer=animation.PillowWriter(fps=fps), dpi=105)
    if snapshot_path is not None:
        snapshot_path = Path(snapshot_path)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        update(count - 1)
        figure.savefig(snapshot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path
