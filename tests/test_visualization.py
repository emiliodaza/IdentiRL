from pathlib import Path

from identirl.visualization import select_run


def test_visualization_joins_passive_and_privileged_rows_by_opaque_run_id():
    artifact_dir = Path(__file__).parents[1] / "artifacts" / "five_seed"
    metadata, passive, privileged = select_run(
        artifact_dir,
        learner="ppo",
        benchmark="factorial",
        label="both",
        seed=0,
    )
    assert metadata["run_id"].startswith("run-")
    assert len(passive) == len(privileged) == 1_000
    assert "latent_state" not in passive[0]
    assert "intended_reward" not in passive[0]
    assert passive[0]["global_step"] == privileged[0]["global_step"]
