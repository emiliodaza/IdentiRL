from identirl.data import train_run
from identirl.diagnostics import build_run_features, leave_one_seed_out, prefix_rows
from identirl.envs import Condition, DiagnosticPOMDP


def _small_dataset():
    rows = []
    labels = {}
    for seed in range(3):
        for condition in Condition:
            passive, _ = train_run(
                DiagnosticPOMDP(condition),
                label=condition.value,
                benchmark="factorial",
                learner_name="tabular",
                seed=seed,
                total_steps=80,
            )
            rows.extend(passive)
            labels[passive[0]["run_id"]] = condition.value
    return rows, labels


def test_prefix_extraction_uses_beginning_only():
    rows = [{"global_step": index} for index in range(100)]
    prefix = prefix_rows(rows, 0.05)
    assert [row["global_step"] for row in prefix] == list(range(5))


def test_run_level_seed_split_evaluates_all_runs():
    rows, labels_by_run = _small_dataset()
    features, labels, seeds = build_run_features(
        rows,
        labels_by_run=labels_by_run,
        prefix_fraction=0.2,
        feature_set="standard",
    )
    result = leave_one_seed_out(
        features,
        labels,
        seeds,
        class_order=("neither", "reward", "observation", "both"),
    )
    assert result["n_runs"] == 12
    assert len(result["confusion_matrix"]) == 4
    assert 0 <= result["balanced_accuracy"] <= 1
