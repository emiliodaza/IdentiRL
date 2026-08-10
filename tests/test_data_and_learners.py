import pytest

from identirl.data import PASSIVE_FIELDS, assert_passive_schema, train_run
from identirl.envs import DiagnosticPOMDP
from identirl.interventions import collect_probe, intervention_cost
from identirl.learners import TabularPPOLearner


def test_passive_and_privileged_logs_are_separate():
    passive, privileged = train_run(
        DiagnosticPOMDP("both"),
        label="both",
        benchmark="factorial",
        learner_name="tabular",
        seed=0,
        total_steps=20,
    )
    assert len(passive) == len(privileged) == 20
    assert set(passive[0]) == set(PASSIVE_FIELDS)
    assert "label" not in passive[0]
    assert "both" not in passive[0]["run_id"]
    assert "latent_state" not in passive[0]
    assert "intended_reward" not in passive[0]
    assert {"latent_state", "intended_reward"} <= set(privileged[0])
    contaminated = [{**passive[0], "intended_reward": 1.0}]
    with pytest.raises(ValueError, match="violates schema"):
        assert_passive_schema(contaminated)


def test_tabular_ppo_updates_policy_and_value():
    learner = TabularPPOLearner(3, 2, seed=4, rollout_steps=4)
    for _ in range(12):
        sample = learner.act(0)
        reward = float(sample.action == 1)
        learner.observe(0, sample.action, reward, 0, True, sample.probabilities[sample.action], sample.value)
    learner.finish()
    assert learner.values[0] > 0
    assert learner.action_probabilities(0)[1] > 0.5


def test_probe_schemas_and_costs():
    factory = lambda: DiagnosticPOMDP("both", horizon=1)
    state = collect_probe(factory, label="both", benchmark="factorial", learner="tabular", seed=0, kind="state", probe_episodes=8)
    reward = collect_probe(factory, label="both", benchmark="factorial", learner="tabular", seed=0, kind="reward", probe_episodes=8)
    combined = collect_probe(factory, label="both", benchmark="factorial", learner="tabular", seed=0, kind="combined", probe_episodes=8)
    assert "latent_state" in state[0] and "intended_reward" not in state[0]
    assert "intended_reward" in reward[0] and "latent_state" not in reward[0]
    assert {"latent_state", "intended_reward"} <= set(combined[0])
    assert intervention_cost("state", 8).total == 40
    assert intervention_cost("combined", 8).total == 72
