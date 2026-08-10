import numpy as np
from gymnasium.utils.env_checker import check_env

from identirl.envs import Condition, DiagnosticPOMDP, PassiveAmbiguityEnv


def test_factorial_environments_pass_gymnasium_checker():
    for condition in Condition:
        check_env(DiagnosticPOMDP(condition), skip_render_check=True)


def test_factorial_reward_and_observation_toggles_are_independent():
    aligned = DiagnosticPOMDP(Condition.NEITHER)
    hidden = DiagnosticPOMDP(Condition.OBSERVATION)
    inverted = DiagnosticPOMDP(Condition.REWARD)
    both = DiagnosticPOMDP(Condition.BOTH)
    for env in (aligned, hidden, inverted, both):
        env.reset(seed=7)
    assert aligned._state == hidden._state == inverted._state == both._state
    state = aligned._state
    assert aligned._observation() == state
    assert inverted._observation() == state
    assert hidden._observation() == both._observation() == 2
    assert aligned.rewards_for(state) == hidden.rewards_for(state) == (1.0, 1.0)
    assert inverted.rewards_for(state) == both.rewards_for(state) == (0.0, 1.0)


def test_matched_pair_has_same_conditional_passive_reward_distribution():
    samples = 8_000
    for action, expected in ((0, 0.75), (1, 0.25)):
        rates = []
        for mechanism in ("observation", "reward"):
            env = PassiveAmbiguityEnv(mechanism, horizon=1)
            rewards = []
            for seed in range(samples):
                env.reset(seed=seed)
                _, reward, *_ = env.step(action)
                rewards.append(reward)
            rates.append(float(np.mean(rewards)))
            assert abs(rates[-1] - expected) < 0.025
        assert abs(rates[0] - rates[1]) < 0.025
