from env import TinyDiagnosticPOMDP

"""same as env.py - needs official mathematical formulation for env before being used; this is an ex until 
otherwise adjusted"""
def run_test(name, observation_corruption, reward_corruption):
    print(f"\n{name}")

    env = TinyDiagnosticPOMDP(
        observation_corruption=observation_corruption,
        reward_corruption=reward_corruption,
    )

    for episode in range(5):
        observation, info = env.reset(seed=episode)

        action = env.action_space.sample()

        next_observation, reward, terminated, truncated, info = env.step(
            action
        )

        print(
            f"episode={episode}, "
            f"state={info['latent_state']}, "
            f"observation={observation}, "
            f"action={action}, "
            f"training_reward={reward}, "
            f"intended_reward={info['intended_reward']}, "
            f"proxy_reward={info['proxy_reward']}"
        )

    env.close()


run_test(
    name="Neither failure",
    observation_corruption=False,
    reward_corruption=False,
)

run_test(
    name="Reward failure",
    observation_corruption=False,
    reward_corruption=True,
)

run_test(
    name="Observation failure",
    observation_corruption=True,
    reward_corruption=False,
)

run_test(
    name="Both failures",
    observation_corruption=True,
    reward_corruption=True,
)

from gymnasium.utils.env_checker import check_env
from env import TinyDiagnosticPOMDP

env = TinyDiagnosticPOMDP()
check_env(env)

print("Gymnasium environment check passed.")