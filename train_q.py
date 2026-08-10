import numpy as np
import pandas as pd
import os
from envs.tiny_one_step import TinyDiagnosticPOMDP

NUM_EPISODES = 1000
LEARNING_RATE =  0.1
EPSILON = 0.1
SEED = 0

CONDITIONS = {
    "neither": {
        "observation_corruption": False,
        "reward_corruption": False,
    },

    "reward_failure": {
        "observation_corruption": False,
        "reward_corruption": True,
    },

    "observation_failure": {
        "observation_corruption": True,
        "reward_corruption": False,
    },

    "both": {
        "observation_corruption": True,
        "reward_corruption": True,
    },
}

def train_q_learning(
        observation_corruption,
        reward_corruption,
        seed,
):

    env = TinyDiagnosticPOMDP(
        observation_corruption=observation_corruption,
        reward_corruption=reward_corruption,
    )

    env.action_space.seed(seed)

    num_observations = env.observation_space.n
    num_actions = env.action_space.n
    q_table = np.zeros((num_observations, num_actions))

    print("Initial Q-table")
    print(q_table)

    rng = np.random.default_rng(seed)

    passive_logs = []
    privileged_logs = []

    for episode in range(NUM_EPISODES):
        observation,info = env.reset(seed = seed + episode)

        #Compute prob of each action exactly - epsilon greedy
        greedy_action = int(np.argmax(q_table[observation]))

        action_probabilities = np.full(
            num_actions,
            EPSILON / num_actions
        )

        action_probabilities[greedy_action] += 1.0 - EPSILON

        if rng.random() < EPSILON:
            action = env.action_space.sample()
            action_type = "explore"
        else:
            action = greedy_action
            action_type = "exploit"

        next_observation,reward,terminated,truncated, info = env.step(action)

        old_q = q_table[observation, action]
        value_estimate= np.max(q_table[observation])

        td_error = reward - old_q

        passive_logs.append({
            "seed": seed,
            "episode": episode,
            "step": 0,
            "observation": observation,
            "action": action,
            "training_reward": reward,
            "action_prob_0": action_probabilities[0],
            "action_prob_1": action_probabilities[1],
            "value_estimate": value_estimate,
            "td_error": td_error,
            "terminated": terminated,
        })

        privileged_logs.append({
            "seed": seed,
            "episode": episode,
            "step": 0,
            "latent_state": info["latent_state"],
            "intended_reward": info["intended_reward"],
            "proxy_reward": info["proxy_reward"],
        })
        q_table[observation,action] = (
            old_q + LEARNING_RATE * td_error
        )
        if episode % 100  == 0:
            print(

            f"Episode {episode}: "
            f"observation={observation}, "
            f"action={action}, "
            f"type={action_type}, "
            f"reward={reward}, "
            f"td_error={td_error:.3f}"
        )
    print("\nFinal Q-table:")
    print(q_table)

    print("\nLearned policy:")

    for observation in range(num_observations):
        best_action = int(np.argmax(q_table[observation]))

        print(
            f"Observation {observation} "
            f"-> Action {best_action}"
        )

    env.close()
    return q_table, passive_logs, privileged_logs

#make log folder if does not exist
os.makedirs("logs", exist_ok=True)
#5x seed per run
for seed in range(5):

    for condition_name, settings in CONDITIONS.items():
        print("\n==========================")
        print(condition_name)
        print("==========================")


        q_table,passive_logs, privileged_logs = train_q_learning(
            observation_corruption=settings["observation_corruption"],
            reward_corruption=settings["reward_corruption"],
            seed=seed,
        )

        passive_df = pd.DataFrame(passive_logs)
        privileged_df = pd.DataFrame(privileged_logs)

        #save passive cv
        passive_df.to_csv(
            f"logs/passive_{condition_name}_seed_{seed}.csv",
            index=False,
        )

        #save priv cv
        privileged_df.to_csv(
            f"logs/privileged_{condition_name}_seed_{seed}.csv",
            index=False,
        )


