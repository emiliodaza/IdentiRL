import gymnasium as gym
from gymnasium import spaces

ENV_VERSION = "tiny-one-step-v1"

"""temporary - env and test_env needs proper mathematical formulation before writing completely"""
class TinyDiagnosticPOMDP(gym.Env):
    """
    A one-step POMDP for studying reward misspecification
    and observation insufficiency.

    Latent states:
        0: action 0 is correct
        1: action 1 is correct

    Actions:
        0: choose action 0
        1: choose action 1

    Conditions:
        observation_corruption=False, reward_corruption=False
            Neither failure

        observation_corruption=False, reward_corruption=True
            Reward failure

        observation_corruption=True, reward_corruption=False
            Observation failure

        observation_corruption=True, reward_corruption=True
            Both failures
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        observation_corruption=False,
        reward_corruption=False,
        render_mode=None,
    ):
        super().__init__()

        self.observation_corruption = observation_corruption
        self.reward_corruption = reward_corruption
        self.render_mode = render_mode

        # Two available actions: 0 or 1.
        self.action_space = spaces.Discrete(2)

        # Keep the observation space consistent across all conditions.
        # In the corrupted condition, both latent states produce observation 0.
        self.observation_space = spaces.Discrete(2)

        # The true state is hidden from the agent.
        self.state = None

        self.last_action = None
        self.last_intended_reward = None
        self.last_proxy_reward = None

    def _get_observation(self):
        """
        Convert the latent state into the observation shown to the agent.
        """
        if self.observation_corruption:
            return 0

        return self.state

    def _get_intended_reward(self, action):
        """
        The intended objective rewards choosing the action that matches
        the latent state.
        """
        return float(action == self.state)

    def _get_proxy_reward(self, action):
        """
        The misspecified proxy rewards action 0 regardless of the state.
        """
        return float(action == 0)

    def _get_info(self):
        """
        Privileged information for debugging and evaluation.

        Do not include these fields in passive diagnostic logs.
        """
        return {
            "environment_version": ENV_VERSION,
            "latent_state": self.state,
            "intended_reward": self.last_intended_reward,
            "proxy_reward": self.last_proxy_reward,
            "observation_corruption": self.observation_corruption,
            "reward_corruption": self.reward_corruption,
        }

    def reset(self, seed=None, options=None):
        """
        Begin a new one-step episode.
        """
        super().reset(seed=seed)

        # Randomly select latent state 0 or 1.
        self.state = int(self.np_random.integers(0, 2))

        self.last_action = None
        self.last_intended_reward = None
        self.last_proxy_reward = None

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        """
        Apply one action and end the episode.
        """
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action {action}. Expected 0 or 1."
            )

        if self.state is None:
            raise RuntimeError(
                "You must call reset() before step()."
            )

        self.last_action = int(action)

        intended_reward = self._get_intended_reward(action)
        proxy_reward = self._get_proxy_reward(action)

        self.last_intended_reward = intended_reward
        self.last_proxy_reward = proxy_reward

        if self.reward_corruption:
            training_reward = proxy_reward
        else:
            training_reward = intended_reward

        # This environment has one action per episode.
        terminated = True
        truncated = False

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return (
            observation,
            training_reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        observation = self._get_observation()

        print(
            f"state={self.state}, "
            f"observation={observation}, "
            f"action={self.last_action}, "
            f"intended_reward={self.last_intended_reward}, "
            f"proxy_reward={self.last_proxy_reward}"
        )

    def close(self):
        pass