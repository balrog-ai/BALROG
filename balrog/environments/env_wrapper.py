import gymnasium as gym


class EnvWrapper(gym.Wrapper):
    """
    A wrapper class for gym environments to standardize interactions across different environments.
    It provides additional functionalities, such as handling specific observation processing,
    managing action validity, retrieving instruction prompts, and tracking failed action candidates.
    """

    def __init__(self, env, env_name, task_name):
        super().__init__(env)
        self.env_name = env_name
        self.task_name = task_name
        self.failed_candidates = []

    @property
    def max_steps(self):
        return self.env.max_steps

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed_obs = self._process_observation(obs)
        return processed_obs, reward, terminated, truncated, info

    def _process_observation(self, obs):
        if self.env_name in ["nle", "minihack"]:
            obs = obs
        elif self.env_name == "babyai":
            obs = obs
        elif self.env_name == "textworld":
            obs = obs
        elif self.env_name == "babaisai":
            obs = obs
        elif self.env_name == "crafter":
            obs = obs
        else:
            raise ValueError(f"Unknown environment: {self.env_name}")

        return obs

    @property
    def actions(self):
        # This property should return the list of available actions
        return self.env.actions if hasattr(self.env, "actions") else list(range(len(self.env.action_space)))

    def get_text_action(self, action):
        return self.env.get_text_action(action)

    def get_instruction_prompt(self, instructions=None):
        if self.env_name == "nle":
            from balrog.environments.nle import get_instruction_prompt

            return get_instruction_prompt()
        elif self.env_name == "minihack":
            from balrog.environments.minihack import get_instruction_prompt

            return get_instruction_prompt(self.env, self.task_name)
        elif self.env_name == "babyai":
            from balrog.environments.babyai_text import get_instruction_prompt

            return get_instruction_prompt(self.env, mission=instructions)
        elif self.env_name == "textworld":
            from balrog.environments.textworld import get_instruction_prompt

            return get_instruction_prompt(self.env, self.task_name)
        elif self.env_name == "babaisai":
            from balrog.environments.babaisai import get_instruction_prompt

            return get_instruction_prompt(self.env, self.task_name)
        elif self.env_name == "crafter":
            from balrog.environments.crafter import get_instruction_prompt

            return get_instruction_prompt(self.task_name)
        else:
            raise ValueError(f"Unknown environment: {self.env_namee}")

    def check_action_validity(self, candidate_action):
        valid_action = None
        if candidate_action in self.env.language_action_space:
            valid_action = candidate_action
        else:
            valid_action = self.env.default_action
            self.failed_candidates.append(candidate_action)
        return valid_action

    def get_stats(self):
        return self.env.get_stats()
