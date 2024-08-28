import logging
import numpy as np


class EnvWrapper:
    def __init__(self, env, env_name, task_name):
        self.env = env
        self.env_name = env_name
        self.task_name = task_name

    def reset(self):
        obs = self.env.reset()
        return self._process_observation(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        processed_obs = self._process_observation(obs)
        return processed_obs, reward, done, info

    def _process_observation(self, obs):
        if self.env_name in ["nle", "minihack"]:
            return obs
        elif self.env_name == "babyai":
            raise NotImplementedError("BabyAI environment is not supported yet.")
        elif self.env_name == "craftax":
            raise NotImplementedError("Craftax environment is not supported yet.")
        else:
            raise ValueError(f"Unknown environment: {self.env_name}")

        return obs

    @property
    def actions(self):
        # This property should return the list of available actions
        return (
            self.env.actions
            if hasattr(self.env, "actions")
            else list(range(len(self.env.action_space)))
        )

    def get_instruction_prompt(self):
        if self.env_name == "nle":
            from iclbench.environments.nle import get_instruction_prompt

            return get_instruction_prompt()
        elif self.env_name == "minihack":
            from iclbench.environments.minihack import get_instruction_prompt

            return get_instruction_prompt(self.env, self.task_name)
        elif self.env_name == "babyai":
            raise NotImplementedError("BabyAI environment is not supported yet.")
        elif self.env_name == "craftax":
            raise NotImplementedError("Craftax environment is not supported yet.")

        else:
            raise ValueError(f"Unknown environment: {self.env_namee}")

    def check_action_validity(self, action):
        valid_action = None
        for choice in action.choices:
            candidate_action = (
                choice.text
                if not hasattr(choice, "message")
                else choice.message.content
            )
            if candidate_action in self.env.language_action_space:
                valid_action = candidate_action
                break
        if not valid_action:
            valid_action = self.env.default_action
            logging.warn(
                f'Failed to generate a valid action. Output: "{action.choices}".\
                    Selecting default action "{valid_action}".'
            )
            # self.failed_generation_counter += 1
        return valid_action
