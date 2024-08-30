import os
import glob
import importlib.resources
from collections import defaultdict
from pathlib import Path

import gym
import textworld
import textworld.gym


nle_utils_dir = os.path.dirname(importlib.resources.files("iclbench").__str__())
TEXTWORLD_GAMES_PATH = os.path.join(nle_utils_dir, "iclbench/environments/textworld/tw_games")

TASKS = [
    "treasure_hunter",
    "the_cooking_game",
    "coin_collector",
]


class TextWorldFactory:
    """
    A factory class for creating TextWorld environments.

    This class manages the creation of TextWorld environments for different tasks,
    cycling through available games for each task or allowing specific game selection.
    """
    _instance = None

    def __new__(cls, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize(**kwargs)
        return cls._instance
    
    def initialize(self, **kwargs):
        self.count = defaultdict(int)
        
        assert "objective" in kwargs and "description" in kwargs and kwargs["objective"] and kwargs["description"], "objective and description parameters are required."
        self.request_infos = textworld.EnvInfos(**kwargs)
        
        self.env_ids = defaultdict(list)
        for pattern in ["*.ulx", "*.z8"]:
            for entry in glob.glob(os.path.join(TEXTWORLD_GAMES_PATH, f"**/{pattern}"), recursive=True):
                task = Path(entry).parent.name
                if task in TASKS:
                    env_id = textworld.gym.register_game(entry, self.request_infos)
                    self.env_ids[task].append(env_id)

    def get_textworld_env(self, task, prompt_mode="language", seed=None, **kwargs):
        """
        Create and return a TextWorld environment for the specified task.

        Args:
            task (str): The name of the task for which to create an environment.
            seed (int, optional): If provided, creates the environment for the
                                      specific game index. If None, cycles through
                                      available games.

        Returns:
            gym.Env: A TextWorld gym environment.

        Raises:
            KeyError: If the specified task is not found in the available tasks.
        """
        if task not in self.env_ids:
            raise KeyError(f"Task '{task}' not found. Available tasks are: {list(self.env_ids.keys())}")
        
        if seed is not None:
            env_id = seed % len(self.env_ids[task])
        else:
            self.count[task] += 1
            env_id = self.env_ids[task][self.count[task] % len(self.env_ids[task])]
        
        env = textworld.gym.make(env_id, **kwargs)
        env = TextWorldWrapper(env, prompt_mode=prompt_mode)
        return env

    def __call__(self, task, **kwargs):
        return self.get_textworld_env(task, **kwargs)


class AlwaysTrue:
    def __contains__(self, item):
        return True


class TextWorldWrapper(gym.Wrapper):
    def __init__(self, env, prompt_mode="language"):
        super().__init__(env)
        self.prompt_mode = prompt_mode
        self.language_action_space = AlwaysTrue()

    def textworld_process_obsv(self, textworld_obsv):
        if self.prompt_mode == "language":
            return {"text": (textworld_obsv, "")}
        else:
            raise ValueError(f'"{self.prompt_mode}" is not a valid prompt mode.')
        
    def filter_objective(self, obs, info):
        objective = info["objective"]
        parts = obs.split(objective)
        if len(parts) == 1:
            return parts[0].strip()
        else:
            return parts[-1].strip()

    def reset(self):
        obs, info = self.env.reset()
        obs = self.filter_objective(obs, info)
        
        return self.textworld_process_obsv(obs)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.filter_objective(obs, info)

        return self.textworld_process_obsv(obs), reward, done, info
