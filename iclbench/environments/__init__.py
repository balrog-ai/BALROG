# Here we should have an environment manager function that can be used to instantiate
# environments with the correct wrappers.
import random

import gym
import numpy as np
from gym import spaces

from iclbench.environments.env_wrapper import EnvWrapper


def make_env(env_name, task, config):
    if env_name == "nle":
        from iclbench.environments.nle import NLELanguageWrapper

        nle_kwargs = dict(config.envs.nle_kwargs)
        skip_more = nle_kwargs.pop("skip_more", False)
        vlm = True if config.agent.max_image_history > 0 else False
        env = gym.make(task, **nle_kwargs)
        base_env = NLELanguageWrapper(env, vlm=vlm, skip_more=skip_more)
    elif env_name == "minihack":
        import minihack

        from iclbench.environments.nle import NLELanguageWrapper

        minihack_kwargs = dict(config.envs.minihack_kwargs)
        skip_more = minihack_kwargs.pop("skip_more", False)
        vlm = True if config.agent.max_image_history > 0 else False
        env = gym.make(
            task,
            observation_keys=[
                "glyphs",
                "blstats",
                "tty_chars",
                "inv_letters",
                "inv_strs",
                "tty_cursor",
                "tty_colors",
            ],
            **minihack_kwargs,
        )
        base_env = NLELanguageWrapper(env, vlm=vlm, skip_more=skip_more)
    elif env_name == "babyai":
        import babyai_text

        from iclbench.environments.babyai_text import BabyAITextCleanLangWrapper

        base_task, goal = task.split("/")
        while 1:
            env = gym.make(base_task)
            if env.env.action_kinds[0].replace(" ", "_") == goal:
                break

        base_env = BabyAITextCleanLangWrapper(env, **config.envs.babyai_kwargs)
    elif env_name == "crafter":
        import crafter

        from iclbench.environments.crafter import CrafterLanguageWrapper

        crafter_kwargs = dict(config.envs.crafter_kwargs)
        max_episode_steps = crafter_kwargs.pop("max_episode_steps", 2)

        for param in ["area", "view", "size"]:
            if param in crafter_kwargs:
                crafter_kwargs[param] = tuple(crafter_kwargs[param])

        env = crafter.Env(**crafter_kwargs)
        base_env = CrafterLanguageWrapper(env, task, max_episode_steps=max_episode_steps)
    elif env_name == "craftax":
        from iclbench.environments.craftax import CraftaxLanguageWrapper

        base_env = CraftaxLanguageWrapper(task, **config.envs.craftax_kwargs)
    elif env_name == "textworld":
        from iclbench.environments.textworld import global_textworld_context

        textworld_context = global_textworld_context(tasks=config.tasks.textworld_tasks, **config.envs.textworld_kwargs)
        base_env = textworld_context(task, **config.envs.env_kwargs)
    elif env_name == "babaisai":
        from baba import make

        from iclbench.environments.baba_is_ai import BabaIsAIWrapper

        base_env = BabaIsAIWrapper(make(task, **config.envs.babaisai_kwargs))
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    return EnvWrapper(base_env, env_name, task)


class Strings(spaces.Space):
    def __init__(self, values, seed=None):
        super().__init__((len(values),), str, seed)
        self._dict = {value: i for i, value in enumerate(values)}
        self._values = values

    def sample(self):
        return self.np_random.choice(self._values)

    def map(self, action):
        return self._dict[action]

    def contains(self, value):
        return value in self._dict

    def __iter__(self):
        return self._values.__iter__()
