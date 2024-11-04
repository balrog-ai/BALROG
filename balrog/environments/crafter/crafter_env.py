from typing import Optional

import crafter

from balrog.environments.crafter import CrafterLanguageWrapper
from balrog.environments.wrappers import GymV21CompatibilityV0


def make_crafter_env(env_name, task, config, render_mode: Optional[str] = None):
    crafter_kwargs = dict(config.envs.crafter_kwargs)
    max_episode_steps = crafter_kwargs.pop("max_episode_steps", 2)

    for param in ["area", "view", "size"]:
        if param in crafter_kwargs:
            crafter_kwargs[param] = tuple(crafter_kwargs[param])

    env = crafter.Env(**crafter_kwargs)
    env = CrafterLanguageWrapper(env, task, max_episode_steps=max_episode_steps)
    env = GymV21CompatibilityV0(env=env, render_mode=render_mode)

    return env
