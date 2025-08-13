from typing import Optional

import gymnasium as gym
import nle  # NOQA: F401
import nle_progress  # NOQA: F401
from gymnasium import registry

from balrog.environments.nle import AutoMore, NLELanguageWrapper

NETHACK_ENVS = [env_spec.id for env_spec in registry.values() if "NetHack" in env_spec.id]


def make_nle_env(env_name, task, config, render_mode: Optional[str] = None):
    nle_kwargs = dict(config.envs.nle_kwargs)
    skip_more = nle_kwargs.pop("skip_more", False)
    vlm = True if config.agent.max_image_history > 0 else False
    env = gym.make(task, **nle_kwargs)
    if skip_more:
        env = AutoMore(env)

    env = NLELanguageWrapper(env, vlm=vlm)

    return env
