from typing import Optional

import gymnasium as gym
from gymnasium import registry

import minihack  # NOQA: F401
from balrog.environments.nle import AutoMore, NLELanguageWrapper

MINIHACK_ENVS = [env_spec.id for env_spec in registry.values() if "MiniHack" in env_spec.id]


def make_minihack_env(env_name, task, config, render_mode: Optional[str] = None):
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
        render_mode=render_mode,
    )
    if skip_more:
        env = AutoMore(env)

    env = NLELanguageWrapper(env, vlm=vlm)

    return env
