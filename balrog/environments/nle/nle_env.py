from typing import Optional

import gym
import nle  # NOQA: F401

from balrog.environments.nle import NLELanguageWrapper
from balrog.environments.wrappers import GymV21CompatibilityV0, NLETimeLimit

NETHACK_ENVS = []
for env_spec in gym.envs.registry.all():
    id = env_spec.id
    if "NetHack" in id:
        NETHACK_ENVS.append(id)


def make_nle_env(env_name, task, config, render_mode: Optional[str] = None):
    nle_kwargs = dict(config.envs.nle_kwargs)
    skip_more = nle_kwargs.pop("skip_more", False)
    vlm = True if config.agent.max_image_history > 0 else False
    env = gym.make(task, **nle_kwargs)
    env = NLELanguageWrapper(env, vlm=vlm, skip_more=skip_more)

    # wrap NLE with timeout
    env = NLETimeLimit(env)

    env = GymV21CompatibilityV0(env=env, render_mode=render_mode)

    return env
