from pathlib import Path
from typing import Optional

from baba import make

from iclbench.environments.wrappers import GymV21CompatibilityV0
from iclbench.environments.babaisai import BabaIsAIWrapper


def make_babaisai_env(env_name, task, config, render_mode: Optional[str] = None):
    env = make(task, **config.envs.babaisai_kwargs)
    env = BabaIsAIWrapper(env)
    env = GymV21CompatibilityV0(env=env, render_mode=render_mode)

    return env
