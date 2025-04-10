from pathlib import Path
from typing import Optional

import gym
import gym_battleship

from balrog.environments.battleships import BattleshipsWrapper
from balrog.environments.battleships.place_ship import PlaceShip
from balrog.environments.wrappers import GymV21CompatibilityV0


def make_battleships_env(env_name, task, config, render_mode: Optional[str] = None):
    env = gym.make(task, **config.envs.battleships_kwargs)
    env = PlaceShip(env)
    env = BattleshipsWrapper(env)
    env = GymV21CompatibilityV0(env=env, render_mode=render_mode)

    return env
