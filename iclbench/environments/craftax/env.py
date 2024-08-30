import gym
import numpy as np
import jax
import jax.numpy as jnp
import craftax
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.renderer import render_craftax_pixels, render_craftax_text
from craftax.craftax.constants import (
    OBS_DIM,
    BLOCK_PIXEL_SIZE_HUMAN,
    INVENTORY_OBS_HEIGHT,
)

from iclbench.environments import Strings

USEFUL_ACTION = [
    "noop",
    "up",
    "right",
    "down",
    "left",
    "do",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_diamond_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
    "make_diamond_sword",
    "place_table",
    "sleep",
    "place_stone",
    "place_furnace",
    "place_plant",
    "rest",
    "ascend",
    "descend",
    "make_iron_armour",
    "make_diamond_armour",
    "shoot_arrow",
    "make_arrow",
    "cast_fireball",
    "cast_iceball",
    "place_torch",
    "drink_potion_red",
    "drink_potion_green",
    "drink_potion_blue",
    "drink_potion_pink",
    "drink_potion_cyan",
    "drink_potion_yellow",
    "read_book",
    "enchant_sword",
    "enchant_armour",
    "make_torch",
    "level_up_dexterity",
    "level_up_strength",
    "level_up_intelligence",
    "enchant_bow",
]


class CraftaxLanguageWrapper(gym.Env):
    def __init__(self, env_id: str = "Craftax-Symbolic-v1", seed=None):
        super(CraftaxLanguageWrapper, self).__init__()

        env = make_craftax_env_from_name(env_id, auto_reset=True)
        self._step = jax.jit(env.step)
        self._reset = jax.jit(env.reset)
        self._render = jax.jit(render_craftax_pixels)
        self._env_params = env.default_params

        if seed is None:
            seed = np.random.randint(2**31)
        self._rng = jax.random.PRNGKey(seed)
        self._env_state = None

        self.language_action_space = Strings(USEFUL_ACTION)

    @property
    def default_action(self):
        return "noop"

    def reset(self):
        # Reset the state of the environment to an initial state
        self._rng, _rng = jax.random.split(self._rng)
        obs, self._env_state = self._reset(_rng, self._env_params)
        obs = {"obs": obs, "text": (render_craftax_text(self._env_state), "")}
        return obs

    def step(self, language_action):
        if language_action not in self.language_action_space:
            raise ValueError(
                f"Action {repr(language_action)} not recognized / supported by this environment."
            )
        action = jnp.array(self.language_action_space.map(language_action))
        self._rng, _rng = jax.random.split(self._rng)
        obs, self._env_state, reward, done, info = self._step(
            _rng, self._env_state, action, self._env_params
        )
        # To decide whether craftax has long and short term context observations
        obs = {"obs": obs, "text": (render_craftax_text(self._env_state), "")}
        return obs, reward.item(), done, info

    def render(self, mode="human"):
        return np.array(
            self._render(self._env_state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)
        )

    def get_stats(self):
        # TODO: convert to string list rather than bool list
        achievements = list(map(int, np.array(self._env_state.achievements)))
        return {"achievements": achievements}
