import gym
import numpy as np
import jax
import jax.numpy as jnp
import craftax
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.renderer import render_craftax_pixels, render_craftax_text
from functools import partial
from craftax.craftax.constants import (
    OBS_DIM,
    BLOCK_PIXEL_SIZE_HUMAN,
    INVENTORY_OBS_HEIGHT,
    Action,
    Achievement,
)
from iclbench.environments.spaces import Strings

ALL_CRAFTAX_ACTION_MAP = {
    Action.NOOP: ["noop", "q"],
    Action.UP: ["up", "w"],
    Action.RIGHT: ["right", "d"],
    Action.DOWN: ["down", "s"],
    Action.LEFT: ["left", "a"],
    Action.DO: ["do", " "],
    Action.MAKE_WOOD_PICKAXE: ["make_wood_pickaxe", "1"],
    Action.MAKE_STONE_PICKAXE: ["make_stone_pickaxe", "2"],
    Action.MAKE_IRON_PICKAXE: ["make_iron_pickaxe", "3"],
    Action.MAKE_DIAMOND_PICKAXE: ["make_diamond_pickaxe", "4"],
    Action.MAKE_WOOD_SWORD: ["make_wood_sword", "5"],
    Action.MAKE_STONE_SWORD: ["make_stone_sword", "6"],
    Action.MAKE_IRON_SWORD: ["make_iron_sword", "7"],
    Action.MAKE_DIAMOND_SWORD: ["make_diamond_sword", "8"],
    Action.PLACE_TABLE: ["place_table", "t"],
    Action.SLEEP: ["sleep", "\t"],
    Action.PLACE_STONE: ["place_stone", "r"],
    Action.PLACE_FURNACE: ["place_furnace", "f"],
    Action.PLACE_PLANT: ["place_plant", "p"],
    Action.REST: ["rest", "e"],
    Action.ASCEND: ["ascend", ","],
    Action.DESCEND: ["descend", "."],
    Action.MAKE_IRON_ARMOUR: ["make_iron_armour", "y"],
    Action.MAKE_DIAMOND_ARMOUR: ["make_diamond_armour", "u"],
    Action.SHOOT_ARROW: ["shoot_arrow", "i"],
    Action.MAKE_ARROW: ["make_arrow", "o"],
    Action.CAST_FIREBALL: ["cast_fireball", "g"],
    Action.CAST_ICEBALL: ["cast_iceball", "h"],
    Action.PLACE_TORCH: ["place_torch", "j"],
    Action.DRINK_POTION_RED: ["drink_potion_red", "z"],
    Action.DRINK_POTION_GREEN: ["drink_potion_green", "x"],
    Action.DRINK_POTION_BLUE: ["drink_potion_blue", "c"],
    Action.DRINK_POTION_PINK: ["drink_potion_pink", "v"],
    Action.DRINK_POTION_CYAN: ["drink_potion_cyan", "b"],
    Action.DRINK_POTION_YELLOW: ["drink_potion_yellow", "n"],
    Action.READ_BOOK: ["read_book", "m"],
    Action.ENCHANT_SWORD: ["enchant_sword", "k"],
    Action.ENCHANT_ARMOUR: ["enchant_armour", "l"],
    Action.MAKE_TORCH: ["make_torch", "("],
    Action.LEVEL_UP_DEXTERITY: ["level_up_dexterity", ")"],
    Action.LEVEL_UP_STRENGTH: ["level_up_strength", "-"],
    Action.LEVEL_UP_INTELLIGENCE: ["level_up_intelligence", "="],
    Action.ENCHANT_BOW: ["enchant_bow", ";"],
}

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
        
        self.language_action_space = Strings(
            [action_strs[0] for action, action_strs in ALL_CRAFTAX_ACTION_MAP.items()]
        )
        
    @property
    def default_action(self):
        return "noop"

    def reset(self):
        # Reset the state of the environment to an initial state
        self._rng, _rng = jax.random.split(self._rng)
        obs, self._env_state = self._reset(_rng, self._env_params)
        obs = {"obs": obs, "text": render_craftax_text(self._env_state)}
        return obs

    def step(self, language_action):
        if language_action not in self.language_action_space:
            raise ValueError(f"Action {repr(language_action)} not recognized / supported by this environment.")
        action = jnp.array(self.language_action_space.map(language_action))
        self._rng, _rng = jax.random.split(self._rng)
        obs, self._env_state, reward, done, info = self._step(_rng, self._env_state, action, self._env_params)
        obs = {"obs": obs, "text": render_craftax_text(self._env_state)}
        return obs, reward.item(), done, info

    def render(self, mode='human'):
        return np.array(self._render(self._env_state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN))
    
    def get_stats(self):
        # TODO: convert to string list rather than bool list
        achievements = list(map(int, np.array(self._env_state.achievements)))
        return {
            "achievements": achievements
        }