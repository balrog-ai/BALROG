from gym import spaces
import nle_language_wrapper
from nle.nethack import ACTIONS

from fmrl.environments.spaces import Strings
from .render import tty_render_image
from .render_rgb import rgb_render_image
from .utils import render_tty, render_text, render_hybrid

class NLELanguageWrapper(nle_language_wrapper.NLELanguageWrapper):
    def __init__(self, env, prompt_mode="tty"):
        super().__init__(env, use_language_action=True)
        self.prompt_mode = prompt_mode
        self.observation_space = spaces.Space() # TODO: dict here
        self.language_action_space = Strings([
            action_strs[0]
            for action, action_strs in NLELanguageWrapper.all_nle_action_map.items()
            if action in ACTIONS
        ])
        
    # def nle_obsv_to_language(self, nle_obsv):
    def nle_process_obsv(self, nle_obsv): # why the name change?
        if self.prompt_mode == "tty":
            return {"obs": nle_obsv, "text": render_tty(nle_obsv)}
        elif self.prompt_mode == "language":
            return {"obs": nle_obsv, "text": render_text(nle_obsv)}
        elif self.prompt_mode == "hybrid":
            return {"obs": nle_obsv, "text": render_hybrid(nle_obsv)}
        else:
            raise ValueError(f"\"{self.prompt_mode}\" is not a valid prompt mode.")
        
    # def step(..):
    #     obs, ... = super().step(...)
    #     info["progress"] = ...
    #     return ...
        
    def render(self, mode="human"):
        if mode == "tty_image":
            obs = self.env.last_observation            
            glyphs = obs[self.env._observation_keys.index("glyphs")]
            return rgb_render_image(glyphs)
        elif mode == "image":
            obs = self.env.last_observation
            tty_chars = obs[self.env._observation_keys.index("tty_chars")]
            tty_colors = obs[self.env._observation_keys.index("tty_colors")]
            # tty_cursor = obs[self.env._observation_keys.index("tty_cursor")]
            return tty_render_image(tty_chars, tty_colors)
        else:
            return super().render(mode)
            
        