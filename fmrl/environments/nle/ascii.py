import numpy as np
from gym import spaces
from nle.nethack import tty_render
from nle_language_wrapper import NLELanguageWrapper

class NLEAsciiWrapper(NLELanguageWrapper):
    def __init__(self, env, use_language_action=True):
        super().__init__(env, use_language_action)
        self.observation_space = spaces.Space()
        
    def nle_obsv_to_language(self, nle_obsv):
        nle_obsv["prompt"] = tty_render(nle_obsv["tty_chars"], nle_obsv["tty_colors"], nle_obsv["tty_cursor"])
        return nle_obsv