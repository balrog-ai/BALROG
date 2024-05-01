import numpy as np
from gym import spaces
from nle.nethack import tty_render
from nle_language_wrapper import NLELanguageWrapper


class NLEHybridWrapper(NLELanguageWrapper):
    def __init__(self, env, use_language_action=True):
        super().__init__(env, use_language_action)
        self.observation_space = spaces.Space()

    def nle_process_obsv(self, nle_obsv):
        nle_obsv["prompt"] = super().nle_hybrid_obsv(nle_obsv)
        return nle_obsv
