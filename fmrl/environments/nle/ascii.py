import numpy as np
from gym import spaces
from nle.nethack import tty_render
from nle_language_wrapper import NLELanguageWrapper


class NLEAsciiWrapper(NLELanguageWrapper):
    def __init__(self, env, use_language_action=True):
        super().__init__(env, use_language_action)
        self.observation_space = spaces.Space()

    def nle_process_obsv(self, nle_obsv):
        # This is not really ASCII, it's ASCII with ANSI color codes
        message, nle_obsv = super().clean_message(nle_obsv)

        ascii_map = super().ascii_render(nle_obsv["tty_chars"])
        ascii_map = ascii_map.split("\n")
        ascii_map[0] = message
        ascii_map = "\n".join(ascii_map)

        nle_obsv["prompt"] = ascii_map
        return nle_obsv
