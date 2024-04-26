import numpy as np
from gym import spaces
from nle.nethack import tty_render
from nle_language_wrapper import NLELanguageWrapper


class NLEAsciiWrapper(NLELanguageWrapper):
    def __init__(self, env, use_language_action=True):
        super().__init__(env, use_language_action)
        self.observation_space = spaces.Space()

    def ascii_render(self, chars):
        rows, cols = chars.shape
        result = ""
        for i in range(rows):
            result += "\n"
            for j in range(cols):
                entry = chr(chars[i, j])
                result += entry
        return result

    def nle_process_obsv(self, nle_obsv):
        # This is not really ASCII, it's ASCII with ANSI color codes
        nle_obsv["prompt"] = self.ascii_render(nle_obsv["tty_chars"])
        return nle_obsv
