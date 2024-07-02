import numpy as np
from gym import spaces
from nle_language_wrapper import NLELanguageWrapper


class NLERawWrapper(NLELanguageWrapper):
    def __init__(self, env, use_language_action=True):
        super().__init__(env, use_language_action)
        self.observation_space = spaces.Space()

    def nle_process_obsv(self, nle_obsv):

        text_obsv = super().nle_obsv_to_language(nle_obsv)
        render = super().ascii_render(nle_obsv["tty_chars"])
        inventory = text_obsv["text_inventory"]
        cursor = np.array2string(nle_obsv["tty_cursor"])

        return render, inventory, cursor
