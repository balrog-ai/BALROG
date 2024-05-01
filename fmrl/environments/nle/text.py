import numpy as np
from gym import spaces
from nle.nethack import tty_render
from nle_language_wrapper import NLELanguageWrapper


class NLETextWrapper(NLELanguageWrapper):
    def __init__(self, env, use_language_action=True):
        super().__init__(env, use_language_action)
        self.observation_space = spaces.Space()

    def nle_process_obsv(self, nle_obsv):
        key_name_pairs = [
            ("text_blstats", "statistics"),
            ("text_glyphs", "glyphs"),
            ("text_message", "message"),
            ("text_inventory", "inventory"),
            ("text_cursor", "cursor"),
        ]
        text_obsv = super().nle_obsv_to_language(nle_obsv)
        nle_obsv["prompt"] = "\n".join(
            [f"{name}[\n{text_obsv[key]}\n]" for key, name in key_name_pairs]
        )
        return nle_obsv
