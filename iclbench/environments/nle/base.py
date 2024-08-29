from gym import spaces
import nle_language_wrapper
from nle.nethack import USEFUL_ACTIONS

from iclbench.environments import Strings
from .render import tty_render_image
from .render_rgb import rgb_render_image
from .utils import render_ascii_map, render_text, render_hybrid
from .progress import get_progress_system


class NLELanguageWrapper(nle_language_wrapper.NLELanguageWrapper):
    def __init__(self, env, prompt_mode="tty", seed=None):
        super().__init__(env, use_language_action=True)
        self.prompt_mode = prompt_mode
        self.observation_space = spaces.Space()
        self.language_action_space = self.create_action_space()
        if seed is not None:
            self.env.seed(seed)

        self.env = env
        self.progress = get_progress_system(self.env)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.progress.update(obs["obs"], reward, done, info)
        return obs, reward, done, info

    def reset(self):
        self.progress = get_progress_system(self.env)
        return super().reset()

    @property
    def default_action(self):
        return "esc"

    def nle_process_obsv(self, nle_obsv):
        return self.nle_obsv_to_language(nle_obsv)

    def nle_obsv_to_language(self, nle_obsv):
        if self.prompt_mode == "ascii_map":
            return {"obs": nle_obsv, "text": render_ascii_map(nle_obsv)}
        elif self.prompt_mode == "language":
            return {"obs": nle_obsv, "text": render_text(nle_obsv)}
        elif self.prompt_mode == "hybrid":
            return {"obs": nle_obsv, "text": render_hybrid(nle_obsv)}
        else:
            raise ValueError(f'"{self.prompt_mode}" is not a valid prompt mode.')

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

    def get_stats(self):
        return self.progress.__dict__

    # def default_system_prompt(self):
    #     ACTION_NAMES = [
    #         action_strs[0]
    #         for action, action_strs in self.all_nle_action_map.items()
    #         if action in ACTIONS
    #     ]
    #     ACTIONS_LIST_STR = ",\n".join(ACTION_NAMES)
    #     INSTRUCTION_PROMPT = f"""
    #     You are an agent playing NetHack. In a moment I will present you an observation. Only output an action from the following list:
    #     {ACTIONS_LIST_STR}.

    #     You can only output one action at a time. The goal is to maximize the reward.
    #     """.strip()
        
    #     return INSTRUCTION_PROMPT

    def create_action_space(self):
        nle_actions = [
            action_strs[0]
            for action, action_strs in NLELanguageWrapper.all_nle_action_map.items()
            if action in USEFUL_ACTIONS
        ]
        single_chars = [chr(i) for i in range(ord("a"), ord("z") + 1)] + [
            chr(i) for i in range(ord("A"), ord("Z") + 1)
        ]
        single_digits = [str(i) for i in range(10)]
        double_digits = [f"{i:02d}" for i in range(100)]

        all_actions = nle_actions + single_chars + single_digits + double_digits
        return Strings(all_actions)
