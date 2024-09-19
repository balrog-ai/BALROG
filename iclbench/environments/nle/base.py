import nle_language_wrapper
from nle.nethack import USEFUL_ACTIONS
from PIL import Image

from iclbench.environments import Strings

from .progress import get_progress_system
from .render import tty_render_image
from .render_rgb import rgb_render_image
from .utils import render_hybrid, render_text


class NLELanguageWrapper(nle_language_wrapper.NLELanguageWrapper):
    def __init__(self, env, seed=None, vlm=False):
        super().__init__(env, use_language_action=True)
        self.language_action_space = self.create_action_space()
        if seed is not None:
            self.env.seed(seed)
        self.vlm = vlm

        if not vlm:
            self.prompt_mode = "hybrid"
        else:
            self.prompt_mode = "language"

        self.env = env
        self.progress = get_progress_system(self.env)
        self.max_steps = self.env._max_episode_steps

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
        img = Image.fromarray(self.render("tiles")).convert("RGB") if self.vlm else None
        text = self.nle_obsv_to_language(nle_obsv)

        return {
            "text": text,
            "image": img,
            "obs": nle_obsv,
        }

    def nle_obsv_to_language(self, nle_obsv):
        if self.prompt_mode == "language":
            return render_text(nle_obsv)
        elif self.prompt_mode == "hybrid":
            return render_hybrid(nle_obsv)
        else:
            raise ValueError(f'"{self.prompt_mode}" is not a valid prompt mode.')

    def render(self, mode="human"):
        if mode == "tiles":
            obs = self.env.last_observation
            glyphs = obs[self.env._observation_keys.index("glyphs")]
            return rgb_render_image(glyphs)
        elif mode == "tty_image":
            obs = self.env.last_observation
            tty_chars = obs[self.env._observation_keys.index("tty_chars")]
            tty_colors = obs[self.env._observation_keys.index("tty_colors")]
            return tty_render_image(tty_chars, tty_colors)
        else:
            return super().render(mode)

    def get_stats(self):
        return self.progress.__dict__

    def create_action_space(self):
        nle_actions = [
            action_strs[0]
            for action, action_strs in NLELanguageWrapper.all_nle_action_map.items()
            if action in USEFUL_ACTIONS
        ]
        single_chars = [chr(i) for i in range(ord("a"), ord("z") + 1)] + [chr(i) for i in range(ord("A"), ord("Z") + 1)]
        single_digits = [str(i) for i in range(10)]
        double_digits = [f"{i:02d}" for i in range(100)]

        all_actions = nle_actions + single_chars + single_digits + double_digits
        return Strings(all_actions)
