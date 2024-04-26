# Super minimized version of what diff_history does

from nle_language_wrapper import NLELanguageWrapper
from prompt.prompt_builder import (
    DiffPromptBuilder,
    ConcatPromptBuilder,
    SimpleQAPromptBuilder,
    nle_text_obs,
)
from progress import Progress


# what call this?
class NLEExtendedLanguageWrapper(NLELanguageWrapper):
    def __init__(self, env, *, prompt_builder=None, max_history=None, max_length=None):
        super().__init__(env)
        if prompt_builder is None:
            self._prompt_builder = ConcatPromptBuilder(
                max_history=max_history, max_length=max_length
            )
        self._progress = Progress()

    # override
    def pre_reset(self):
        self._prompt_builder.reset()
        self._progress.reset()
        return super().pre_reset()

    # override
    def post_reset(self, nle_obsv):
        obsv = nle_text_obs(super().post_step(nle_obsv))
        self._prompt_builder.append_observation(obsv)
        return self._prompt_builder.get_prompt()

    # override
    def pre_step(self, action):
        self._prompt_builder.append_action(action)
        return super().pre_step(action)

    # override
    def post_step(self, nle_obsv):
        obsv = super().post_step(nle_obsv)
        self._progress.update(obsv["text_message"], obsv["text_blstats"])
        obsv = nle_text_obs(obsv)
        self._prompt_builder.append_observation(obsv)
        return self._prompt_builder.get_prompt()

    # override
    def step(self, action):
        obs, reward, done, info = super().step(action)
        info["progress"] = self._progress.get_progress()
        info["highest_achievement"] = self._progress.get_highest_achievement()
        info["achievements"] = self._progress.get_achievements()
        return obs, reward, done, info
