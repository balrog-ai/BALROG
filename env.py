# Super minimized version of what diff_history does

from nle_language_wrapper import NLELanguageWrapper
from prompt_builder import DiffPromptBuilder, ConcatPromptBuilder, SimpleQAPromptBuilder, nle_text_obs
from progress import Progress

# what call this?
class NLEExtendedLanguageWrapper(NLELanguageWrapper):
    def __init__(self, env, *, max_history=None, max_length=None, use_diff_history=False, action_token="<|action|>", obs_token="<|observation|>"):
        super().__init__(env)
        action_names = [action_strs[0] for action, action_strs in self.all_nle_action_map.items() if action in env.actions]
        prefix = "You are an agent playing NetHack. Predict the next keypresses.\n\n"
        if use_diff_history:
            prefix += f"Output only one of the following actions:\n\n" + ", ".join(action_names) + "\n\n"
            self._prompt_builder = DiffPromptBuilder(max_history=max_history, max_length=max_length, prefix=prefix, action_token=action_token, obs_token=obs_token)
        else:
            # self._prompt_builder = SimpleQAPromptBuilder(max_history=max_history, max_length=max_length, prefix=prefix)
            self._prompt_builder = ConcatPromptBuilder(max_history=max_history, max_length=max_length, prefix=prefix, action_token=action_token, obs_token=obs_token)
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
    
    def history(self):
        text = ""
        for obs, action in zip(self._prompt_builder._obs_history[:-1], self._prompt_builder._action_history):
            text += f"OBSERVATION\n===========\n" + obs + "\n\nACTION\n======\n" + action + "\n\n"
        # text += f"OBSERVATION:\n" + self._prompt_builder._obs_history[-1]
        text += "GAMEOVER"
        return text