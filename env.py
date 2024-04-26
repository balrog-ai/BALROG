# Super minimized version of what diff_history does

# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from nle_language_wrapper import NLELanguageWrapper
import difflib
from prompt_builder import DiffPromptBuilder, ConcatPromptBuilder
from progress import Progress


# what call this?
class NLEExtendedLanguageWrapper(NLELanguageWrapper):
    def __init__(
        self,
        env,
        *,
        max_history=None,
        max_length=None,
        use_diff_history=False,
        action_token="<|action|>",
        obs_token="<|observation|>",
    ):
        super().__init__(env)
        action_names = [
            action_strs[0]
            for action, action_strs in self.all_nle_action_map.items()
            if action in env.actions
        ]
        prefix = "You are an agent playing NetHack. Predict the next keypresses.\n\n"
        prefix += (
            f"Output only one of the following actions:\n\n"
            + ", ".join(action_names)
            + "\n\n"
        )
        if use_diff_history:
            self._prompt_builder = DiffPromptBuilder(
                max_history=max_history,
                max_length=max_length,
                prefix=prefix,
                action_token=action_token,
                obs_token=obs_token,
            )
        else:
            self._prompt_builder = ConcatPromptBuilder(
                max_history=max_history,
                max_length=max_length,
                prefix=prefix,
                action_token=action_token,
                obs_token=obs_token,
            )
        self._progress = Progress()

    # override
    def pre_reset(self):
        self._prompt_builder.reset()
        self._progress.reset()
        return super().pre_reset()

    # override
    def post_reset(self, nle_obsv):
        obsv = super().post_step(nle_obsv)
        self._prompt_builder.append_observation(obsv)
        return self._prompt_builder.get_prompt()

    # override
    def pre_step(self, action):
        self._prompt_builder.append_action(action)
        return super().pre_step(action)

    # override
    def post_step(self, nle_obsv):
        obsv = super().post_step(nle_obsv)
        self._prompt_builder.append_observation(obsv)
        return self._prompt_builder.get_prompt()

    # override
    def step(self, action):
        obs, reward, done, info = super().step(action)
        info["progress"] = self._progress.get_progress()
        info["highest_achievement"] = self._progress.get_highest_achievement()
        info["achievements"] = self._progress.get_achievements()
        return obs, reward, done, info


if __name__ == "__main__":
    from nle.env import tasks

    base_env = tasks.NetHackChallenge(
        **dict(
            # savedir="./experiment_outputs/dummy_ttyrec",
            character="@",
            max_episode_steps=100000000,
            observation_keys=(
                "blstats",
                "tty_chars",
                "tty_cursor",
                "glyphs",
                "inv_strs",
                "inv_letters",
            ),
            penalty_step=0.0,
            penalty_time=0.0,
            penalty_mode="constant",
            no_progress_timeout=100,
            # save_ttyrec_every=1,
        )
    )

    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    # model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto")
    # pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=100000, return_full_text=False)

    env = NLEExtendedLanguageWrapper(base_env, use_diff_history=False)

    obs = env.reset()

    for n_steps in range(100000):
        print(obs)
        # action = pipe(obs, max_length=1000)[0]["generated_text"]
        obs, reward, done, info = env.step("north")
        print(info)
        # sleep(0.1)
        quit()
