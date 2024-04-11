# Super minimized version of what diff_history does

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from nle_language_wrapper import NLELanguageWrapper
import difflib

# what call this?
class NLEExtendedLanguageWrapper(NLELanguageWrapper):
    def __init__(self, env, *, max_history=None, max_length=None, use_diff_history=False):
        super().__init__(env)
        self._action_names = [action_strs[0] for action, action_strs in self.all_nle_action_map.items() if action in env.actions]
        self._max_history = max_history
        self._max_length = max_length
        self._obs_history = []
        self._action_history = []
        self._format_history = self._diff_history if use_diff_history else self._concat_history

    # override
    def pre_reset(self):
        self._obs_history = []
        self._action_history = []
        return super().pre_reset()

    # override
    def post_reset(self, nle_obsv):
        obsv = self._text_obs_concate(super().post_step(nle_obsv))
        self._obs_history.append(obsv)
        return self._format()
    
    # override
    def pre_step(self, action):
        self._action_history.append(action)
        return super().pre_step(action)
    
    # override
    def post_step(self, nle_obsv):
        obsv = self._text_obs_concate(super().post_step(nle_obsv))
        self._obs_history.append(obsv)
        return self._format()
    
    # ===
    
    def _text_obs_concate(self, text_obsv):
        key_name_pairs = [
            ("text_blstats", "statistics"),
            ("text_glyphs", "glyphs"),
            ("text_message", "message"),
            ("text_inventory", "inventory"),
            ("text_cursor", "cursor"),
        ]
        return "\n".join([f"{name}[\n{text_obsv[key]}\n]" for key, name in key_name_pairs])

    def _format(self):
        n_steps = len(self._action_history)
        start_idx = min(self._max_history or len(self._action_history), n_steps + 1)
        
        if self._max_length is None:
            return self._format_history(self._obs_history[-start_idx-1:], self._action_history[-start_idx:])
        
        for i in reversed(range(1, start_idx+1)):
            text = self._format_history(self._obs_history[-i-1:], self._action_history[-i:])
            num_tokens = len(text.encode('utf-8')) # Ideal world, we know exactly how many tokens this string is, but estimate using num bytes
            if num_tokens <= self._max_length:
                return text
            
        raise ValueError("Unable to generate context that fits within max_length.")

    def _diff_history(self, obs_history, action_history):
        text = "You are an agent playing NetHack. Predict the next keypresses.\n\n"
        text += f"Output only one of the following actions:\n\n" + ", ".join(self._action_names) + "\n\n"
        text += "**OBSERVATION**:\n\n" + obs_history[0] + "\n\n"
        for action, (prev_obs, obs) in zip(action_history, zip(obs_history[:-1], obs_history[1:])):
            prev_obs = prev_obs.strip().splitlines()
            obs = obs.strip().splitlines()
            obs = "\n".join(difflib.unified_diff(prev_obs, obs, n=0, lineterm=""))
            text += "**ACTION**:\n\n" + action + "\n\n" + "**OBSERVATION**:\n\n" + obs + "\n\n"
        text += "**ACTION**:\n\n"
        return text

    def _concat_history(self, obs_history, action_history):
        text = "You are an agent playing NetHack. Predict the next keypresses.\n\n"
        text += f"Output only one of the following actions:\n\n" + ", ".join(self._action_names) + "\n\n"
        for obs, action in zip(obs_history[:-1], action_history):
            text += "**OBSERVATION**:\n\n" + obs + "\n\n" + "**ACTION**:\n\n" + action + "\n\n"
        text += "**OBSERVATION**:\n\n" + obs_history[-1] + "\n\n" + "**ACTION**:\n\n"
        return text

if __name__ == '__main__':
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
    
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=100000, return_full_text=False)

    env = NLEExtendedLanguageWrapper(base_env, use_diff_history=True)
    
    obs = env.reset()
    
    for n_steps in range(100000):
        print(obs)
        action = pipe(obs, max_length=1000)[0]['generated_text']
        if action in env.all_nle_action_map:
            obs, reward, done, info = env.step(action)
        else:
            raise ValueError(f"Invalid action: {action}")