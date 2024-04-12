from nle_language_wrapper import NLELanguageWrapper
import difflib

class PromptBuilder(object):
    def __init__(self, *, max_history=None, max_length=None, prefix="", action_token="<|action|>", obs_token="<|observation|>"):
        self._prefix  = prefix
        self._max_history = max_history
        self._max_length = max_length
        self._obs_history = []
        self._action_history = []
        self._action_token = action_token
        self._obs_token = obs_token
        
    def append_action(self, action):
        self._action_history.append(action)
        
    def append_observation(self, obs):
        self._obs_history.append(obs)
        
    def get_prompt(self):
        return self._prefix + self._format()
    
    def reset(self):
        self._obs_history = []
        self._action_history = []

    def _format(self):
        n_steps = len(self._action_history)
        start_idx = min(self._max_history or (n_steps + 1), n_steps + 1)
        
        if self._max_length is None:
            return self._format_history(self._obs_history[-start_idx-1:], self._action_history[-start_idx:])
        
        for i in reversed(range(1, start_idx+1)):
            text = self._format_history(self._obs_history[-i-1:], self._action_history[-i:])
            num_tokens = len(text.encode('utf-8')) # Ideal world, we know exactly how many tokens this string is, but estimate using num bytes
            if num_tokens <= self._max_length:
                return text
            
        raise ValueError("Unable to generate context that fits within max_length.")
        
    def _format_history(self, obs_history, action_history):
        raise NotImplementedError
    
class ConcatPromptBuilder(PromptBuilder):
    def _format_history(self, obs_history, action_history):
        text = ""
        for obs, action in zip(obs_history[:-1], action_history):
            text += f"{self._obs_token}" + obs + f"{self._action_token}" + action
        text += f"{self._obs_token}" + obs_history[-1] + f"{self._action_token}"
        return text
    
class DiffPromptBuilder(PromptBuilder):
    def _format_history(self, obs_history, action_history):
        text = f"{self._obs_token}" + obs_history[0]
        for action, (prev_obs, obs) in zip(action_history, zip(obs_history[:-1], obs_history[1:])):
            prev_obs = prev_obs.strip().splitlines()
            obs = obs.strip().splitlines()
            obs = "\n".join(difflib.unified_diff(prev_obs, obs, n=0, lineterm=""))
            text += f"{self._action_token}" + action + f"{self._obs_token}" + obs
        text += f"{self._action_token}"
        return text
    
def get_default_prefix():
    # action_names = [action_strs[0] for action, action_strs in self.all_nle_action_map.items() if action in env.actions]
    prefix = "You are an agent playing NetHack. Predict the next keypresses.\n\n"
    return prefix
    # prefix += f"Output only one of the following actions:\n\n" + ", ".join(action_names) + "\n\n"

def nle_text_obs(text_obsv):
    key_name_pairs = [
        ("text_blstats", "statistics"),
        ("text_glyphs", "glyphs"),
        ("text_message", "message"),
        ("text_inventory", "inventory"),
        ("text_cursor", "cursor"),
    ]
    return "\n".join([f"{name}[\n{text_obsv[key]}\n]" for key, name in key_name_pairs])