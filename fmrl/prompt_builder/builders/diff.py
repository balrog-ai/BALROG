from .history import HistoryPromptBuilder
import difflib

class DiffHistoryPromptBuilder(HistoryPromptBuilder):
    def format_history(self, obs_history, action_history):
        text = f"{self._obs_token}" + obs_history[0]
        for action, (prev_obs, obs) in zip(action_history, zip(obs_history[:-1], obs_history[1:])):
            prev_obs = prev_obs.strip().splitlines()
            obs = obs.strip().splitlines()
            obs = "\n".join(difflib.unified_diff(prev_obs, obs, n=0, lineterm=""))
            text += f"{self._action_token}" + action + f"{self._obs_token}" + obs
        text += f"{self._action_token}"
        return text