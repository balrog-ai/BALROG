from .history import HistoryPromptBuilder

class ConcatHistoryPromptBuilder(HistoryPromptBuilder):
    def format_history(self, obs_history, action_history):
        text = ""
        for obs, action in zip(obs_history[:-1], action_history):
            text += f"{self._obs_token}" + obs + f"{self._action_token}" + action
        text += f"{self._obs_token}" + obs_history[-1] + f"{self._action_token}"
        return text