from .history import HistoryPromptBuilder

class ConcatHistoryPromptBuilder(HistoryPromptBuilder):
    def format_history(self, obs_history, action_history):
        text = ""
        for obs, action in zip(obs_history[:-1], action_history):
            text += f"{self._obs_delim}" + obs + f"{self._action_delim}" + action
        text += f"{self._obs_delim}" + obs_history[-1] + f"{self._action_delim}"
        return text