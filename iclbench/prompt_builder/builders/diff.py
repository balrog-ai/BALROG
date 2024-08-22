from .history import HistoryPromptBuilder
import difflib


def get_diff(a, b):
    return "\n".join(
        difflib.unified_diff(
            a.splitlines(),
            b.splitlines(),
            n=0,
            lineterm="",
        )
    )


def clean_diff(diff, remove=["---", "+++", "@@"]):
    return "\n".join(
        [
            line
            for line in diff.splitlines()
            if not any(line.strip() == r for r in remove)
        ]
    )


class DiffHistoryPromptBuilder(HistoryPromptBuilder):
    def format_history(self, obs_history, action_history):
        text = f"{self._obs_token}" + obs_history[0]
        for action, (prev_obs, obs) in zip(
            action_history, zip(obs_history[:-1], obs_history[1:])
        ):
            obs = clean_diff(get_diff(prev_obs, obs), remove=["---", "+++"])
            text += f"{self._action_token}" + action + f"{self._obs_token}" + obs
        text += f"{self._action_token}"
        return text
