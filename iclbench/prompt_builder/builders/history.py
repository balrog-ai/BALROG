from collections import deque
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


class HistoryPromptBuilder:
    def __init__(
        self,
        *,
        max_history=128,
        max_length=128000,
        diff=False,
        use_history=True,
        prefix="You are about to be presented with an observation history. Respond with an appropriate action.\n\n",
    ):

        self._max_history = 1 if not use_history else max_history
        self._max_length = max_length
        self.diff = diff
        self.use_history = use_history
        self.prefix = prefix

        self._near_history = deque(maxlen=2)
        self._obs_history = deque(maxlen=self._max_history)
        self.previous_action = None
        self._last_obs = None

    def update_observation(self, obs, action=None):
        long_term_context, short_term_context = obs
        self._last_obs = short_term_context + "\n" + long_term_context
        self._near_history.append(long_term_context)

        if len(self._near_history) > 1 and self.diff:
            diff = clean_diff(get_diff(self._near_history[0], self._near_history[-1]))
            last_timestep = f"\n##########\nAction: {self.previous_action}\n\n" + diff
        else:
            last_timestep = (
                f"\n##########\nAction: {self.previous_action}\n\n" + long_term_context
            )

        self._obs_history.append(last_timestep)
        self.previous_action = action

    def update_action(self, action):
        self.previous_action = action

    def update_instruction_prompt(self, prompt):
        self.prefix = prompt

    def reset(self):
        self._near_history.clear()
        self._obs_history.clear()
        self.previous_action = None

    def get_prompt(self):
        history = f"\n##########\nCurrent observation\n{self._last_obs}"

        for i in range(len(self._obs_history) - 1, -1, -1):

            prev_obs = self._obs_history[i]
            history = prev_obs + history

        prompt = self.prefix + "\n\nObservation history\n" + history
        return [
            {
                "role": "user",
                "content": prompt,
            },
        ]
