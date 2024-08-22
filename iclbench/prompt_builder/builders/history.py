from . import PromptBuilder
from abc import ABC
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


class HistoryPromptBuilder(PromptBuilder, ABC):
    def __init__(
        self,
        *,
        max_history=128,
        max_length=128000,
        language_obs=True,
        ascii_map=False,
        diff=True,
        summary=None,
    ):
        self._max_history = max_history
        self._max_length = max_length

        self._obs_history = deque(maxlen=2)
        self._cursor_history = deque(maxlen=2)
        self._inventory_history = deque(maxlen=2)

        self._action_history = deque(maxlen=self._max_history)
        self._diff_history = deque(maxlen=self._max_history)

        self.language_obs = language_obs
        self.ascii_map = ascii_map
        self.diff = diff
        self.summary = summary

    # Use a deque for the history in order to be efficient
    def update_history(self, obs):
        inventory, render, language_obs, cursor, action = obs

        ascii_map = f"\nMap\n{render}" if self.ascii_map else ""
        language_obs = f"\nObservation\n{language_obs}" if self.language_obs else ""

        obs = (
            f"\nAction: {self.previous_action}\n"
            + f"Inventory\n{inventory}\n"
            + ascii_map
            + cursor
            + language_obs
        )

        if len(self._obs_history) > 1 and self.diff:
            map_diff = clean_diff(
                get_diff(self._obs_history[0][0], self._obs_history[-1][0]),
                remove=["---", "+++"],
            )
            cursor_diff = clean_diff(
                get_diff(self._cursor_history[0][0], self._cursor_history[-1][0])
            )
            if cursor_diff:
                cursor_diff = "\n".join(cursor_diff.splitlines()[1:])
            inventory_diff = clean_diff(
                get_diff(self._inventory_history[0][0], self._inventory_history[-1][0])
            )
            inventory_diff = f"Inventory\n{inventory_diff}" if inventory_diff else ""

            diff = (
                f"\nAction: {self.previous_action}\n"
                + inventory_diff
                + map_diff
                + "\n"
                + cursor_diff
                + "\n"
            )
            self._diff_history.append(diff)  # THIS IS WAAAY OFF
        self.previous_action = action

    def reset(self):
        self._obs_history = deque(maxlen=self._max_history)
        self._action_history = deque(maxlen=self._max_history)

    def get_prompt(self):
        current_obs, current_obs_tokens = self._obs_history[-1]
        inventory, inventory_tokens = self._inventory_history[-1]

        inventory = f"Inventory\n{inventory}" if inventory else ""

        history = (
            f"\nCurrent observation\n{inventory}\n" + "Map\n" + current_obs
        )  # We are not adding this but okay
        cur_tokens = current_obs_tokens + inventory_tokens
        count = 0
        for i in range(len(self._diff_history) - 1, -1, -1):

            diff, diff_tokens = self._diff_history[i]
            count += 1
            if cur_tokens + diff_tokens >= self._max_length:
                print(i, cur_tokens + diff_tokens, self._max_length)
                break

            history = diff + history
        prompt = (
            self._obs_start
            + f"You are a {self.summary}\nHistory\n"
            + history
            + self._cursor_history[-1][0]
            + self._obs_end
            + "\n"
            + self._action_delim
        )
        return prompt
