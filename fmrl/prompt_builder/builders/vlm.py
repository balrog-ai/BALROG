from .. import PromptBuilder
from abc import ABC
from collections import deque
import re


class VLMHistoryPromptBuilder(PromptBuilder, ABC):
    def __init__(
        self,
        *,
        max_history=100,
        image_delim="<|image_1|>",
        summary="lawful dwarven Valkyrie",
    ):
        self._max_history = max_history

        self._obs_history = deque(maxlen=2)
        self._cursor_history = deque(maxlen=2)
        self._inventory_history = deque(maxlen=2)

        self.history = deque(maxlen=self._max_history)
        self._image_delim = image_delim
        self.image = None
        self.full_stats = ""
        self.current_obs = ""
        self.summary = summary

    # Use a deque for the history in order to be efficient
    def update_history(
        self, inventory, message, menu, stats, cursor, action, image_path
    ):

        if stats != "":
            self.full_stats = stats
            stats = stats.split("\n")[1]
            regex = r"Dlvl:(\d+).*?HP:(\d+\(\d+\)).*?T:(\d+)"

            matches = re.search(regex, stats)
            if matches:
                dlvl = matches.group(1)
                hp = matches.group(2)
                t = matches.group(3)
                stats = f"Dlvl: {dlvl}, HP: {hp}, T: {t}\n"

        history = (
            "msg: "
            + message
            + "\n"
            + stats
            + "cursor: "
            + cursor
            + "\n"
            + "action taken: "
            + action
            + "\n\n"
        )
        self.history.append(history)

        inventory = f"Inventory\n{inventory}" if inventory else ""

        if menu != "":
            message = "Menu interaction\n" + menu

        if menu == "":
            self.image = image_path

        self.current_obs = (
            f"\nCurrent observation:\n{inventory}\n\n"
            + "Current message: "
            + message
            + "\n"
            + self.full_stats
            + "cursor: "
            + cursor
            + "\n"
        )

    def reset(self):
        self.history = deque(maxlen=self._max_history)

    def get_prompt(self):
        history = ""
        for i in range(len(self.history) - 2, -1, -1):
            obs = self.history[i]
            history = obs + history

        history = history + self.current_obs
        history = (history 
            + "\n" 
            + self._image_delim 
            + "\n" 
            + "NEXT ACTION:"
        )
        
        prompt = (
            f"You are playing NetHack as a {self.summary}\nPrevious observations:\n"
            + history
            # + self._obs_end
            + "\n"
        )
        return prompt, self.image
