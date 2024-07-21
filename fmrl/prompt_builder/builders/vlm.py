from .. import PromptBuilder
from abc import ABC
from collections import deque
import re


class VLMHistoryPromptBuilder(PromptBuilder, ABC):
    def __init__(
        self,
        *,
        max_history=100,
        image_history=1,
        image_delims=("<|image_1|>", "<|image_2|>"),
        summary="lawful dwarven Valkyrie",
    ):
        self._max_history = max_history

        self.history_imgs = image_history
        self._image_history = deque(maxlen=image_history)
        self.history = deque(maxlen=self._max_history)
        self._image_delims = image_delims
        self.image = None
        self.full_stats = ""
        self.current_obs = ""
        self.summary = summary

    # Use a deque for the history in order to be efficient
    def update_history(
        self, inventory, message, menu, stats, cursor, action, image
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
        else:
            self.image = image
            self._image_history.append(image)
            
        if len(self._image_history) == 1 and self.history_imgs == 2:
            self._image_history.append(image)

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

    def get_prompt(self):
        history = ""
        for i in range(len(self.history) - 2, -1, -1):
            obs = self.history[i]
            history = obs + history

        image_idx = 0
        if self.history_imgs == 2:
            history = (history         
                + "\nPrevious observation image:\n"
                + self._image_delims[image_idx]
                + "\n"
            )
            image_idx += 1
            

        history = history + self.current_obs
        history = (history 
            + "\n" 
            + self._image_delims[image_idx]
            + "\n" 
            + "NEXT ACTION:"
        )
        
        prompt = (
            f"You are playing NetHack as a {self.summary}\nOservation history:\n"
            + history
            + "\n"
        )
        print(len(self._image_history))
        return prompt, list(self._image_history)
