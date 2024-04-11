from typing import List, Optional, Dict, Union
from pathlib import Path
import json
import re

class Progress:
    """
    Class to keep track of the user's progress in the game.
    """

    def __init__(
        self,
        achievements: Path = Path("achievements.json"),
        spam_file: Path = Path("spam.txt"),
    ):
        """
        Args:
            achievements (Path): A path to a json file containing the achievements dictionary.
        """

        with open(achievements, "r") as f:
            self.achievements_dict = json.load(f)

        with open(spam_file, "r") as f:
            spam = f.readlines()
            self.spam = [line.strip() for line in spam]

    def reset(self):
        self.achievement_list = ["Welcome to experience level 1"]
        self.dlvl_list = [self.achievements_dict[self.achievement_list[0]]]
        self.highest_achievement = None
        self.progression = 0

    def update(self, message, stats):
        """
        Update the progress of the player given a message and stats.

        Args:
            message (str): The message to check for achievements.
            stats (str): The stats to check for achievements.

        Returns:
            float: The progression of the player.
        """
        for achievement in self.achievements_dict.keys():
            if (
                achievement in message
                and achievement not in self.achievement_list
                and message not in self.spam
            ):
                self.achievement_list.append(achievement)
                if self.achievements_dict[achievement] > self.progression:
                    self.progression = self.achievements_dict[achievement]
                    self.highest_achievement = achievement

            dlvl = self._get_dlvl(stats)
            if dlvl not in self.dlvl_list:
                self.dlvl_list.append(dlvl)
                if self.achievements_dict[dlvl] > self.progression:
                    self.progression = self.achievements_dict[dlvl]
                    self.highest_achievement = dlvl

        return self.progression

    def get_highest_achievement(self):
        """
        Get the highest achievement of the player.

        Returns:
            Optional[str]: The highest achievement of the player.
        """
        return self.highest_achievement

    def get_progress(self):
        """
        Get the progression of the player.

        Returns:
            float: The progression of the player.
        """
        return self.progression

    def get_achievements(self):
        """
        Get the achievements of the player.

        Returns:
            List[str]: The achievements of the player.
        """
        return self.achievement_list

    def _get_dlvl(self, string):
        """
        Get the dungeong lvl from the stats string.

        Args:
            string (str): The stats string.
        Returns:
            str: The dungeong lvl
        """
        return "Dlvl:" + re.search(r'Depth:\s*(\d+)', string).group(1)
        # return string.split("$")[0]