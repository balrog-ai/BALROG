import json
import os
from dataclasses import dataclass, field
from typing import Optional

with open(os.path.join(os.path.dirname(__file__), "achievements.json"), "r") as f:
    ACHIEVEMENTS = json.load(f)


def get_progress_system(env):
    if "NetHackChallenge" in env.spec.id:
        return Progress()
    elif "MiniHack" in env.spec.id:
        return BaseProgress()
    else:
        raise ValueError(f"Unsupported environment type: {type(env)}")


@dataclass
class Progress:
    episode_return: float = 0.0
    score: int = 0
    depth: int = 1
    gold: int = 0
    experience_level: int = 1
    time: int = 0
    dlvl_list: list = field(default_factory=list)
    xplvl_list: list = field(default_factory=list)
    highest_achievement: Optional[str] = None
    progression: float = 0.0
    end_reason: Optional[str] = None

    def update(self, nle_obsv, reward, done, info):
        """
        Update the progress of the player given a message and stats.

        Returns:
            float: The progression of the player.
        """
        self.episode_return += reward

        stats = self._update_stats(nle_obsv["blstats"])

        if done:
            tty_chars = bytes(nle_obsv["tty_chars"].reshape(-1)).decode(errors="ignore")
            self.end_reason = self._get_end_reason(tty_chars, info["end_status"])

        xp = self._get_xp(stats)
        if xp not in self.xplvl_list and xp in ACHIEVEMENTS.keys():
            self.xplvl_list.append(xp)
            if ACHIEVEMENTS[xp] > self.progression:
                self.progression = ACHIEVEMENTS[xp]
                self.highest_achievement = xp

        dlvl = self._get_dlvl(stats)
        if dlvl not in self.dlvl_list and dlvl in ACHIEVEMENTS.keys():
            self.dlvl_list.append(dlvl)
            if ACHIEVEMENTS[dlvl] > self.progression:
                self.progression = ACHIEVEMENTS[dlvl]
                self.highest_achievement = dlvl

    def _update_stats(self, blstats):
        # see: https://arxiv.org/pdf/2006.13760#page=16
        stats_names = [
            "x_pos",
            "y_pos",
            "strength_percentage",
            "strength",
            "dexterity",
            "constitution",
            "intelligence",
            "wisdom",
            "charisma",
            "score",
            "hitpoints",
            "max_hitpoints",
            "depth",
            "gold",
            "energy",
            "max_energy",
            "armor_class",
            "monster_level",
            "experience_level",
            "experience_points",
            "time",
            "hunger_state",
            "carrying_capacity",
            "dungeon_number",
            "level_number",
        ]
        stats = {name: value for name, value in zip(stats_names, blstats)}

        self.score = int(stats["score"])
        self.depth = int(stats["depth"])
        self.gold = int(stats["gold"])
        self.experience_level = int(stats["experience_level"])
        self.time = int(stats["time"])

        return stats

    def _get_end_reason(self, tty_chars, end_status):
        end_reason_words = tty_chars.replace("You made the top ten list!", "").split()

        if len(end_reason_words) > 7 and end_reason_words[7].startswith("Agent"):
            end_reason = " ".join(end_reason_words[8:-2])
        else:
            end_reason = " ".join(end_reason_words[7:-2])
        sentences = end_reason.split(".")
        first_sentence = sentences[0].split()

        if "in" in first_sentence:
            index_in = first_sentence.index("in")
            first_part = " ".join(first_sentence[:index_in])
        else:
            first_part = " ".join(first_sentence)

        remaining_sentences = ".".join(sentences[1:]).strip()
        end_reason_final = f"{end_status.name}: " f"{first_part}." f" {remaining_sentences}".strip()

        return end_reason_final

    def _get_dlvl(self, stats):
        """
        Get the dungeong lvl from the stats string.

        Args:
            string (str): The stats string.
        Returns:
            str: The dungeong lvl
        """
        # dlvl = string.split("$")[0]
        dlvl = f"Dlvl:{stats['depth']}"
        return dlvl

    def _get_xp(self, stats):
        """
        Get the experience points from the stats string.

        Args:
            string (str): The stats string.
        Returns:
            str: The experience points
        """
        xp = f"Xp:{stats['experience_level']}"
        return xp


class BaseProgress:
    episode_return: float = 0.0
    progression: float = 0.0
    end_reason: Optional[str] = None

    def update(self, nle_obsv, reward, done, info):
        """
        Update the progress of the player given a message and stats.

        Args:
            message (str): The message to check for achievements.
            stats (str): The stats to check for achievements.

        Returns:
            float: The progression of the player.
        """
        self.episode_return += reward
        if reward >= 1.0:
            self.progression = 1.0
        else:
            self.progression = 0.0
        self.end_reason = info["end_status"]
