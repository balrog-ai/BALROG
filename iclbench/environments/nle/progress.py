from typing import List, Optional, Dict, Union
from pathlib import Path
import json
import re
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

with open(os.path.join(os.path.dirname(__file__), "achievements.json"), "r") as f:
    ACHIEVEMENTS = json.load(f)

with open(os.path.join(os.path.dirname(__file__), "spam.txt"), "r") as f:
    spam = f.readlines()
    SPAM = [line.strip() for line in spam]


@dataclass
class Progress:
    # score: list                        = field(default_factory=list)
    # depth: list                        = field(default_factory=list)
    # gold: list                         = field(default_factory=list)
    # experience_level: list             = field(default_factory=list)
    # time: list                         = field(default_factory=list)
    episode_return: float              = 0.
    score: int                         = 0
    depth: int                         = 1
    gold: int                          = 0
    experience_level: int              = 1
    time: int                          = 0
    achievement_list: list             = field(default_factory=list)
    dlvl_list: list                    = field(default_factory=list)
    highest_achievement: Optional[str] = None
    progression: float                 = 0.
    end_reason: Optional[str]          = None

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
        
        # message = ''.join(np.vectorize(chr)(message)).strip()
        message = bytes(nle_obsv["message"]).decode()
        stats = self._update_stats(nle_obsv["blstats"])
        
        if done:
            tty_chars = bytes(nle_obsv["tty_chars"].reshape(-1)).decode()
            self.end_reason = self._get_end_reason(tty_chars, info["end_status"])

        for spam_m in SPAM:
            if spam_m in message:
                return self.progression

        for achievement in ACHIEVEMENTS.keys():
            if achievement in message and achievement not in self.achievement_list:
                self.achievement_list.append(achievement)
                if ACHIEVEMENTS[achievement] > self.progression:
                    self.progression = ACHIEVEMENTS[achievement]
                    self.highest_achievement = achievement

            dlvl = self._get_dlvl(stats)
            if dlvl not in self.dlvl_list and dlvl in ACHIEVEMENTS.keys():
                self.dlvl_list.append(dlvl)
                if ACHIEVEMENTS[dlvl] > self.progression:
                    self.progression = ACHIEVEMENTS[dlvl]
                    self.highest_achievement = dlvl

        return self.progression
    
    def _update_stats(self, blstats):
        # see: https://arxiv.org/pdf/2006.13760#page=16
        stats_names = [
            "x_pos", "y_pos",
            "strength_percentage", "strength", "dexterity", "constitution", "intelligence", "wisdom",
            "charisma", "score", "hitpoints", "max_hitpoints", "depth", "gold", "energy", "max_energy",
            "armor_class", "monster_level", "experience_level", "experience_points", "time", "hunger_state",
            "carrying_capacity", "dungeon_number", "level_number"
        ]
        stats = {name: value for name, value in zip(stats_names, blstats)}
        
        # self.score.append(stats["score"])
        # self.depth.append(stats["depth"])
        # self.gold.append(stats["gold"])
        # self.experience_level.append(stats["experience_level"])
        # self.time.append(stats["time"])
        self.score = stats["score"]
        self.depth = stats["depth"]
        self.gold = stats["gold"]
        self.experience_level = stats["experience_level"]
        self.time = stats["time"]
        
        return stats
    
    def _get_end_reason(self, tty_chars, end_status):
        end_reason = tty_chars.replace('You made the top ten list!', '').split()
        if end_reason[7].startswith('Agent'):
            end_reason = ' '.join(end_reason[8:-2])
        else:
            end_reason = ' '.join(end_reason[7:-2])
        first_sentence = end_reason.split('.')[0].split()
        return end_status.name + ': ' + \
               (' '.join(first_sentence[:first_sentence.index('in')]) + '. ' +
               '.'.join(end_reason.split('.')[1:]).strip()).strip()

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