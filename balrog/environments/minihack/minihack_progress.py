from typing import Optional

import gymnasium as gym


class MiniHackProgress:
    episode_return: float = 0.0
    progression: float = 0.0
    end_reason: Optional[str] = None

    def update(self, reward, info):
        self.episode_return += reward
        if reward >= 1.0:
            self.progression = 1.0
        else:
            self.progression = 0.0
        self.end_reason = info["end_status"]


class MiniHackProgressWrapper(gym.Wrapper):
    def __init__(self, env, progression_on_done_only: bool = True):
        super().__init__(env)
        self.progression_on_done_only = progression_on_done_only

    def reset(self, **kwargs):
        self.progress = MiniHackProgress()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, term, trun, info = self.env.step(action)
        self.progress.update(reward, info)

        done = term or trun
        if not self.progression_on_done_only or done:
            info["episode_extra_stats"] = self.episode_extra_stats(info)

        return obs, reward, term, trun, info

    def episode_extra_stats(self, info):
        extra_stats = info.get("episode_extra_stats", {})
        new_extra_stats = {
            "progression": self.progress.progression,
        }

        return {**extra_stats, **new_extra_stats}
