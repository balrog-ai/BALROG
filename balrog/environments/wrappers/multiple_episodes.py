import random
from collections import defaultdict

import gymnasium as gym
import numpy as np


class MultiEpisodeWrapper(gym.Wrapper):
    def __init__(self, env, num_episodes, aggregate_function=max):
        """
        A Gym environment wrapper to run multiple episodes and aggregate statistics.
        Args:
            env (gym.Env): The environment to wrap.
            num_episodes (int): The number of episodes to run.
            aggregate_function (callable, optional): The function to aggregate statistics across episodes. Defaults to max. Could be mean, min, etc.
        """
        super().__init__(env)
        self.num_episodes = num_episodes
        self.aggregate_function = aggregate_function
        self.current_episode = 0
        self.total_stats = defaultdict(list)
        self.max_steps = self.num_episodes * self.env.max_steps

    def reset(self, seed=None, **kwargs):
        # we have to reseed the environments every time we do resets!
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.current_episode = 0
        return self.env.reset(seed=seed, **kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        if done:
            self.current_episode += 1
            if self.current_episode <= self.num_episodes:
                observation, _ = self.env.reset()
                done = False
                stats = self.env.get_stats()
                for key, value in stats.items():
                    self.total_stats[key].append(value)

        return observation, reward, terminated, truncated, info

    def get_stats(self):
        """
        returns stats for aggregate_function progression
        """
        best_stats = self.aggregate_function(
            (stats for stats in self.total_stats),
            key=lambda stats: stats["progression"],
            default=None,
        )

        return best_stats
