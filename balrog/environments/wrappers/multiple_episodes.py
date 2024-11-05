import random

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
        self.total_stats = []
        self.max_steps = (self.num_episodes + 1) * self.env.max_steps

    def reset(self, **kwargs):
        self.current_episode = 0
        self.current_kwargs = kwargs
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        if done:
            if self.current_episode <= self.num_episodes:
                # we have to reseed the environments every time we do resets!
                seed = self.current_kwargs.get("seed", None)
                if seed is not None:
                    random.seed(seed)
                    np.random.seed(seed)
                new_obs, new_info = self.env.reset(**self.current_kwargs)
                terminated = truncated = False

                stats = self.env.get_stats()
                self.total_stats.append(stats)

                new_info["final_observation"] = obs
                new_info["final_info"] = info

                obs = new_obs
                info = new_info

            self.current_episode += 1

        # TODO: for now episode return will be inflated, since evaluator does `episode_return += reward`
        return obs, reward, terminated, truncated, info

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
