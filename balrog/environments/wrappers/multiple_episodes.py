from collections import defaultdict

import gym


class MultiEpisodeWrapper(gym.Wrapper):
    def __init__(self, env, num_episodes):
        super().__init__(env)
        self.num_episodes = num_episodes
        self.current_episode = 0
        self.total_stats = defaultdict(list)
        self.max_steps = self.num_episodes * self.env.max_steps

    def reset(self):
        # TODO: we have to reseed the environments every time we do resets!
        self.current_episode = 0
        return self.env.reset()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        if done:
            self.current_episode += 1
            if self.current_episode < self.num_episodes:
                observation = self.env.reset()
                done = False
                stats = self.env.get_stats()
                for key, value in stats.items():
                    self.total_stats[key].append(value)

        return observation, reward, done, info

    def get_stats(self):
        """
        returns stats for max progression
        """
        best_stats = max(
            (stats for stats in self.total_stats),
            key=lambda stats: stats["progression"],
            default=None,
        )

        return best_stats
