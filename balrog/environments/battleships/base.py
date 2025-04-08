from collections import defaultdict
from itertools import product

import gym
import numpy as np
import pandas as pd
from scipy.ndimage import label


class BattleshipsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    @property
    def max_steps(self):
        return self.env.episode_steps

    @property
    def default_action(self):
        return self.get_text_action(self.env.action_space.sample())

    def get_text_action(self, action):
        return self.language_action_space[action]

    def get_text_observation(self, obs):
        board = np.empty(self.env.board_size, dtype=str)
        board[obs[0] != 0] = "❌"
        board[obs[1] != 0] = "⚫"

        num_rows, num_columns = board.shape
        columns = [chr(i) for i in range(ord("A"), ord("A") + num_columns)]
        index = [i + 1 for i in range(num_rows)]

        dataframe = pd.DataFrame(board, columns=columns, index=index)
        dataframe = dataframe.replace([""], "⬜")
        obsv = str(dataframe)

        return obsv

    def get_feedback(self, reward, old_reward):
        if reward is None:
            return ""

        if reward == 10:
            return "HIT AND SINK! You've sunk an enemy ship!"
        elif reward == 1:
            return "HIT! You've struck an enemy vessel!"
        elif old_reward == -1:
            return "WASTED SHOT! You've already fired at this empty location."
        elif reward == -0.5:
            return "REDUNDANT HIT! This part of the ship was already destroyed."
        else:
            return "MISS! Your missile splashed into empty water."

    def battleships_process_obsv(self, obs, reward, old_reward):
        text_observation = self.get_text_observation(obs)
        feedback = self.get_feedback(reward, old_reward)

        prompt = (
            f"Objects on the map:\n{text_observation}\n{feedback}"
            if feedback
            else f"Objects on the map:\n{text_observation}"
        )

        obs = defaultdict(lambda: None)

        obs["text"] = {"long_term_context": prompt, "short_term_context": ""}
        image = None  # TODO add rendering
        obs["image"] = image

        return obs

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        num_rows, num_columns = self.env.board.shape
        columns = [chr(i) for i in range(ord("A"), ord("A") + num_columns)]
        index = [i + 1 for i in range(num_rows)]

        self.language_action_space = list(map(lambda x: "".join((x[0], str(x[1]))), product(columns, index)))
        self.progression = 0.0
        self.total_reward = 0.0

        self.sunk_ships = set()
        self.ships, self.num_ships = label(self.env.board_generated)
        self.max_reward = self.num_ships * 10 + np.count_nonzero(self.env.board_generated)

        return self.battleships_process_obsv(obs, None, None)

    def _reward_shaping(self, obs, reward):
        # If the reward is for a hit (touched)
        if reward == self.env.reward_dictionary["touched"]:
            hits = obs[0]

            for i in range(1, self.num_ships + 1):
                ship = (self.ships == i).astype(int)
                ship_sunk = np.all(np.bitwise_and(ship, hits[0].astype(int)) == ship)
                if ship_sunk and i not in self.sunk_ships:
                    self.sunk_ships.add(i)
                    return 10  # +10 reward for sinking the ship

            return 1  # +1 reward for hitting a ship for the first time

        return 0

    def step(self, action):
        action_int = self.language_action_space.index(action)
        obs, old_reward, done, info = self.env.step(action_int)

        reward = self._reward_shaping(obs, old_reward)

        self.total_reward += reward

        if done:
            self.progression = min(1.0, self.total_reward / self.max_reward)

        return self.battleships_process_obsv(obs, reward, old_reward), reward, done, info

    def get_stats(self):
        return {"progression": self.progression}
