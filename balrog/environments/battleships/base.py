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
        dataframe = self.get_dataframe(obs)

        text_observation = self.get_text_observation(dataframe)
        feedback = self.get_feedback(reward, old_reward)

        prompt = f"{text_observation}\n{feedback}" if feedback else f"Objects on the map:\n{text_observation}"

        obs = defaultdict(lambda: None)

        obs["text"] = {"long_term_context": prompt, "short_term_context": ""}
        image = self.get_image_observation(dataframe)
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

    def get_dataframe(self, obs):
        board = np.empty(self.env.board_size, dtype=str)

        # Create a mask for sunk ships
        sunk_mask = np.zeros_like(self.ships, dtype=bool)
        for i in self.sunk_ships:
            sunk_mask = np.logical_or(sunk_mask, self.ships == i)

        board[obs[0] != 0] = "X"
        board[obs[1] != 0] = "O"
        board[sunk_mask] = "Z"  # Sunk ships

        num_rows, num_columns = board.shape
        columns = [chr(i) for i in range(ord("A"), ord("A") + num_columns)]
        index = [i + 1 for i in range(num_rows)]

        dataframe = pd.DataFrame(board, columns=columns, index=index)
        dataframe = dataframe.replace([""], " ")

        return dataframe

    def get_text_observation(self, dataframe):
        obsv = str(dataframe)

        return obsv

    def get_image_observation(self, dataframe):
        # import matplotlib.pyplot as plt
        # from matplotlib.colors import LinearSegmentedColormap
        # from matplotlib.figure import Figure
        # from matplotlib.backends.backend_agg import FigureCanvasAgg
        # import io
        # from PIL import Image

        # # Define colors for each cell type
        # color_map = {
        #     "⬜": [0.9, 0.9, 1.0],    # Light blue for empty water
        #     "❌": [1.0, 0.0, 0.0],    # Red for hits
        #     "⚫": [0.3, 0.3, 0.3],    # Dark gray for misses
        #     "💥": [1.0, 0.6, 0.0]     # Orange for sunk ships
        # }

        # # Create a numerical representation for colormapping
        # numeric_board = np.zeros(dataframe.shape + (3,), dtype=float)

        # for i in range(dataframe.shape[0]):
        #     for j in range(dataframe.shape[1]):
        #         cell_value = dataframe.iloc[i, j]
        #         numeric_board[i, j] = color_map.get(cell_value, [1, 1, 1])

        # # Create a figure with the right dimensions and no padding
        # fig_width = dataframe.shape[1] + 1  # +1 for row labels
        # fig_height = dataframe.shape[0] + 1  # +1 for column labels
        # fig = Figure(figsize=(fig_width, fig_height), dpi=72)
        # canvas = FigureCanvasAgg(fig)
        # ax = fig.add_subplot(111)

        # # Plot the board
        # ax.imshow(numeric_board, aspect='equal')

        # # Add grid lines
        # ax.set_xticks(np.arange(-0.5, dataframe.shape[1], 1), minor=True)
        # ax.set_yticks(np.arange(-0.5, dataframe.shape[0], 1), minor=True)
        # ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

        # # Add column labels (A, B, C, ...)
        # ax.set_xticks(np.arange(dataframe.shape[1]))
        # ax.set_xticklabels(dataframe.columns)

        # # Add row labels (1, 2, 3, ...)
        # ax.set_yticks(np.arange(dataframe.shape[0]))
        # ax.set_yticklabels(dataframe.index)

        # # Remove axis padding
        # ax.set_xlim(-0.5, dataframe.shape[1] - 0.5)
        # ax.set_ylim(-0.5, dataframe.shape[0] - 0.5)

        # # Render the figure to a numpy array
        # canvas.draw()
        # buf = io.BytesIO()
        # fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        # buf.seek(0)

        # return Image.open(buf)
        return None
