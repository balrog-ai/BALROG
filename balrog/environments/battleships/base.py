from collections import defaultdict
from itertools import product

import gym
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
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
        text_observation = self.get_text_observation(obs)
        image = self.get_image_observation(obs)

        feedback = self.get_feedback(reward, old_reward)

        prompt = f"{text_observation}\n{feedback}" if feedback else f"Objects on the map:\n{text_observation}"

        obs = defaultdict(lambda: None)

        obs["text"] = {"long_term_context": prompt, "short_term_context": ""}
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
                ship_sunk = np.all(np.bitwise_and(ship, hits.astype(int)) == ship)
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

    def get_text_observation(self, obs):
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

        obsv = str(dataframe)

        return obsv

    def get_image_observation(self, obs, symbol=" ", cell_size=80, font_size=40, border_width=2):
        board = np.empty(self.env.board_size, dtype=str)
        board[obs[0] != 0] = "X"
        board[obs[1] != 0] = "O"

        num_rows, num_columns = board.shape
        columns = [chr(i) for i in range(ord("A"), ord("A") + num_columns)]
        index = [str(i + 1) for i in range(num_rows)]

        # Calculate image dimensions with space for row/column labels
        header_size = cell_size // 2
        width = (num_columns * cell_size) + header_size
        height = (num_rows * cell_size) + header_size

        # Create image with white background
        image = Image.new("RGB", (width, height), color="white")
        draw = ImageDraw.Draw(image)

        try:
            # Try to load a font that supports Unicode symbols
            font = ImageFont.truetype("Arial Unicode MS", font_size)
        except IOError:
            try:
                # Try another common font
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except IOError:
                # Fallback to default font
                font = ImageFont.load_default()

        # Draw column headers (A, B, C, ...)
        for col_idx, col in enumerate(columns):
            x = header_size + (col_idx * cell_size) + (cell_size // 2)
            y = header_size // 2
            draw.text((x, y), col, fill="black", font=font, anchor="mm")

        # Draw row headers (1, 2, 3, ...)
        for row_idx, row in enumerate(index):
            x = header_size // 2
            y = header_size + (row_idx * cell_size) + (cell_size // 2)
            draw.text((x, y), row, fill="black", font=font, anchor="mm")

        # Draw grid
        for row_idx in range(num_rows + 1):
            y = header_size + (row_idx * cell_size)
            draw.line([(header_size, y), (width, y)], fill="black", width=border_width)

        for col_idx in range(num_columns + 1):
            x = header_size + (col_idx * cell_size)
            draw.line([(x, header_size), (x, height)], fill="black", width=border_width)

        # Draw cell contents
        for row_idx in range(num_rows):
            for col_idx in range(num_columns):
                cell_content = board[row_idx, col_idx]
                x0 = header_size + (col_idx * cell_size) + 5
                y0 = header_size + (row_idx * cell_size) + 5
                x1 = header_size + ((col_idx + 1) * cell_size) - 5
                y1 = header_size + ((row_idx + 1) * cell_size) - 5

                # Center of the cell for drawing shapes
                center_x = header_size + (col_idx * cell_size) + (cell_size // 2)
                center_y = header_size + (row_idx * cell_size) + (cell_size // 2)
                radius = (cell_size // 2) - 10
                radius = int(radius * 0.7)
                # Draw based on cell content
                if cell_content == "X":  # Hit
                    # Draw a red X
                    draw.line(
                        [(center_x - radius, center_y - radius), (center_x + radius, center_y + radius)],
                        fill="red",
                        width=10,
                    )
                    draw.line(
                        [(center_x + radius, center_y - radius), (center_x - radius, center_y + radius)],
                        fill="red",
                        width=10,
                    )
                elif cell_content == "O":  # Miss
                    # Draw a blue circle
                    radius = 5
                    draw.ellipse(
                        [(center_x - radius, center_y - radius), (center_x + radius, center_y + radius)],
                        outline="black",
                        width=5,
                    )
                else:  # Empty
                    # Draw a light white square
                    draw.rectangle([x0, y0, x1, y1], fill="white")

        return image
