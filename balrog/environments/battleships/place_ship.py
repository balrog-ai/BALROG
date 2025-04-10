import gym
import numpy as np


class PlaceShip(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.env.unwrapped._place_ship = self.place_ship

    def place_ship(self, ship_size: int) -> None:
        can_place_ship = False
        while not can_place_ship:  # todo add protection infinite loop
            ship = self.env.unwrapped._get_ship(ship_size, self.board_size)
            can_place_ship = self._is_place_empty_and_non_adjacent(ship)
        self.board[ship.min_x : ship.max_x, ship.min_y : ship.max_y] = True

    def _is_place_empty_and_non_adjacent(self, ship) -> bool:
        # Check if the ship placement area is empty
        if not self._is_place_empty(ship):
            return False

        # Check adjacency by expanding the ship's bounding box
        min_x = max(ship.min_x - 1, 0)
        max_x = min(ship.max_x + 1, self.board_size[0])
        min_y = max(ship.min_y - 1, 0)
        max_y = min(ship.max_y + 1, self.board_size[1])

        # Ensure no adjacent ships
        return np.count_nonzero(self.board[min_x:max_x, min_y:max_y]) == 0

    def _is_place_empty(self, ship) -> bool:
        return np.count_nonzero(self.board[ship.min_x : ship.max_x, ship.min_y : ship.max_y]) == 0
