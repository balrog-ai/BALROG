import numpy as np

from balrog.environments.battleships.base import BattleshipsWrapper


def get_instruction_prompt(env, instruction):
    ship_names = {
        5: "Carrier",
        4: "Battleship",
        3: "Cruiser",
        2: "Destroyer",
    }

    ships_strings = "\n".join(
        [
            f"{number} {ship_names[ship_size]} {ship_size} cells {'each' if number > 1 else ''}"
            for ship_size, number in env.ship_sizes.items()
        ]
    )

    num_rows, num_columns = env.board.shape

    instruction_prompt = f"""
You are an AI agent playing a Battleships game on a {num_rows}x{num_columns} grid. Your mission is to strategically locate and sink all enemy ships hidden on the board.

Game Rules:
- The board is a {num_rows}x{num_columns} grid with coordinates from {env.language_action_space[0]} to {env.language_action_space[-1]}
- Ships are placed horizontally or vertically, never diagonally
- Ships cannot be adjacent to each other (not even diagonally)
- A hit will be reported when you successfully strike a ship
- A miss will be reported when you strike empty water

The enemy has the following ships:
{ships_strings}

In a moment I will present you an observation grid. This grid represents the current state of a Battleship game. The format uses the following notation:
- O: Water (missed shot)
- X: Hit (part of a ship that has been hit)
- Z: Sunk (indicates that the entire ship has been sunk)

Tips:
- When you get a hit, try to sunk the ship as you get more reward for that.
- Avoid targeting cells adjacent to sunken ships.

IMPORTANT: Your response must be EXACTLY one coordinate in the format of a letter followed by a number (e.g., "E5", "A1", "J10"). Do not provide any explanation or reasoning in your response.

PLAY
""".strip()

    return instruction_prompt
