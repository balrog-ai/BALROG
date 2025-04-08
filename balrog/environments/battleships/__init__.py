import numpy as np

from balrog.environments.battleships.base import BattleshipsWrapper


def get_instruction_prompt(env, instruction):
    action_strings = ""
    for row in np.array(env.language_action_space).reshape(env.board.shape):
        for el in row:
            action_strings += f"{el} "
        action_strings += "\n"

    instruction_prompt = f"""
You are an agent playing a battleships game. Your goal is to sink all ships. The possible actions are coordinates:
{action_strings}

In a moment I will present you an observation.

Tips:
- Ships can be adjacent to each other.
- Try to finish of ships when you hit them.

It's your turn. What coordinate would you like to attack?
""".strip()

    return instruction_prompt
