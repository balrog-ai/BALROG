import enum

from .auto_more import AutoMore
from .base import NLELanguageWrapper


class Role(enum.Enum):
    ARCHEOLOGIST = "arc"
    BARBARIAN = "bar"
    CAVEMAN = "cav"
    HEALER = "hea"
    KNIGHT = "kni"
    MONK = "mon"
    PRIEST = "pri"
    RANGER = "ran"
    ROGUE = "rog"
    SAMURAI = "sam"
    TOURIST = "tou"
    VALKYRIE = "val"
    WIZARD = "wiz"


def get_instruction_prompt(env, task=None):
    action_strings = ",\n".join(f"{action}: {description}" for action, description in env.action_str_desc_map.items())
    instruction_prompt = f"""
You are an agent playing NetHack. The following are the possible actions you can take in the game, followed by a short description of each action:

{action_strings}.

Tips:
- When the message asks for a completion, such as: "What do you want to eat? [d or ?*]", you should respond with a single character corresponding to the item you want to eat/use.
    - For example, "What do you want to eat? [dgh or ?*]" -> Possible answers are "d", "g", or "h" to eat the associated food.
- When the message asks for a direction, such as: "In what direction?" you should respond with a direction.
- When the message has --More-- at the end, your next action should be "more" to see the rest of the message.
- Explore the environment to find the stairs down to the next level.
- Always carefully read the last message to understand the current state of the game and decide your next action accordingly.
- If you keep moving in the same direction, you will eventually hit a wall and stop moving. Your message might be: "It's solid stone", or "It's a wall". Change your action to move in another direction to continue exploring the environment.
- Read the language observation carefully and look at ascii map or image observation provided to decide the next action to take and where to move next.
- You can attack monsters by moving into them.

In a moment I will present a history of actions and observations from the game.
Your goal is to get as far as possible in the game.

PLAY!
""".strip()

    return instruction_prompt
