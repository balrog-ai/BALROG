from .env import CraftaxLanguageWrapper
from craftax.craftax.constants import Action

ACTIONS = [
    "noop",
    "up",
    "right",
    "down",
    "left",
    "do",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_diamond_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
    "make_diamond_sword",
    "place_table",
    "sleep",
    "place_stone",
    "place_furnace",
    "place_plant",
    "rest",
    "ascend",
    "descend",
    "make_iron_armour",
    "make_diamond_armour",
    "shoot_arrow",
    "make_arrow",
    "cast_fireball",
    "cast_iceball",
    "place_torch",
    "drink_potion_red",
    "drink_potion_green",
    "drink_potion_blue",
    "drink_potion_pink",
    "drink_potion_cyan",
    "drink_potion_yellow",
    "read_book",
    "enchant_sword",
    "enchant_armour",
    "make_torch",
    "level_up_dexterity",
    "level_up_strength",
    "level_up_intelligence",
    "enchant_bow",
]


TASKS = ["Craftax-Symbolic-v1"]


def get_instruction_prompt(task=None):
    action_strings = ",\n".join(f"{action}" for action in ACTIONS)
    instruction_prompt = f"""
You are an agent playing Craftax. The following are the possible actions you can take in the game:

{action_strings}.

In a moment I will present a history of actions and observations from the game.
You can only output one of the above actions at a time.

Your goal is to get as far as possible in the game.

PLAY!
""".strip()

    return instruction_prompt
