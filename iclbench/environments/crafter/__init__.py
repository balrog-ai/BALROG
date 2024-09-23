from .env import CrafterLanguageWrapper, ACTIONS


ACTION_DICT = {
    "Noop": "do nothing",
    "Move West": "move west on flat ground",
    "Move East": "move east on flat ground",
    "Move North": "move north on flat ground",
    "Move South": "move south on flat ground",
    "Do": "interact with a creature or material with the necessary tool",
    "Sleep": "sleep when energy level is below maximum",
    "Place Stone": "place a stone in front",
    "Place Table": "place a table",
    "Place Furnace": "place a furnace",
    "Place Plant": "place a plant",
    "Make Wood Pickaxe": "make a wood pickaxe with a nearby table and wood in inventory",
    "Make Stone Pickaxe": "make a stone pickaxe with a nearby table, wood, and stone in inventory",
    "Make Iron Pickaxe": "make an iron pickaxe with a nearby table and furnace, wood, coal, and iron in inventory",
    "Make Wood Sword": "make a wood sword with a nearby table and wood in inventory",
    "Make Stone Sword": "make a stone sword with a nearby table, wood, and stone in inventory",
    "Make Iron Sword": "make an iron sword with a nearby table and furnace, wood, coal, and iron in inventory",
}


def get_instruction_prompt(task=None):
    action_strings = ",\n".join(f"{action}: {ACTION_DICT[action]}" for action in ACTIONS)
    instruction_prompt = f"""
You are an agent playing Crafter. The following are the possible actions you can take in the game:

{action_strings}.

The following are the game achievements you get in the game:
1. Collect Wood
2. Place Table
3. Eat Cow
4. Collect Sampling
5. Collect Drink
6. Make Wood Pickaxe
7. Make Wood Sword
8. Place Plant
9. Defeat Zombie
10. Collect Stone
11. Place Stone
12. Eat Plant
13. Defeat Skeleton
14. Make Stone Pickaxe
15. Make Stone Sword
16. Wake Up
17. Place Furnace
18. Collect Coal
19. Collect Iron
20. Make Iron Pickaxe
21. Make Iron Sword
22. Collect Diamond

In a moment I will present a history of actions and observations from the game.
Your goal is to get as far as possible by completing all the achievements.

PLAY!
""".strip()

    return instruction_prompt
