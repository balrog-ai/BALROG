from nle_language_wrapper import NLELanguageWrapper


ACTIONS = {
    "north": "move north",
    "east": "move east",
    "south": "move south",
    "west": "move west",
    "northeast": "move northeast",
    "southeast": "move southeast",
    "southwest": "move southwest",
    "northwest": "move northwest",
    "far north": "move far north",
    "far east": "move far east",
    "far south": "move far south",
    "far west": "move far west",
    "far northeast": "move far northeast",
    "far southeast": "move far southeast",
    "far southwest": "move far southwest",
    "far northwest": "move far northwest",
    "up": "go up a staircase",
    "down": "go down a staircase",
    "wait": "rest one move while doing nothing",
    "more": "display more of the message",
    "apply": "apply (use) a tool",
    "close": "close an adjacent door",
    "open": "open an adjacent door",
    "eat": "eat something",
    "force": "force a lock",
    "kick": "kick an enemy or a locked door or chest",
    "loot": "loot a box on the floor",
    "pickup": "pick up things at the current location if there are any",
    "pray": "pray to the gods for help",
    "puton": "put on an accessory",
    "quaff": "quaff (drink) something",
    "search": "search for hidden doors and passages",
    "zap": "zap a wand",
}


def get_available_actions(env):
    return {
        NLELanguageWrapper.all_nle_action_map[action][0]: ACTIONS[
            NLELanguageWrapper.all_nle_action_map[action][0]
        ]
        for action in env.actions
    }


TASKS = [
    "MiniHack-Corridor-R5-v0",
    # "MiniHack-KeyRoom-Fixed-S5-v0"
    # "KeyRoom-Dark-S15-v0",
    "MiniHack-MazeWalk-Mapped-15x15-v0",
    # "MazeWalk-Mapped-45x19-v0",
    "MiniHack-River-v0",
    # "HideNSeek-Mapped-v0",
    # "MiniHack-Memento-F4-v0",
    # "CorridorBattle-v0",
    # "MazeExplore-Easy-v0",
    # "MazeExplore-Hard-v0",
]


def get_instruction_prompt(env, task="MiniHack-ExploreMaze-Hard-Mapped-v0"):

    if "mazewalk" in task or "corridor" in task.lower():
        goal = "Your goal is to explore the level and reach the staircase down."
    elif "key" in task:
        "Your goal is to pick up the key, navigate to the door and use the key to unlock the door, reaching the staircase down within the locked room"
    elif "room" in task:
        "Your goal is to explore the room and reach the staircase down."
    else:
        goal = "Your goal is to get as far as possible in the game."

    available_actions = get_available_actions(env)
    action_strings = ",\n".join(
        f"{action}: {description}" for action, description in available_actions.items()
    )
    instruction_prompt = f"""
You are an agent playing MiniHack. The following are the possible actions you can take in the game, followed by a short description of each action:

{action_strings}.

In a moment I will present you an observation.
You can only output one of the above actions at a time. {goal}
""".strip()

    return instruction_prompt
