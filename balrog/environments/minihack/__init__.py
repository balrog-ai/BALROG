from nle.language_wrapper.wrappers.nle_language_wrapper import NLELanguageWrapper

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
    "up": "go up the stairs",
    "down": "go down the stairs",
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
    available_actions = {}
    for action in env.actions:
        action_key = NLELanguageWrapper.all_nle_action_map[action][0]
        if action_key not in ACTIONS:
            continue
        available_actions[action_key] = ACTIONS[action_key]
    return available_actions


def get_instruction_prompt(env, task="MiniHack-ExploreMaze-Hard-Mapped-v0"):
    if "corridor" in task.lower():
        goal = "Your goal is to explore the level and reach the stairs down"
    elif "quest" in task.lower():
        goal = "Your goal is to explore the level, fight monsters, and navigate rooms and mazes to ultimately reach the stairs down."
    elif "boxoban" in task.lower():
        goal = "You are playing Boxoban, a box-pushing game inspired by Sokoban. Your goal is to push the boulders onto the fountains on the map. You can push the boulders by walking into them, as long as there are no obstacles behind them."
    else:
        goal = "Your goal is to get as far as possible in the game."

    available_actions = get_available_actions(env)
    action_strings = ",\n".join(f"{action}: {description}" for action, description in available_actions.items())
    instruction_prompt = f"""
You are an agent playing MiniHack. The following are the possible actions you can take in the game, followed by a short description of each action:

{action_strings}.

In a moment I will present a history of actions and observations from the game.

Tip: there is no point in outputting the same action over and over if nothing changes.

{goal}

PLAY!
""".strip()

    return instruction_prompt
