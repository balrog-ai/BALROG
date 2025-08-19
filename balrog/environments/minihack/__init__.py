from nle.language_wrapper.wrappers.nle_language_wrapper import NLELanguageWrapper


def get_instruction_prompt(env, task="MiniHack-ExploreMaze-Hard-Mapped-v0"):
    if "corridor" in task.lower():
        goal = "Your goal is to explore the level and reach the stairs down"
    elif "quest" in task.lower():
        goal = "Your goal is to explore the level, fight monsters, and navigate rooms and mazes to ultimately reach the stairs down."
    elif "boxoban" in task.lower():
        goal = "You are playing Boxoban, a box-pushing game inspired by Sokoban. Your goal is to push the boulders onto the fountains on the map. You can push the boulders by walking into them, as long as there are no obstacles behind them."
    else:
        goal = "Your goal is to get as far as possible in the game."

    available_actions = env.action_str_desc_map
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
