from .clean_lang_wrapper import BabyAITextCleanLangWrapper

ACTIONS = {
    "turn left": "turn to the left",
    "turn right": "turn to the right",
    "go forward": "take one step forward",
    "pick up": "pick up the object below you",
    "drop": "drop the object that you are holding",
    "toggle": "manipulate the object in front of you",
}

TASKS = [
    "BabyAI-MixedTrainLocal-v0",
]


def get_instruction_prompt(env, mission="BabyAI-MixedTrainLocal-v0"):

    action_strings = ",\n".join(
        f"{action}: {description}" for action, description in ACTIONS.items()
    )

    instruction_prompt = f"""
You are an agent playing a simple navigation game. Your goal is to {mission}. The following are the possible actions you can take in the game, followed by a short description of each action:

{action_strings}.

In a moment I will present you an observation.
You can only output one of the above actions at a time.

Tip: Once you the desired object you want to interact or pickup in front of you, you can use the 'toggle' action to interact with it.

PLAY!
""".strip()

    return instruction_prompt
