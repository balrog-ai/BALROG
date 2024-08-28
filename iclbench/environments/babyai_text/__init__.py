from .clean_lang_wrapper import BabyAITextCleanLangWrapper

ACTIONS = {
    "turn left": "turn to the left",
    "turn right": "turn to the right",
    "go forward": "take one step forward",
    "pick up": "pick up the object below you",
    "drop": "drop the object that you are holding",
    "toggle": "manipulate the object in front of you",
}


def get_instruction_prompt(**kwargs):
    action_strings = ",\n".join(
        f"{action}: {description}" for action, description in ACTIONS.items()
    )

    assert "mission" in kwargs, "A text specification of the current game goal must be passed to the instruction builder in BabyAI-Text!"

    instruction_prompt = f"""
You are an agent playing a simple navigation game. Your goal is to {kwargs['mission']}. The following are the possible actions you can take in the game, followed by a short description of each action:

{action_strings}.

In a moment I will present you an observation.
You can only output one of the above actions at a time. Your goal is to get as far as possible in the game.
""".strip()

    return instruction_prompt
