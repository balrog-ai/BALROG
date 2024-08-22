from ..prompt_builder.builders.diff import DiffHistoryPromptBuilder
from ..prompt_builder.builders.chat import ChatPromptBuilder
from ..prompt_builder.builders.history import HistoryPromptBuilder
from ..prompt_builder.builders.vlm import VLMHistoryPromptBuilder

from iclbench.environments.nle import NLELanguageWrapper
from nle.nethack import ACTIONS


def create_prompt_builder(strategy):
    ACTION_NAMES = [
        action_strs[0]
        for action, action_strs in NLELanguageWrapper.all_nle_action_map.items()
        if action in ACTIONS
    ]
    ACTIONS_LIST_STR = ",\n".join(ACTION_NAMES)
    INSTRUCTION_PROMPT = f"""
    You are an agent playing NetHack. In a moment I will present you an observation. Only output an action from the following list:
    {ACTIONS_LIST_STR}.

    You can only output one action at a time. The goal is to maximize the reward.
    """.strip()

    if strategy == "chat":
        return ChatPromptBuilder(INSTRUCTION_PROMPT)
    elif strategy == "history":
        return HistoryPromptBuilder(INSTRUCTION_PROMPT)
    elif strategy == "diff_history":
        return DiffHistoryPromptBuilder()
    elif strategy == "vlm":
        return VLMHistoryPromptBuilder()
    else:
        raise ValueError(f"Unknown prompt_builder_strategy: {strategy}")
