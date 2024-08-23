from .builders.diff import DiffHistoryPromptBuilder
from .builders.zeroshot import ZeroShotPromptBuilder
from .builders.history import HistoryPromptBuilder
from .builders.vlm import VLMHistoryPromptBuilder


def create_prompt_builder(strategy):
    if strategy == "zero-shot":
        return ZeroShotPromptBuilder()
    elif strategy == "history":
        return HistoryPromptBuilder()
    elif strategy == "diff_history":
        return DiffHistoryPromptBuilder()
    elif strategy == "vlm":
        return VLMHistoryPromptBuilder()
    else:
        raise ValueError(f"Unknown prompt_builder_strategy: {strategy}")
