from .builders.history import HistoryPromptBuilder
from .builders.vlm import VLMHistoryPromptBuilder


def create_prompt_builder(config):
    if config.vlm:
        raise NotImplementedError("VLM prompt builder is not implemented yet.")

    return HistoryPromptBuilder(
        max_history=config.max_history,
        max_length=config.max_length,
        diff=config.diff,
        use_history=config.history,
    )
