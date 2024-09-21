from .builders.history import HistoryPromptBuilder


def create_prompt_builder(config):
    # if config.vlm:
    return HistoryPromptBuilder(
        max_history=config.max_history,
        max_image_history=config.max_image_history,
        max_cot_history=config.max_cot_history,
    )
