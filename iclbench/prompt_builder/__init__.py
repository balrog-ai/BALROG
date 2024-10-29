from .builders.history import HistoryPromptBuilder


def create_prompt_builder(config):
    """
    Creates an instance of a prompt builder based on the provided configuration.

    This function initializes a prompt builder by extracting relevant configuration
    parameters. It can be extended or modified to support different types of prompt
    builders beyond just the HistoryPromptBuilder.

    Args:
        config (Config): An object containing configuration settings, which must
            include the following keys:
            - max_history (int): Maximum number of text history entries to retain.
            - max_image_history (int): Maximum number of image history entries to retain.
            - max_cot_history (int): Maximum number of chain-of-thought history entries to retain.

    Returns:
        PromptBuilder: An instance of a prompt builder configured with the specified
            history limits and any additional parameters defined in the config.
    """
    return HistoryPromptBuilder(
        max_history=config.max_history,
        max_image_history=config.max_image_history,
        max_cot_history=config.max_cot_history,
    )
