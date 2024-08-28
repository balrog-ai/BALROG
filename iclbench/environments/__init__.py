# Here we should have an environment manager function that can be used to instantiate
# environments with the correct wrappers.
import gym


def make_env(env_name, **kwargs):
    if env_name == "nle":
        from iclbench.environments.nle import NLELanguageWrapper

        return NLELanguageWrapper(gym.make("NetHackChallenge-v0"), **kwargs)
    elif env_name == "babyai":
        import babyai_text
        from iclbench.environments.babyai_text import BabyAITextCleanLangWrapper

        return BabyAITextCleanLangWrapper(gym.make("BabyAI-MixedTrainLocal-v0", **kwargs))
    elif env_name == "craftax":
        raise NotImplementedError("Craftax environment is not supported yet.")
    elif env_name == "minihack":
        raise NotImplementedError("MiniHack environment is not supported yet.")
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def get_instruction_prompt(env_name, **kwargs):
    if env_name == "nle":
        from iclbench.environments.nle import get_instruction_prompt

        return get_instruction_prompt()

    elif env_name == "babyai":
        from iclbench.environments.babyai_text import get_instruction_prompt

        return get_instruction_prompt(**kwargs)
    elif env_name == "craftax":
        raise NotImplementedError("Craftax environment is not supported yet.")
    elif env_name == "minihack":
        raise NotImplementedError("MiniHack environment is not supported yet.")
    else:
        raise ValueError(f"Unknown environment: {env_name}")
