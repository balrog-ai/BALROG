# Here we should have an environment manager function that can be used to instantiate
# environments with the correct wrappers.
import gym


def make_env(env_name, **kwargs):
    if env_name == "nle":
        from iclbench.environments.nle import NLELanguageWrapper

        return NLELanguageWrapper(gym.make("NetHackChallenge-v0"), **kwargs)
    elif env_name == "babyai":
        raise NotImplementedError("BabyAI environment is not supported yet.")
    elif env_name == "craftax":
        raise NotImplementedError("CraftAssist environment is not supported yet.")
    elif env_name == "minihack":
        raise NotImplementedError("MiniHack environment is not supported yet.")
    else:
        raise ValueError(f"Unknown environment: {env_name}")
