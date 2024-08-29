# Here we should have an environment manager function that can be used to instantiate environments with the correct wrappers.

def make(env_id, **env_kwargs):
    # Sam: I'm going with lazy imports here because I'm getting annoying docker errors with NLE
    #      again and it's not prudent for me to fix that right now.
    if env_id == "NetHack":
        import gym
        from iclbench.environments.nle import NLELanguageWrapper
        
        return NLELanguageWrapper(gym.make("NetHackChallenge-v0"), **env_kwargs)
    if env_id == "Craftax":
        from iclbench.environments.craftax import CraftaxLanguageWrapper
        
        return CraftaxLanguageWrapper("Craftax-Symbolic-v1", **env_kwargs)
    raise ValueError(f"Unknown environment {env_id}.")