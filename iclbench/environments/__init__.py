import nle
import craftax


def create_env(config):
    if config.environment == "NetHack":
        env = nle.NLE(**config.env_kwargs)
        
    elif config.environment == "Craft":
        