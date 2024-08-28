# Here we should have an environment manager function that can be used to instantiate
# environments with the correct wrappers.
import gym


from iclbench.environments.env_wrapper import EnvWrapper


def make_env(env_name, task, **kwargs):
    if env_name == "nle":
        from iclbench.environments.nle import NLELanguageWrapper

        base_env = NLELanguageWrapper(gym.make(task), **kwargs)
    elif env_name == "minihack":
        import minihack
        from iclbench.environments.nle import NLELanguageWrapper

        base_env = NLELanguageWrapper(
            gym.make(
                task,
                observation_keys=[
                    "glyphs",
                    "blstats",
                    "tty_chars",
                    "inv_letters",
                    "inv_strs",
                    "tty_cursor",
                    "tty_colors",
                ],
            ),
            **kwargs,
        )
    elif env_name == "babyai":
        # Placeholder for BabyAI environment
        raise NotImplementedError("BabyAI environment is not supported yet.")
    elif env_name == "craftax":
        # Placeholder for Craftax environment
        raise NotImplementedError("Craftax environment is not supported yet.")
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    return EnvWrapper(base_env, env_name, task)


def get_tasks(env_name):
    if env_name == "nle":
        from iclbench.environments.nle import TASKS as NLE_TASKS

        return NLE_TASKS
    elif env_name == "minihack":
        from iclbench.environments.minihack import TASKS as MINIHACK_TASKS

        return MINIHACK_TASKS
    elif env_name == "babyai":
        raise NotImplementedError("BabyAI environment is not supported yet.")
    elif env_name == "craftax":
        raise NotImplementedError("Craftax environment is not supported yet.")
    else:
        raise ValueError(f"Unknown environment: {env_name}")
