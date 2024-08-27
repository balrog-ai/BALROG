# Here we should have an environment manager function that can be used to instantiate
# environments with the correct wrappers.
import gym


def make_env(env_name, task, **kwargs):
    if env_name == "nle":
        from iclbench.environments.nle import NLELanguageWrapper

        return NLELanguageWrapper(gym.make(task), **kwargs)
    elif env_name == "minihack":
        import minihack
        from iclbench.environments.nle import NLELanguageWrapper

        return NLELanguageWrapper(
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
        raise NotImplementedError("BabyAI environment is not supported yet.")
    elif env_name == "craftax":
        raise NotImplementedError("Craftax environment is not supported yet.")

    else:
        raise ValueError(f"Unknown environment: {env_name}")


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


def get_instruction_prompt(env, env_name, task):
    if env_name == "nle":
        from iclbench.environments.nle import get_instruction_prompt

        return get_instruction_prompt()
    elif env_name == "minihack":
        from iclbench.environments.minihack import get_instruction_prompt

        return get_instruction_prompt(env, task)
    elif env_name == "babyai":
        raise NotImplementedError("BabyAI environment is not supported yet.")
    elif env_name == "craftax":
        raise NotImplementedError("Craftax environment is not supported yet.")

    else:
        raise ValueError(f"Unknown environment: {env_name}")
