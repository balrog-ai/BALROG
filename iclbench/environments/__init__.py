# Here we should have an environment manager function that can be used to instantiate
# environments with the correct wrappers.
import gym
from gym import spaces

from iclbench.environments.env_wrapper import EnvWrapper


def make_env(env_name, task, config):
    if env_name == "nle":
        from iclbench.environments.nle import NLELanguageWrapper

        base_env = NLELanguageWrapper(gym.make(task), **config.env_kwargs)
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
            **config.env_kwargs,
        )
    elif env_name == "babyai":
        import babyai_text
        from iclbench.environments.babyai_text import BabyAITextCleanLangWrapper

        base_env = BabyAITextCleanLangWrapper(
            gym.make("BabyAI-MixedTrainLocal-v0", **config.babyai_kwargs)
        )
    elif env_name == "craftax":
        # Placeholder for Craftax environment
        raise NotImplementedError("Craftax environment is not supported yet.")
    elif env_name == "textworld":
        from iclbench.environments.textworld import TextWorldFactory

        textworld_factory = TextWorldFactory(**config.textworld_kwargs)
        base_env = textworld_factory(task, **config.env_kwargs)
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
        from iclbench.environments.babyai_text import TASKS as BABYAI_TASKS

        return BABYAI_TASKS
    elif env_name == "textworld":
        from iclbench.environments.textworld import TASKS as TEXTWORLD_TASKS
        
        return TEXTWORLD_TASKS
    elif env_name == "craftax":
        raise NotImplementedError("Craftax environment is not supported yet.")
    else:
        raise ValueError(f"Unknown environment: {env_name}")


class Strings(spaces.Space):
    def __init__(self, values, seed=None):
        super().__init__((len(values),), str, seed)
        self._dict = {value: i for i, value in enumerate(values)}
        self._values = values

    def sample(self):
        return self.np_random.choice(self._values)

    def contains(self, value):
        return value in self._dict

    def __iter__(self):
        return self._values.__iter__()
