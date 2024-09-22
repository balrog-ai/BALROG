# Here we should have an environment manager function that can be used to instantiate
# environments with the correct wrappers.
import gym
from gym import spaces

from iclbench.environments.env_wrapper import EnvWrapper


def make_env(env_name, task, config):

    if env_name == "nle":
        from iclbench.environments.nle import NLELanguageWrapper

        vlm = True if config.agent.max_image_history > 0 else False
        env = gym.make(task, **config.envs.nle_kwargs)
        base_env = NLELanguageWrapper(env, vlm=vlm)
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
                **config.envs.minihack_kwargs,
            ),
            **config.env_kwargs,
        )
    elif env_name == "babyai":
        from iclbench.environments.babyai_text import BabyAITextCleanLangWrapper

        base_env = BabyAITextCleanLangWrapper(task, **config.envs.babyai_kwargs)
    elif env_name == "craftax":
        from iclbench.environments.craftax import CraftaxLanguageWrapper

        base_env = CraftaxLanguageWrapper(task, **config.envs.craftax_kwargs)
    elif env_name == "textworld":
        from iclbench.environments.textworld import TextWorldFactory

        textworld_factory = TextWorldFactory(tasks=config.tasks.textworld_tasks, **config.envs.textworld_kwargs)
        base_env = textworld_factory(task, **config.envs.env_kwargs)
    elif env_name == "babaisai":
        from baba import make

        from iclbench.environments.baba_is_ai import BabaIsAIWrapper

        base_env = BabaIsAIWrapper(make(task, **config.envs.babaisai_kwargs))
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    return EnvWrapper(base_env, env_name, task)


class Strings(spaces.Space):
    def __init__(self, values, seed=None):
        super().__init__((len(values),), str, seed)
        self._dict = {value: i for i, value in enumerate(values)}
        self._values = values

    def sample(self):
        return self.np_random.choice(self._values)

    def map(self, action):
        return self._dict[action]

    def contains(self, value):
        return value in self._dict

    def __iter__(self):
        return self._values.__iter__()
