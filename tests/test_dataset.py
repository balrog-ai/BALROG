import copy
import random

import numpy as np
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from balrog.dataset import InContextDataset
from balrog.environments import make_env
from balrog.utils import setup_environment

environments = [
    "nle",
    "minihack",
    "babyai",
    "crafter",
    "textworld",
    "babaisai",
]


@pytest.mark.parametrize("environment", environments)
def test_dataset(environment):
    with initialize(config_path="../config", version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                f"envs.names={environment}",
                "envs.nle_kwargs.character=Mon-Hum-Mal-Neu",
            ],
            return_hydra_config=True,
        )
        gh = GlobalHydra.instance()
        assert gh.is_initialized()
        setup_environment(original_cwd=cfg.hydra.runtime.cwd)

        dataset = InContextDataset(cfg, environment, original_cwd=cfg.hydra.runtime.cwd)

        for task in cfg.tasks[f"{environment}_tasks"]:
            for episode_number in range(len(dataset.icl_episodes(task))):
                demo_config = copy.deepcopy(cfg)
                demo_task = dataset.demo_task(task)
                demo_path = dataset.demo_path(episode_number, demo_task, demo_config)
                dataset.override_incontext_config(demo_config, demo_path)
                recorded_actions = dataset.load_incontext_actions(demo_path)
                recorded_rewards = dataset.load_incontext_rewards(demo_path)

                episode_seed = demo_config.envs.env_kwargs.seed

                env = make_env(environment, demo_task, demo_config, render_mode=None)

                if episode_seed is not None:
                    np.random.seed(episode_seed)
                    random.seed(episode_seed)

                obs, info = env.reset(seed=episode_seed)
                for i, (recorded_action, recorded_reward) in enumerate(zip(recorded_actions, recorded_rewards)):
                    if recorded_action is None:
                        break

                    obs, reward, terminated, truncated, info = env.step(env.get_text_action(recorded_action))
                    done = terminated or truncated

                    # assert reward == recorded_reward

                    if done:
                        break
