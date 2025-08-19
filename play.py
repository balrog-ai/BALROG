import os
import random
import readline
import timeit
from datetime import datetime
from functools import partial
from pathlib import Path
from pprint import pprint

import hydra
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from balrog.agents import AgentFactory
from balrog.environments import make_env
from balrog.evaluator import EvaluatorManager
from balrog.utils import get_unique_seed, setup_environment


def completer(text, state, commands=[]):
    options = [cmd for cmd in commands if cmd.startswith(text)]
    return options[state] if state < len(options) else None


def setup_autocomplete(completer_fn):
    readline.parse_and_bind("tab: complete")
    print("Type commands and use TAB to autocomplete.")
    print("To see strategies use command: `help`")
    readline.set_completer(completer_fn)


def get_action(env, obs):
    language_action_space = env.get_wrapper_attr("language_action_space")
    setup_autocomplete(partial(completer, commands=language_action_space))

    while True:
        command = input("> ")

        if command == "help":
            print(language_action_space)
            continue
        else:
            try:
                assert command in language_action_space
                break
            except Exception:
                print(f"Selected action '{command}' is not in action list. Please try again.")
                continue

    return command


@hydra.main(config_path="balrog/config", config_name="config", version_base="1.1")
def main(config: DictConfig):
    original_cwd = get_original_cwd()
    setup_environment(original_cwd=original_cwd)

    # Determine output directory
    if config.eval.resume_from is not None:
        output_dir = config.eval.resume_from
    else:
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{timestamp}_{config.agent.type}_{config.client.model_id.replace('/', '_')}"
        output_dir = os.path.join(config.eval.output_dir, run_name)

        # Create the directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    env_name = random.choice(config.envs.names.split("-"))
    task = random.choice(config.tasks[f"{env_name}_tasks"])
    print(f"Selected environment: {env_name}, task: {task}")

    env = make_env(env_name, task, config, render_mode="human")

    seed = config.envs.env_kwargs.seed
    if seed is None:
        seed = get_unique_seed(process_num=None, episode_idx=0)
    random.seed(seed)
    np.random.seed(seed)
    obs, info = env.reset(seed=seed)
    env.render()

    steps = 0
    reward = 0.0
    total_reward = 0.0
    action = None

    total_start_time = timeit.default_timer()
    start_time = total_start_time

    while True:
        action = get_action(env, obs)
        if action is None:
            break

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        steps += 1
        total_reward += reward

        if not (terminated or truncated):
            continue

        time_delta = timeit.default_timer() - start_time

        print("Final reward:", reward)
        print(f"Total reward: {total_reward}, Steps: {steps}, SPS: {steps / time_delta}", total_reward)
        pprint.pprint(info)

        break
    env.close()


if __name__ == "__main__":
    main()
