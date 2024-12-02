import copy
import glob
import os
import random
import re
from pathlib import Path

import numpy as np


def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", str(s))]


def choice_excluding(lst, excluded_element):
    possible_choices = [item for item in lst if item != excluded_element]
    return random.choice(possible_choices)


class InContextDataset:
    def __init__(self, config, env_name, original_cwd) -> None:
        self.config = config
        self.env_name = env_name
        self.original_cwd = original_cwd

    def icl_episodes(self, task):
        demos_dir = Path(self.original_cwd) / self.config.eval.icl_dataset / self.env_name / task
        return list(sorted(glob.glob(os.path.join(demos_dir, "**/*.npz"), recursive=True), key=natural_sort_key))

    def check_seed(self, demo_path):
        return int(demo_path.stem.split("seed_")[1])

    def demo_task(self, task):
        # use different task - avoid the case where we put the solution into the context
        if self.env_name == "babaisai":
            task = choice_excluding(self.config.tasks[f"{self.env_name}_tasks"], task)

        return task

    def demo_path(self, i, task, demo_config):
        icl_episodes = self.icl_episodes(task)
        demo_path = icl_episodes[i % len(icl_episodes)]

        # use the same role
        if self.env_name == "nle":
            from balrog.environments.nle import Role

            character = demo_config.envs.nle_kwargs.character
            if character != "@":
                for part in character.split("-"):
                    # check if there is specified role
                    if part.lower() in [e.value for e in Role]:
                        # check if we have games played with this role
                        new_demo_paths = [path for path in icl_episodes if part.lower() in path.stem.lower()]
                        if new_demo_paths:
                            demo_path = random.choice(new_demo_paths)

        # use different seed - avoid the case where we put the solution into the context
        if self.env_name == "textworld":
            from balrog.environments.textworld import global_textworld_context

            textworld_context = global_textworld_context(
                tasks=self.config.tasks.textworld_tasks, **self.config.envs.textworld_kwargs
            )
            next_seed = textworld_context.count[task]
            demo_seed = self.check_seed(demo_path)
            if next_seed == demo_seed:
                demo_path = self.icl_episodes(task)[i + 1]

        return demo_path

    def load_episode(self, filename):
        # Load the compressed NPZ file
        with np.load(filename, allow_pickle=True) as data:
            # Convert to dictionary if you want
            episode = {k: data[k] for k in data.files}
        return episode

    def load_in_context_learning_episode(self, i, task, agent):
        demo_config = copy.deepcopy(self.config)
        demo_task = self.demo_task(task)
        demo_path = self.demo_path(i, demo_task, demo_config)
        episode = self.load_episode(demo_path)

        actions = episode.pop("action")
        rewards = episode.pop("reward")
        terminated = episode.pop("terminated")
        truncated = episode.pop("truncated")
        dones = np.any([terminated, truncated], axis=0)
        observations = [dict(zip(episode.keys(), values)) for values in zip(*episode.values())]

        for observation, action, reward, done in zip(observations, actions, rewards, dones):
            action = str(action)
            if action == "":
                action = None

            agent.update_icl_observation(observation)
            agent.update_icl_action(action)

            if done:
                break

        if not done:
            print("warning: icl trajectory ended without done")

        agent.wrap_episode()