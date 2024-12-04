import glob
import logging
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

    def extract_seed(self, demo_path):
        # extract seed from record, example format: `20241201T225823-seed13-rew1.00-len47.npz`
        seed = [part.removeprefix("seed") for part in Path(demo_path).stem.split("-") if "seed" in part]
        return int(seed[0])

    def demo_task(self, task):
        # use different task - avoid the case where we put the solution into the context
        if self.env_name == "babaisai":
            task = choice_excluding(self.config.tasks[f"{self.env_name}_tasks"], task)

        return task

    def demo_path(self, i, task):
        icl_episodes = self.icl_episodes(task)
        demo_path = icl_episodes[i % len(icl_episodes)]

        # use different seed - avoid the case where we put the solution into the context
        if self.env_name == "textworld":
            from balrog.environments.textworld import global_textworld_context

            textworld_context = global_textworld_context(
                tasks=self.config.tasks.textworld_tasks, **self.config.envs.textworld_kwargs
            )
            next_seed = textworld_context.count[task]
            demo_seed = self.extract_seed(demo_path)
            if next_seed == demo_seed:
                demo_path = self.icl_episodes(task)[i + 1]

        return demo_path

    def load_episode(self, filename):
        # Load the compressed NPZ file
        with np.load(filename, allow_pickle=True) as data:
            # Convert to dictionary if you want
            episode = {k: data[k] for k in data.files}
        return episode

    def load_in_context_learning_episodes(self, num_episodes, task, agent):
        demo_task = self.demo_task(task)
        demo_paths = [self.demo_path(i, demo_task) for i in range(len(self.icl_episodes(task)))]
        random.shuffle(demo_paths)
        demo_paths = demo_paths[:num_episodes]

        for demo_path in demo_paths:
            self.load_in_context_learning_episode(demo_path, agent)

    def load_in_context_learning_episode(self, demo_path, agent):
        episode = self.load_episode(demo_path)

        actions = episode.pop("action").tolist()
        rewards = episode.pop("reward").tolist()
        terminated = episode.pop("terminated")
        truncated = episode.pop("truncated")
        dones = np.any([terminated, truncated], axis=0).tolist()
        observations = [dict(zip(episode.keys(), values)) for values in zip(*episode.values())]

        # first transition only contains observation (like env.reset())
        observation, action, reward, done = observations.pop(0), actions.pop(0), rewards.pop(0), dones.pop(0)
        agent.update_icl_observation(observation)

        for observation, action, reward, done in zip(observations, actions, rewards, dones):
            action = str(action)

            agent.update_icl_action(action)
            agent.update_icl_observation(observation)

            if done:
                break

        if not done:
            logging.info("icl trajectory ended without done")

        agent.wrap_episode()
