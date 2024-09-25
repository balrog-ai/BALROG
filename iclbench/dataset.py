import copy
import pickle
import re
from pathlib import Path

from iclbench.environments.textworld import global_textworld_context


def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", str(s))]


class InContextDataset:
    def __init__(self, config, env_name, original_cwd) -> None:
        self.config = config
        self.env_name = env_name
        self.original_cwd = original_cwd

    def icl_episodes(self, task):
        demos_dir = Path(self.original_cwd) / self.config.eval.icl_dataset / self.env_name / task
        return list(sorted(demos_dir.iterdir(), key=natural_sort_key))

    def check_seed(self, demo_path):
        return int(demo_path.stem.split("seed_")[1])

    def demo_path(self, i, task):
        demo_path = self.icl_episodes(task)[i]

        if self.env_name == "textworld":
            # avoid the case where we put the solution into the context
            textworld_context = global_textworld_context(
                tasks=self.config.tasks.textworld_tasks, **self.config.envs.textworld_kwargs
            )
            next_seed = textworld_context.count[task]
            demo_seed = self.check_seed(demo_path)
            if next_seed == demo_seed:
                demo_path = self.icl_episodes(task)[i + 1]

        return demo_path

    def load_incontext_config(self, i, task):
        episode_config = copy.deepcopy(self.config)
        demo_path = self.demo_path(i, task)
        seed = self.check_seed(demo_path)

        episode_config.envs.env_kwargs.seed = seed

        if self.env_name == "nle" or self.env_name == "minihack":
            episode_config.envs[f"{self.env_name}_kwargs"].skip_more = True

        if self.env_name == "crafter":
            episode_config.envs.crafter_kwargs.seed = seed

        return episode_config

    def load_incontext_actions(self, i, task):
        demo_path = self.demo_path(i, task)

        with open(demo_path, "rb") as f:
            data = pickle.load(f)

        recorded_actions = data["actions"]

        return recorded_actions
