import copy
import pickle
import random
import re
from pathlib import Path


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
        return list(sorted(demos_dir.iterdir(), key=natural_sort_key))

    def check_seed(self, demo_path):
        return int(demo_path.stem.split("seed_")[1])

    def demo_task(self, task):
        # use different task - avoid the case where we put the solution into the context
        if self.env_name == "babaisai":
            task = choice_excluding(self.config.tasks[f"{self.env_name}_tasks"], task)

        return task

    def demo_path(self, i, task, demo_config):
        icl_episodes = self.icl_episodes(task)
        demo_path = icl_episodes[i]

        # use the same role
        if self.env_name == "nle":
            from iclbench.environments.nle import Role

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
            from iclbench.environments.textworld import global_textworld_context

            textworld_context = global_textworld_context(
                tasks=self.config.tasks.textworld_tasks, **self.config.envs.textworld_kwargs
            )
            next_seed = textworld_context.count[task]
            demo_seed = self.check_seed(demo_path)
            if next_seed == demo_seed:
                demo_path = self.icl_episodes(task)[i + 1]

        return demo_path

    def override_incontext_config(self, demo_config, demo_path):
        seed = self.check_seed(demo_path)

        demo_config.envs.env_kwargs.seed = seed

        if self.env_name == "nle" or self.env_name == "minihack":
            # dataset was collected with "more" action
            demo_config.envs[f"{self.env_name}_kwargs"].skip_more = True
            # TODO: this has to be hardcoded because of the way we've generated the trajectories
            # keep in mind this won't affect the global config, only the demo config
            demo_config.envs.nle_kwargs.character = "@"

        if self.env_name == "crafter":
            # crafter passes seed in a specific fashion
            demo_config.envs.crafter_kwargs.seed = seed

    def load_incontext_actions(self, demo_path):
        with open(demo_path, "rb") as f:
            data = pickle.load(f)

        recorded_actions = data["actions"]

        return recorded_actions
