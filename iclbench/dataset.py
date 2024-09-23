import copy
import pickle
from pathlib import Path


class InContextDataset:
    def __init__(self, config, env_name, original_cwd) -> None:
        self.config = config
        self.env_name = env_name
        self.original_cwd = original_cwd

    def demo_path(self, i, task):
        demos_dir = Path(self.original_cwd) / self.config.eval.icl_dataset / self.env_name / task
        demo_path = list(demos_dir.iterdir())[i]

        return demo_path

    def load_incontext_config(self, i, task):
        # TODO: if needed we should use the same character as in nethack
        demo_path = self.demo_path(i, task)
        seed = int(demo_path.stem.split("_")[1])
        episode_config = copy.deepcopy(self.config)
        episode_config.envs.env_kwargs.seed = seed

        return episode_config

    def load_incontext_actions(self, i, task):
        demo_path = self.demo_path(i, task)

        with open(demo_path, "rb") as f:
            data = pickle.load(f)

        recorded_actions = data["actions"]

        return recorded_actions
