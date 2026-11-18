"""Check family-specific budgets without constructing native games."""

import importlib
from contextlib import ExitStack
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch

from omegaconf import OmegaConf

minihack_env = importlib.import_module("balrog.environments.minihack.minihack_env")
CONFIG = Path(__file__).parents[1] / "config" / "config.yaml"


class MiniHackBudgetTests(TestCase):
    def test_minihack_episode_budget(self):
        cases = [
            ("MiniHack-Boxoban-Hard-v0", "default", 200),
            ("MiniHack-Boxoban-Medium-v0", "default", 200),
            ("MiniHack-MazeWalk-9x9-v0", "default", 100),
            ("MiniHack-Quest-Medium-v0", "default", 100),
            ("MiniHack-Boxoban-Hard-v0", "missing", 100),
            ("MiniHack-Boxoban-Hard-v0", None, 100),
            ("MiniHack-Boxoban-Hard-v0", 300, 300),
            ("MiniHack-MazeWalk-9x9-v0", 300, 100),
        ]
        for task, override, expected in cases:
            with self.subTest(task=task, override=override):
                self.check_budget(task, override, expected)

    def check_budget(self, task, override, expected):
        config = OmegaConf.load(CONFIG)
        if override == "missing":
            del config.envs.minihack_kwargs.boxoban_max_episode_steps
        elif override != "default":
            config.envs.minihack_kwargs.boxoban_max_episode_steps = override
        original = OmegaConf.to_container(config, resolve=True)
        sentinel = object()
        with ExitStack() as stack:
            make = stack.enter_context(patch.object(minihack_env.gym, "make", return_value=sentinel))
            for name in ("AutoMore", "NLELanguageWrapper", "NLETimeLimit", "GymV21CompatibilityV0"):
                stack.enter_context(patch.object(minihack_env, name, side_effect=lambda env, **kwargs: env))
            self.assertIs(minihack_env.make_minihack_env("minihack", task, config), sentinel)
        make.assert_called_once()
        args, kwargs = make.call_args
        self.assertEqual(args, (task,))
        self.assertEqual(kwargs["max_episode_steps"], expected)
        self.assertNotIn("boxoban_max_episode_steps", kwargs)
        self.assertEqual(kwargs["penalty_step"], original["envs"]["minihack_kwargs"]["penalty_step"])
        self.assertEqual(OmegaConf.to_container(config, resolve=True), original)


if __name__ == "__main__":
    main()
