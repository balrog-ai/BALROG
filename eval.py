import logging
from collections import defaultdict

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from iclbench.agents import AgentFactory
from iclbench.evaluator import EvaluatorManager
from iclbench.utils import setup_environment, summarize_env_progressions, wandb_save_artifact


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(config: DictConfig):
    original_cwd = get_original_cwd()
    setup_environment(original_cwd=original_cwd)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Instantiate agent factory
    agent_factory = AgentFactory(config)

    # Create an EvaluatorManager
    evaluator_manager = EvaluatorManager(config, original_cwd=original_cwd)
    results_summaries = evaluator_manager.run(agent_factory)

    average_progression = summarize_env_progressions(results_summaries, config)
    print(f"Average progression across all environments: {average_progression}")

    if config.eval.wandb_save:
        wandb_save_artifact(config)


if __name__ == "__main__":
    main()
