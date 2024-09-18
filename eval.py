import logging
import hydra
from collections import defaultdict
from omegaconf import DictConfig
from iclbench.agents import AgentFactory
from iclbench.evaluator import Evaluator
from iclbench.utils import summarize_env_progressions, wandb_save_artifact


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(config: DictConfig):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Instantiate agent factory
    agent_factory = AgentFactory(config)

    results_summaries = defaultdict(list)
    for env_name in config.env_names.split(","):
        evaluator = Evaluator(env_name, config)
        env_result_summary = evaluator.run(agent_factory)
        results_summaries[env_name] = env_result_summary

    average_progression = summarize_env_progressions(results_summaries, config)
    print(f"Average progression across all environments: {average_progression}")

    if config.wandb_save:
        wandb_save_artifact(config)


if __name__ == "__main__":
    main()
