import logging
import hydra
from omegaconf import DictConfig
from iclbench.agents import create_agent
from iclbench.evaluator import Evaluator
from iclbench.client import create_llm_client


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(config: DictConfig):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Instantiate LLM client
    client = create_llm_client(config.client)

    # Instantiate factory for creating agents
    agent_factory = create_agent(client, config)

    for env_name in config.env_names.split(","):
        evaluator = Evaluator(env_name, agent_factory, config)
        results = evaluator.run()
        evaluator.save_results(results, env_name)

    # TODO:
    # - Aggregate results from all environments and save/print final stats


if __name__ == "__main__":
    main()
