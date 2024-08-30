import logging
import json
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

    print(config)

    # Instantiate LLM client
    client = create_llm_client(config.client)

    # Instantiate factory for creating agents
    agent_factory = create_agent(client, config)

    results = []
    for env_name in config.env_names.split(","):
        evaluator = Evaluator(env_name, agent_factory, config)
        results.extend(evaluator.run())

    # Save results
    with open(config.savedir, "w") as file:
        json.dump(results, file, indent=4)


if __name__ == "__main__":
    main()
