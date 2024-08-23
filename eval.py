import logging
import json
from omegaconf import OmegaConf
from openai import OpenAI
from iclbench.agents import create_agent
from iclbench.evaluator import Evaluator


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load configuration
    config = OmegaConf.load("config/eval.yaml")

    # Instantiate LLM client
    client = OpenAI(api_key="EMPTY", base_url=config.base_url)

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
