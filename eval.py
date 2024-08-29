import logging
import json
import hydra
from omegaconf import DictConfig
from openai import OpenAI
from iclbench.agents import create_agent, DummyAgent
from iclbench.evaluator import Evaluator


@hydra.main(config_path="config", config_name="eval")
def main(config: DictConfig):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # pure filth, I know
    if config.agent == "dummy":
        agent_factory = DummyAgent
    else:
        # Instantiate LLM client
        client = OpenAI(
            api_key="EMPTY", base_url=config.base_url, timeout=config.client.timeout
        )

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
