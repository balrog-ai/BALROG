import logging
import gym
from omegaconf import OmegaConf
from openai import OpenAI
from iclbench.environments import make_env
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

    # Instantiate environment
    env = make_env(config.env_name, **config.env_kwargs)

    # Instantiate agent (TODO: support future LangChain integration)
    # This currently would not support multiprocessing, as we have a single agent here.
    agent = create_agent(client, config)

    # Instantiate evaluator and run the evaluation
    evaluator = Evaluator(env, agent, config)
    results = evaluator.run()

    # Save results
    evaluator.save_results(results, config.get("savedir", "eval.json"))


if __name__ == "__main__":
    main()
