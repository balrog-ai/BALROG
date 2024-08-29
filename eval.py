import logging
import hydra
from openai import OpenAI
from iclbench import environments as envs
from iclbench.agents import create_agent, DummyAgent
from iclbench.evaluator import Evaluator


@hydra.main(config_path="config", config_name="eval")
def main(config):

    # Access the config as you would normally with OmegaConf
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # pure filth, I know
    if config.agent == "dummy":
        agent = DummyAgent() 
    else:
        # Instantiate LLM client
        client = OpenAI(api_key="EMPTY", base_url=config.base_url)

        # Instantiate agent (TODO: support future LangChain integration)
        # This currently would not support multiprocessing, as we have a single agent here.
        agent = create_agent(client, config)
    
    # Instantiate environment
    env = envs.make(config.environment, **config.env_kwargs)

    # Instantiate evaluator and run the evaluation
    evaluator = Evaluator(env, agent, config)
    results = evaluator.run()

    # Save results
    evaluator.save_results(results, config.get("savedir", "eval.json"))


if __name__ == "__main__":
    main()
