import pytest
from hydra import compose, initialize

from iclbench.agents import AgentFactory
from iclbench.evaluator import Evaluator

agents = [
    "naive",
]

environments = [
    # "nethack",
    # "minihack",
    "babyai",
    # "textworld",
    # "babaisai",
    # "craftax",
]

clients = [
    "flair",
    # "gemini",
]

@pytest.mark.parametrize("agent", agents)
@pytest.mark.parametrize("environment", environments)
@pytest.mark.parametrize("client", clients)
@pytest.mark.parametrize("vlm", [True])
def test_evaluation(agent, environment, client, vlm):
    with initialize(config_path="../config", version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                f"agent={agent}",
                f"env_names={environment}",
                f"client={client}",
                f"vlm={vlm}",
                # to reduce computational footprint of the tests
                f"num_episodes={1}",
                f"num_workers={1}",
                f"max_steps_per_episode={5}",
            ],
        )

        # Check that the config is correct
        assert cfg.agent == agent
        assert cfg.env_names == environment
        assert cfg.vlm == vlm

        # Run evaluation
        env_name = cfg.env_names.split(",")[0]
        # we could pass task name as an argument, for now just use the first task
        cfg[f"{env_name}_tasks"] = cfg[f"{env_name}_tasks"][:1]
        evaluator = Evaluator(env_name, cfg)
        agent_factory = AgentFactory(cfg)
        evaluator.run(agent_factory)
