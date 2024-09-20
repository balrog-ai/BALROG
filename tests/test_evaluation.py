import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from iclbench.agents import AgentFactory
from iclbench.evaluator import Evaluator
from iclbench.utils import setup_environment


@pytest.mark.parametrize("agent", ["naive"])
@pytest.mark.parametrize("environment", ["nle", "minihack", "babyai", "textworld", "babaisai"])  # "craftax"
@pytest.mark.parametrize(
    "client,model_id", [("gemini", "gemini-1.5-flash"), ("claude", "claude-3-5-sonnet-20240620"), ("openai", "gpt-4o")]
)
@pytest.mark.parametrize("vlm", [True])
def test_evaluation(agent, environment, client, model_id, vlm):
    with initialize(config_path="../config", version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                f"agent={agent}",
                f"env_names={environment}",
                f"client.client_name={client}",
                f"client.model_id={model_id}",
                f"vlm={vlm}",
                # to reduce computational footprint of the tests
                f"num_episodes={1}",
                f"num_workers={1}",
                f"max_steps_per_episode={5}",
            ],
            return_hydra_config=True,
        )
        gh = GlobalHydra.instance()
        assert gh.is_initialized()
        setup_environment(original_cwd=cfg.hydra.runtime.cwd)

        # Check that the config is correct
        assert cfg.agent == agent
        assert cfg.env_names == environment
        assert cfg.client.client_name == client
        assert cfg.vlm == vlm

        # Run evaluation
        env_name = cfg.env_names.split(",")[0]
        # we could pass task name as an argument, for now just use the first task
        cfg[f"{env_name}_tasks"] = cfg[f"{env_name}_tasks"][:1]
        evaluator = Evaluator(env_name, cfg)
        agent_factory = AgentFactory(cfg)
        evaluator.run(agent_factory)
