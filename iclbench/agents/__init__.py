from iclbench.client import create_llm_client

from ..prompt_builder import create_prompt_builder
from .chain_of_thought import ChainOfThoughtAgent
from .naive import NaiveAgent
from .icl import ICLAgent
from .self_refine import SelfRefineAgent
from .dummy import DummyAgent


class AgentFactory:
    """
    A factory class for creating various types of agents based on the provided configuration.

    The `AgentFactory` class encapsulates the logic for creating agents used in the ICLBench framework.
    It generates agents of different types, each tailored for specific tasks and functionality.

    Attributes:
        config (Config): The configuration object containing settings for agent creation,
                         including agent type and client configuration.

    Args:
        config (Config): The configuration settings for the agent, which must include:
            - agent.type (str): The type of agent to create. Supported types include:
                - "naive"
                - "icl"
                - "cot"
                - "self_refine"
                - "dummy"
            - client (dict): The client configuration for the agent.

    Methods:
        create_agent(): Creates and returns an instance of the specified agent type.
    """
    
    def __init__(self, config):
        self.config = config

    def create_agent(self):
        """
        Creates an instance of the agent based on the configuration settings.

        Returns:
            Agent: An instance of the specified agent type, which could be:
                - NaiveAgent
                - ICLAgent
                - ChainOfThoughtAgent
                - SelfRefineAgent
                - DummyAgent

        Raises:
            ValueError: If the specified agent type in the configuration is not recognized.
        """
        
        client_factory = create_llm_client(self.config.client)
        prompt_builder = create_prompt_builder(self.config.agent)

        if self.config.agent.type == "naive":
            return NaiveAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "icl":
            return ICLAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "cot":
            return ChainOfThoughtAgent(client_factory, prompt_builder, config=self.config)
        elif self.config.agent.type == "self_refine":
            return SelfRefineAgent(
                client_factory, prompt_builder, max_iterations=self.config.self_refine_max_iterations
            )
        elif self.config.agent.type == "dummy":
            return DummyAgent(client_factory, prompt_builder)
        else:
            raise ValueError(f"Unknown agent type: {self.config.agent}")
