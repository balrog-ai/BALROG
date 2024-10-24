from balrog.client import create_llm_client

from ..prompt_builder import create_prompt_builder
from .chain_of_thought import ChainOfThoughtAgent
from .naive import NaiveAgent
from .icl import ICLAgent
from .self_refine import SelfRefineAgent
from .dummy import DummyAgent


class AgentFactory:
    def __init__(self, config):
        self.config = config

    def create_agent(self):
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
