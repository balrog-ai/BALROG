from iclbench.client import create_llm_client

from ..prompt_builder import create_prompt_builder
from .chain_of_thought import ChainOfThoughtAgent
from .naive import NaiveAgent
from .self_refine import SelfRefineAgent


class AgentFactory:
    def __init__(self, config):
        self.config = config

    def create_agent(self):
        client_factory = create_llm_client(self.config.client)
        prompt_builder = create_prompt_builder(self.config.prompt_builder_config)

        if self.config.agent == "naive":
            return NaiveAgent(client_factory, prompt_builder)
        elif self.config.agent == "cot":
            return ChainOfThoughtAgent(client_factory, prompt_builder)
        elif self.config.agent == "self_refine":
            return SelfRefineAgent(
                client_factory, prompt_builder, max_iterations=self.config.self_refine_max_iterations
            )
        elif self.config.agent == "react":
            raise NotImplementedError("ReAct agent is not implemented yet.")
        elif self.config.agent == "reflexion":
            raise NotImplementedError("Reflexion agent is not implemented yet.")
        else:
            raise ValueError(f"Unknown agent type: {self.config.agent}")
