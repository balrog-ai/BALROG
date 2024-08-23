from .naive import NaiveAgent
from ..prompt_builder import create_prompt_builder
from ..environments import get_instruction_prompt


def create_agent(client_llm, config):
    def agent_factory():
        prompt_builder = create_prompt_builder(config.prompt_builder_strategy)

        if config.agent == "naive":
            return NaiveAgent(client_llm, prompt_builder, config)
        elif config.agent == "react":
            raise NotImplementedError("ReAct agent is not implemented yet.")
        elif config.agent == "reflexion":
            raise NotImplementedError("Reflexion agent is not implemented yet.")
        else:
            raise ValueError(f"Unknown agent type: {config.agent}")

    return agent_factory
