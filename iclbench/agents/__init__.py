from .dummy import DummyAgent
from .naive import NaiveAgent
from ..prompt_builder import create_prompt_builder

def create_agent(client_llm, config):
    prompt_builder = create_prompt_builder(config.prompt_builder)
    return NaiveAgent(client_llm, prompt_builder, config)
