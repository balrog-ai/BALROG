import logging
from collections import defaultdict

from iclbench.agents.base import BaseAgent


class NaiveAgent(BaseAgent):
    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        # Update observation in the prompt builder
        self.prompt_builder.update_observation(obs["text"])

        input = self.prompt_builder.get_prompt()
        print(input[0]["content"] if isinstance(input, list) else input)

        completion = self.client.generate(input)

        return completion
