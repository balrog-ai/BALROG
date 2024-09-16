import logging
from collections import defaultdict

from iclbench.agents.base import BaseAgent


class NaiveAgent(BaseAgent):
    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.failed_generation_counter = 0
        self.action_history = []
        self.action_frequency = defaultdict(int)

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs["text"])

        input = self.prompt_builder.get_prompt()

        completion = self.client.generate(input)

        return completion
