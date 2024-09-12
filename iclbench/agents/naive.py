import logging
from collections import defaultdict

from iclbench.agents.base import BaseAgent


class NaiveAgent(BaseAgent):
    def __init__(self, client, prompt_builder):
        super().__init__()
        self.client = client
        self.prompt_builder = prompt_builder

        self.failed_generation_counter = 0
        self.action_history = []
        self.action_frequency = defaultdict(int)

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        # Update observation in the prompt builder
        self.prompt_builder.update_observation(obs["text"])

        input = self.prompt_builder.get_prompt()
        print(input[0]["content"] if isinstance(input, list) else input)

        completion = self.client.generate(input)

        return completion

    def update_prompt(self, observation, action):
        self.prompt_builder.update_observation(observation)
        self.prompt_builder.update_action(action)
