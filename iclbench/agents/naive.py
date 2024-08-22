import logging
from collections import defaultdict

from iclbench.agents.base import BaseAgent


class NaiveAgent(BaseAgent):
    def __init__(self, client, prompt_builder, config):
        super().__init__(client, prompt_builder)
        self.failed_generation_counter = 0
        self.action_history = []
        self.action_frequency = defaultdict(int)
        self.client_kwargs = {
            "model": config["model_id"],
            **config["generate_kwargs"],
        }

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
            self.action_history.append(prev_action)
            self.action_frequency[prev_action] += 1

        self.prompt_builder.update_observation(obs["text"])

        input = self.prompt_builder.get_prompt()

        # Handle action generation based on the model
        if isinstance(input, list):  # Chat-based input
            completion = self.client.chat.completions.create(
                **self.client_kwargs, messages=input
            )
        else:  # Text-based input
            completion = self.client.completions.create(prompt=input)

        return completion
