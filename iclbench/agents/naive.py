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

        # Determine if the model supports chat-based input from config or model_id
        self.is_chat_model = config.get("is_chat_model", False)

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
            self.action_history.append(prev_action)
            self.action_frequency[prev_action] += 1

        # Update observation in the prompt builder
        self.prompt_builder.update_observation(obs["text"])

        input = self.prompt_builder.get_prompt()
        print(input[0]["content"] if isinstance(input, list) else input)

        # Handle action generation based on whether it's a chat model or not
        if self.is_chat_model and isinstance(input, list):  # Chat-based input
            completion = self.client.chat.completions.create(
                **self.client_kwargs, messages=input
            )
        else:  # Text-based input
            completion = self.client.completions.create(
                prompt=input[0]["content"], **self.client_kwargs
            )

        return completion
