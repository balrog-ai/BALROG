from balrog.agents.base import BaseAgent


class NaiveAgent(BaseAgent):
    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()

    def act(self, obs, info, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs, info)

        messages = self.prompt_builder.get_prompt()

        # Add naive instructions to the last user message
        naive_instruction = """
You can only output one of the above actions at a time, and always have to output an action until the episode terminates.
Action:
        """.strip()

        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + naive_instruction

        response = self.client.generate(messages)

        return response
