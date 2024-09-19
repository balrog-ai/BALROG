from iclbench.agents.base import BaseAgent


class NaiveAgent(BaseAgent):
    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)

        input = self.prompt_builder.get_prompt()

        # Add Naive instructions to the prompt
        naive_instruction = """
You can only output one of the above actions at a time, and always have to output an action until the episode terminates.
Action: 
        """.strip()
        input[-1]["parts"][0] += "\n\n" + naive_instruction

        completion = self.client.generate(input)

        return completion
