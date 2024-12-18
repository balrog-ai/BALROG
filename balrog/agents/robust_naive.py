import copy
import re

from balrog.agents.base import BaseAgent


class RobustNaiveAgent(BaseAgent):
    """An agent that generates actions based on observations without complex reasoning."""

    def __init__(self, client_factory, prompt_builder):
        """Initialize the NaiveAgent with a client and prompt builder."""
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()

    def act(self, obs, prev_action=None):
        """Generate the next action based on the observation and previous action.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            str: The selected action from the LLM response.
        """
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)

        messages = self.prompt_builder.get_prompt()

        # Updated instructions to require a very strict output format
        naive_instruction = """
You must choose exactly one of the listed actions and output it strictly in the following format:

<|ACTION|>YOUR_CHOSEN_ACTION<|END|>

Replace YOUR_CHOSEN_ACTION with the chosen action. Output no other text, explanation, or reasoning.
""".strip()

        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + naive_instruction

        response = self.client.generate(messages)
        final_answer = self._extract_final_answer(response)
        return final_answer

    def _extract_final_answer(self, answer):
        """Extract the action from the completion by looking for <|ACTION|> and <|END|> tags.

        Args:
            answer (LLMResponse): The response from the LLM.

        Returns:
            LLMResponse: The sanitized response containing just the extracted action.
        """
        completion_text = answer.completion
        # Use a regex to find the text inside <|ACTION|> and <|END|>
        match = re.search(r"<\|ACTION\|>(.*?)<\|END\|>", completion_text, re.DOTALL)
        if match:
            extracted_action = match.group(1).strip()
        else:
            # If no match is found, fallback to the original completion or handle as needed
            extracted_action = completion_text.strip()

        final_answer = copy.deepcopy(answer)
        final_answer = final_answer._replace(completion=extracted_action)

        return final_answer
