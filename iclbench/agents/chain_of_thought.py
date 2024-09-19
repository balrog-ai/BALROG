import copy
import re

from iclbench.agents.base import BaseAgent
from iclbench.client import LLMClientWrapper


class ChainOfThoughtAgent(BaseAgent):
    def __init__(self, client_factory: LLMClientWrapper, prompt_builder):
        super().__init__(client_factory, prompt_builder)

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)

        messages = self.prompt_builder.get_prompt()

        # Add CoT-specific instructions to the prompt
        cot_instructions = """
Let's approach this step-by-step:
1. Analyze the given information
2. Break down the problem into smaller parts
3. Reason about each part
4. Combine the insights to form a conclusion
You can only output one of the above actions at a time, and always have to output an action until the episode terminates.
Please provide reasoning first and only after reasoning provide the final answer in the form of Action: <action>
        """.strip()

        messages[-1].content += "\n\n" + cot_instructions

        # Generate the CoT reasoning
        cot_reasoning = self.client.generate(messages)

        # Extract the final answer from the CoT reasoning
        final_answer = self._extract_final_answer(cot_reasoning)

        return final_answer

    def _extract_final_answer(self, reasoning):
        def filter_letters(input_string):
            return re.sub(r"[^a-zA-Z\s:]", "", input_string)

        answer = copy.deepcopy(reasoning)

        answer.reasoning = answer.completion
        answer.completion = filter_letters(answer.completion).split("ACTION:")[-1].strip()

        return answer
