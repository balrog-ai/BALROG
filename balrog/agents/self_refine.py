import copy
import re

from balrog.agents.base import BaseAgent
from balrog.client import LLMClientWrapper


class SelfRefineAgent(BaseAgent):
    def __init__(self, client_factory: LLMClientWrapper, prompt_builder, max_iterations=3):
        super().__init__(client_factory, prompt_builder)
        self.max_iterations = max_iterations

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)

        input = self.prompt_builder.get_prompt()

        # Add initial instructions to the prompt
        initial_instructions = """
Please provide an answer to the given question or task. After your response, I will provide feedback for improvement.
You can only output one of the above actions at a time, and always have to output an action until the episode terminates.
Please provide reasoning first and only after reasoning provide the final answer in the form of Action: <action>
        """.strip()
        input[-1]["parts"][0] += "\n\n" + initial_instructions

        # Initial response generation
        response = self.client.generate(input)
        # Self-refine loop
        for _ in range(self.max_iterations):
            # Generate feedback
            feedback_prompt = """
"You are an AI assistant tasked with providing feedback on the following response. Analyze the response for clarity, completeness, and accuracy.
Suggest specific improvements or write "No further improvements needed" if no further improvements are needed.

Response to evaluate:"
            """.strip()

            feedback_input = [
                *input,
                {"role": "user", "parts": [feedback_prompt + response.choices[0].message.content]},
            ]

            feedback_response = self.client.generate(feedback_input)

            # If feedback suggests no further improvements, break the loop
            if "No further improvements needed" in feedback_response.choices[0].message.content:
                break

            # Add feedback and refinement instructions to the prompt
            refine_prompt = f"""
Feedback: {feedback_response.choices[0].message.content}

Please refine your previous response based on this feedback.
You can only output one of the above actions at a time, and always have to output an action until the episode terminates.
Please provide reasoning first and only after reasoning provide the final answer in the form of Action: <action>
            """.strip()
            input.append({"role": "user", "parts": [refine_prompt]})

            # Generate refined response
            response = self.client.generate(input)
        # Extract the final answer from the refined response
        final_answer = self._extract_final_answer(response)

        return final_answer

    def _extract_final_answer(self, reasoning):
        def filter_letters(input_string):
            return re.sub(r"[^a-zA-Z\s:]", "", input_string)

        answer = copy.deepcopy(reasoning)
        answer = answer._replace(reasoning=answer.completion)
        answer = answer._replace(completion=filter_letters(answer.completion).split("ACTION:")[-1].strip())

        return answer
