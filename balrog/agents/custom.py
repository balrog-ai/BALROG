from balrog.agents.base import BaseAgent
import re
import copy


class CustomAgent(BaseAgent):
    def __init__(self, client_factory, prompt_builder, config):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.plan = None

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)

        if self.plan:
            plan_text = f"Current Plan:\n{self.plan}\n"
        else:
            plan_text = "You have no plan yet.\n"

        planning_instructions = """
Review the current plan above if present. Decide whether to continue with it or make changes.
If you make changes, provide the updated plan.
Then, provide the next action to take.
You must output an action at every step.
Format your answer in the following way:
PLAN: <your updated plan if changed, or "No changes to the plan.">
ACTION: <your next action>
        """.strip()

        messages = self.prompt_builder.get_prompt()
        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + plan_text + "\n" + planning_instructions

        response = self.client.generate(messages)

        # Extract the plan and action from the LLM's response
        plan, action = self._extract_plan_and_action(response.completion)

        # Update the internal plan if it has changed
        if plan != "No changes to the plan.":
            self.plan = plan

        # Save the plan in the response.reasoning field and the action in response.completion
        modified_response = copy.deepcopy(response)
        modified_response = modified_response._replace(reasoning=plan)
        modified_response = modified_response._replace(completion=action)

        return modified_response

    def _extract_plan_and_action(self, response_text):
        # Initialize plan and action
        plan = "No changes to the plan."
        action = None

        # Extract PLAN and ACTION from the response
        plan_match = re.search(r"PLAN:\s*(.*?)(?=\nACTION:|\Z)", response_text, re.IGNORECASE | re.DOTALL)
        action_match = re.search(r"ACTION:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)

        if plan_match:
            plan_content = plan_match.group(1).strip()
            if plan_content:
                plan = plan_content

        if action_match:
            action = action_match.group(1).strip()

        return plan, action
