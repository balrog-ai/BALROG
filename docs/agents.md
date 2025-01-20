# Agents

In BALROG, agents are entities that typically wrap an LLM client. They are responsible for receiving observations and selecting actions. As such, agents are internally responsible for:
1. Maintaining Observation/Action Histories: Agents keep a record of past observations and actions to support context-aware decision-making.
2. Querying LLMs: Agents send observations to the LLM, receive responses, and use those responses to decide on actions.

## Pre-built agents

BALROG ships with two pre-built agents:

| **Agent Type**          | **Description**                                                                 |
|-------------------------|---------------------------------------------------------------------------------|
| **naive**          | Outputs actions based on the current action/observation history without any additional reasoning. |
| **robust_naive**          | Outputs actions based on the current action/observation history without any additional reasoning using a more robust template. |
| **chain_of_thought** | Generates actions through step-by-step reasoning, providing a final action output. |


We encourage the community to open PRs to include more agents to BALROG.

## 🤖 Creating Custom Agents

The simple zero-shot agent in `naive.py` outputs only a single action with no extra reasoning, but this is often suboptimal. We may want the agent to analyze its situation, form and refine plans, interpret image observations, or handle history more effectively.

To build a custom agent, you’ll mainly work with:
1. `balrog/agents/custom.py` -> your custom agent file.
2. `balrog/prompt_builder/history.py` -> containing the history prompt builder, an helper class to deal with with observation/action history in prompts.

You’re free to modify or create additional files, as long as they don’t interfere with evaluation, logging, or environment processes.


### Simple Planning Agent
The following code demonstrates a custom planning agent that stores and follows a plan, updating it as needed. This agent uses the default history prompt builder.

`custom.py`
```python
from balrog.agents.base import BaseAgent
import re


class CustomAgent(BaseAgent):
    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.plan = None

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)

        plan_text = f"Current Plan:\n{self.plan}\n" if self.plan else "You have no plan yet.\n"

        planning_instructions = """
Review the current plan above if present. Decide whether to continue with it or make changes.
If you make changes, provide the updated plan. Then, provide the next action to take. 
You must output an action at every step.
Format your answer in the following way:
PLAN: <your updated plan if changed, or "No changes to the plan." if the current plan is good>
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
        response = response._replace(reasoning=plan, completion=action)
        return response

    def _extract_plan_and_action(self, response_text):
        plan_match = re.search(r"PLAN:\s*(.*?)(?=\nACTION:|\Z)", response_text, re.IGNORECASE | re.DOTALL)
        action_match = re.search(r"ACTION:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)

        plan = plan_match.group(1).strip() if plan_match else "No changes to the plan."
        action = action_match.group(1).strip() if action_match else None

        return plan, action
```


You can then evaluate the custom agent with:
```
python eval.py \
  agent.type=custom \
  agent.max_image_history=0 \
  agent.max_history=16 \
  eval.num_workers=64 \
  client.client_name=openai \
  client.model_id=gpt-4o-mini-2024-07-18
```

Check our [evaluation tutorial](https://github.com/balrog-ai/BALROG/blob/main/docs/evaluation.md) to learn more on how to evaluate your agents using BALROG.
