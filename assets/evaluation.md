# Submission Tutorial

Davide Paglieri • October 30, 2024

In this tutorial we show how to create custom agents and evaluate them using BALROG.


## 🤖 Creating custom agents

The simple naive zero-shot agent is tasked to only output a single action, and no other text. However, this is suboptimal, as we may want to to allow the the agent to first think about its situation, create plans, refine them, observe and describe the image observations before acting on them, or deal with history in more sophisticated ways.

The main two files you may want to modify to create a custom agent, are:
1. `balrog/agents/custom.py` -> Your custom agent
2. `balrog/prompt_builder/history.py` -> The prompt builder managing the history of observations/actions to the user (optional)

However, you can modify or create any new files you want, as long as they don't interfere with the evaluation, logging, and environments.


### Simple Planning Agent
The following is an example for a custom planning agent, that stores a plan, and at each timestep can either propose a new plan, or follow the current plan. We use here the default history prompt builder.  

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

Feel free to create more complex examples, and experiment with repositories like LangGraph and more. Users are encouraged to propose new reasoning templates by opening a PR with their implementations.


## 🛜 Evaluate using API
You can either export the API key, with one of the following:

```
export OPENAI_API_KEY=<KEY>
export ANTHROPIC_API_KEY=<KEY>
export GEMINI_API_KEY=<KEY>
```

Or you can modify the `SECRETS` file, adding your api keys.

Then run the evaluation with:

```
python eval.py \
  agent.type=custom \
  agent.max_image_history=0 \
  eval.num_workers=16 \
  client.client_name=openai \
  client.model_id=gpt-4o-mini-2024-07-18
```

You can activate the VLM mode by increasing the `max_image_history` argument, for example

```
python eval.py \
  agent.type=custom \
  agent.max_image_history=1 \
  eval.num_workers=16 \
  client.client_name=openai \
  client.model_id=gpt-4o-mini-2024-07-18
```

## ⚡️ Evaluate using vLLM locally
We support the use of vLLM for evaluating LLM/VLM locally.

```
pip install vllm
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --port 8080

python eval.py \
  agent.type=naive \
  agent.max_image_history=0 \
  eval.num_workers=16 \
  client.client_name=vllm \
  client.model_id=meta-llama/Meta-Llama-3.1-8B-Instruct \
  client.base_url=http://0.0.0.0:8080/v1
```


## ⚙️ Resume an evaluation
If for for whatever reason your evaluation was stopped midway through, you can resume running at any point by using the config flag `eval.resume_from`, and it will finish running the remaining tasks on its own. For example, if a previous evaluation in the folder `results/2024-10-30/16-20-30_custom_gpt-4o-mini-2024-07-18` has not properly finished all of its environments, we can resume from it by running:

```
python eval.py \
  agent.type=custom \
  agent.max_image_history=0 \
  eval.num_workers=16 \
  client.client_name=openai \
  client.model_id=gpt-4o-mini-2024-07-18 \
  eval.resume_from=results/2024-10-30/16-20-30_custom_gpt-4o-mini-2024-07-18
```
