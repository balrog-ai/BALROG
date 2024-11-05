# Submission Tutorial

Davide Paglieri • October 30, 2024

In this tutorial we show how to create custom agents and evaluate them using BALROG.


## 🤖 Creating custom agents

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

    def act(self, obs, info, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs, info)

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

Experiment with this example or explore additional templates from repositories like [LangGraph](https://github.com/langchain-ai/langgraph). Feel free to contribute by opening a PR with your own reasoning templates.

## ⚡️ Evaluate using vLLM locally
We support running LLMs/VLMs out of the box using [vLLM](https://github.com/vllm-project/vllm). You can spin up a vLLM client and evaluate your agent on BALROG in the following way:

```
vllm serve meta-llama/Llama-3.2-1B-Instruct --port 8080

python eval.py \
  agent.type=custom \
  agent.max_image_history=0 \
  agent.max_history=16 \
  eval.num_workers=16 \
  client.client_name=vllm \
  client.model_id=meta-llama/Llama-3.2-1B-Instruct \
  client.base_url=http://0.0.0.0:8080/v1
```

Check out [vLLM](https://github.com/vllm-project/vllm) for more options on how to serve your models fast and efficiently.


## 🛜 Evaluate using API
We support how of the box clients for OpenAI, Anthropic and Google Gemini APIs. If you want to evaluate an agent using one of these APIs, you first have to set up your API key in one of two ways:

You can either directly export it:

```
export OPENAI_API_KEY=<KEY>
export ANTHROPIC_API_KEY=<KEY>
export GEMINI_API_KEY=<KEY>
```

Or you can modify the `SECRETS` file, adding your api keys.

You can then run the evaluation with:

```
python eval.py \
  agent.type=custom \
  agent.max_image_history=0 \
  agent.max_history=16 \
  eval.num_workers=16 \
  client.client_name=openai \
  client.model_id=gpt-4o-mini-2024-07-18
```

## VLM mode and more configurations

You can activate the VLM mode by increasing the `max_image_history` argument, for example

```
python eval.py \
  agent.type=custom \
  agent.max_history=16 \
  agent.max_image_history=1 \
  eval.num_workers=16 \
  client.client_name=openai \
  client.model_id=gpt-4o-mini-2024-07-18
```

Have a look at more options on the config file `config/config.yaml` for additional arguments you can change.


## ⚙️ Resume an evaluation
To resume an incomplete evaluation, use eval.resume_from. For example, if an evaluation in the folder results/2024-10-30/16-20-30_custom_gpt-4o-mini-2024-07-18 is unfinished, resume it with:

```
python eval.py \
  agent.type=custom \
  agent.max_image_history=0 \
  agent.max_history=16 \
  eval.num_workers=16 \
  client.client_name=openai \
  client.model_id=gpt-4o-mini-2024-07-18 \
  eval.resume_from=results/2024-10-30_16-20-30_custom_gpt-4o-mini-2024-07-18
```
