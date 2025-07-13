# 🚀 Evaluation
The following are various options you can use to evaluate your agents on BALROG.

## ⚡️ Evaluate using local vLLM server
We support running LLMs/VLMs out of the box using [vLLM](https://github.com/vllm-project/vllm). You can spin up a vLLM client and evaluate your agent on BALROG in the following way:

```
vllm serve meta-llama/Llama-3.2-1B-Instruct --port 8080

python eval.py \
  agent.type=naive \
  agent.max_image_history=0 \
  agent.max_text_history=16 \
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

Or you can modify the `SECRETS` file, adding your api keys. xAI or Nvidia API keys need to be added as OpenAI API keys, with client_name=xai or nvidia, together with their baseurl.

You can then run the evaluation with:

```
python eval.py \
  agent.type=naive \
  agent.max_image_history=0 \
  agent.max_text_history=16 \
  eval.num_workers=16 \
  client.client_name=openai \
  client.model_id=gpt-4o-mini-2024-07-18
```

## 🖼️ VLM mode

You can activate the VLM mode by increasing the `max_image_history` argument, for example

```
python eval.py \
  agent.type=naive \
  agent.max_text_history=16 \
  agent.max_image_history=1 \
  eval.num_workers=16 \
  client.client_name=openai \
  client.model_id=gpt-4o-mini-2024-07-18
```

## ▶️ Resume an evaluation
To resume an incomplete evaluation, use eval.resume_from. For example, if an evaluation in the folder results/2024-10-30/16-20-30_naive_gpt-4o-mini-2024-07-18 is unfinished, resume it with:

```
python eval.py \
  agent.type=naive \
  agent.max_image_history=0 \
  agent.max_text_history=16 \
  eval.num_workers=16 \
  client.client_name=openai \
  client.model_id=gpt-4o-mini-2024-07-18 \
  eval.resume_from=results/2024-10-30_16-20-30_naive_gpt-4o-mini-2024-07-18
```

## ⚙️ Configuring Eval

`eval.py` is configured using Hydra. We list some options below. For more details, refer to the [eval config](https://github.com/balrog-ai/BALROG/blob/main/balrog/config/config.yaml).

| Parameter                 | Description                                                                                       | Default Value                             |
|---------------------------|---------------------------------------------------------------------------------------------------|-------------------------------------------|
| **agent.type**            | Type of agent used                                 | `naive`                                      |
| **agent.remember_cot**    | Whether the agent should remember chain-of-thought (CoT) during episodes.                         | `True`                                    |
| **agent.max_text_history**     | Maximum number of dialogue history entries to retain.                                             | `16`                                      |
| **agent.max_image_history**| Maximum number of images included in the history. Use >= 1 if you want to use VLM mode           | `0`                                      |
| **eval.num_workers**      | Number of parallel environment workers for parallel evaluation.                                                        | `1`                                       |
| **eval.num_episodes**     | Number of episodes per environment for evaluation.                                                | `{nle: 5, minihack: 5, babyai: 25, ...}` |
| **eval.save_trajectories**| Whether to save agent trajectories during evaluation.                                             | `True`                                    |
| **eval.save_images**      | Whether to save images of the trajectory  during evaluation.                                      | `False`                                    |
| **client.client_name**    | Type of the client used, `vllm`, `openai`, `gemini`, `claude`                              | `openai`                                  |
| **client.model_id**       | Name of the model used.                                                                           | `gpt-4o`                                  |
| **client.base_url**       | Base URL of the model server for API requests with vllm.                                          | `http://localhost:8080/v1`                       |
| **client.is_chat_model**  | Indicates if the model follows a chat-based interface.                                            | `True`                                    |
| **client.generate_kwargs.temperature** | Temperature for model response randomness.                                           | `0.0`                                     |
| **client.alternate_roles** | If True the instruction prompt will be fused with first observation. Required by some LLMs.      | `False`                                     |
| **client.temperature**    | If set to null will default to the API default temperature. Use a float from 0.0 to 2.0. otherwise.  | `1.0`                                     |
| **envs.names**            | Dash-separated list of environments to evaluate, e.g., `nle-minihack`.                            | `babyai-babaisai-textworld-crafter-nle-minihack`|



## FAQ:

- Mac fork error:
  Mac systems might complain about fork when evaluating in multiprocessing mode (`eval.num_workers > 1`). To fix this export the following before running eval: `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`
- Alternate roles:
  Some LLMs/VLMs require alternating roles. You can fuse the instruction prompt with the first observation to comply with this with the following: `client.alternate_roles=True`
- Temperature:
  We recommend running models with temperature ranges around 0.7-1.0, or to use the default temperature of the model APIs. Too low temperatures can cause some of the more brittle models to endlessly repeat actions or create incoherent outputs.
