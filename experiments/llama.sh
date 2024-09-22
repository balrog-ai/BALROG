#./bin/bash

python eval.py \
  envs.names=babyai,babaisai,textworld,nle,minihack \
  agent.type=naive,cot \
  agent.max_image_history=0,1 \
  eval.num_workers=4 \
  client.client_name=vllm \
  client.model_id=meta-llama/Meta-Llama-3.1-70B-Instruct \
  -m