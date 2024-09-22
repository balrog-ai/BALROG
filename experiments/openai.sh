#./bin/bash

python eval.py \
  envs.names=babyai,babaisai,textworld,nle,minihack \
  agent.type=naive,cot \
  agent.max_image_history=0,1 \
  eval.num_workers=4 \
  client.client_name=openai \
  client.model_id=gpt-4o-mini-2024-07-18 \
  -m