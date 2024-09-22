#./bin/bash

python eval.py \
  envs.names=babyai,babaisai,textworld,nle,minihack \
  agent.type=naive,cot \
  agent.max_image_history=0,1 \
  eval.num_workers=4 \
  client.client_name=claude \
  client.model_id=claude-3-5-sonnet-20240620 \
  -m