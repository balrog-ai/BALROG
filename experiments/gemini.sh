#./bin/bash

python eval.py \
  envs.names=babyai,babaisai,textworld,nle,minihack \
  agent.type=naive,cot \
  agent.max_image_history=0,1 \
  eval.num_workers=4 \
  client.client_name=gemini \
  client.model_id=gemini-1.5-flash,gemini-1.5-pro \
  -m