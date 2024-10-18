#./bin/bash

python eval.py \
  envs.names=nle \
  agent.type=few_shot \
  agent.max_image_history=0,1 \
  envs.env_kwargs.nle_kwargs.character=bar,mon,val,cav,kni \
  eval.icl_episodes=1 \
  eval.num_workers=1 \
  client.client_name=openai \
  client.model_id=gpt-4o-mini-2024-07-18 \
  -m

python eval.py \
  envs.names=textworld,minihack,crafter,babyai \
  agent.type=few_shot \
  agent.max_image_history=0,1 \
  eval.icl_episodes=1 \
  eval.num_workers=1 \
  client.client_name=openai \
  client.model_id=gpt-4o-mini-2024-07-18 \
  -m

python eval.py \
  envs.names=babaisai \
  agent.type=few_shot \
  agent.max_image_history=0,1 \
  eval.icl_episodes=3 \
  eval.num_workers=1 \
  client.client_name=openai \
  client.model_id=gpt-4o-mini-2024-07-18 \
  -m
