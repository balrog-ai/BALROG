#./bin/bash

python eval.py \
  envs.names=nle \
  agent.type=icl \
  agent.max_image_history=0,1 \
  envs.env_kwargs.nle_kwargs.character=bar,mon,val,cav,kni \
  eval.icl_episodes=1 \
  eval.num_workers=1 \
  client.client_name=gemini \
  client.model_id=gemini-1.5-flash,gemini-1.5-pro \
  -m

python eval.py \
  envs.names=textworld,minihack,crafter,babyai \
  agent.type=icl \
  agent.max_image_history=0,1 \
  eval.icl_episodes=1 \
  eval.num_workers=1 \
  client.client_name=gemini \
  client.model_id=gemini-1.5-flash,gemini-1.5-pro \
  -m

python eval.py \
  envs.names=babaisai \
  agent.type=icl \
  agent.max_image_history=0,1 \
  eval.icl_episodes=3 \
  eval.num_workers=1 \
  client.client_name=gemini \
  client.model_id=gemini-1.5-flash,gemini-1.5-pro \
  -m
