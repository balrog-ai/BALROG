#./bin/bash

agent=naive
num_episodes=5
num_workers=4

common_args=(
  "agent=$agent"
  "num_episodes=$num_episodes"
  "num_workers=$num_workers"
)

babaisai_args=(
  "agent=$agent"
  "num_episodes=2"
  "num_workers=$num_workers"
)

client_args=(
    "client.client_name=claude"
    "client.model_id=claude-3-5-sonnet-20240620"
)

# python -m eval env_names=nle "${common_args[@]}" "${client_args[@]}" -m
# python -m eval env_names=minihack "${common_args[@]}" "${client_args[@]}" -m
# python -m eval env_names=craftax "${common_args[@]}" "${client_args[@]}" -m
# python -m eval env_names=babyai "${common_args[@]}" "${client_args[@]}" -m
# python -m eval env_names=textworld "${common_args[@]}" "${client_args[@]}" -m
python -m eval env_names=babaisai "${babaisai_args[@]}" "${client_args[@]}" -m
