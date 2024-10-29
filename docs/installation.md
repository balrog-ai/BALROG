# Installation

To get setup, please install the following dependencies:

```
conda create --y --name iclbench python=3.10
conda activate iclbench
pip install -e external/Grounding_LLMs_with_online_RL/babyai-text
pip install -e external/Grounding_LLMs_with_online_RL/babyai-text/babyai
pip install -e external/Grounding_LLMs_with_online_RL/babyai-text/gym-minigrid
pip install -e external/nle-language-wrapper
pip install -e external/nle
pip install textworld
pip install craftax
pip install git+https://github.com/facebookresearch/minihack
pip install git+https://github.com/nacloos/baba-is-ai.git

pip install openai
pip install anthropic
pip install google-generativeai
pip install replicate
pip install wandb
pip install pytest
```