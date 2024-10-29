# BALROG: Benchmarking Agenting LLM/VLM Reasoning On Games

![Alt text](assets/figures/balrog.jpg)

### Collaboration guide:
When implementing majour features:
1. Branch out from develop
2. Open PR from your feature branch to develop, describing what you are working on
3. Implement feature on feature branch
4. Once feature is ready, solve conflicts and ask for review
5. Merge when approved (replying with LGTM is fine fow now)

Let's try to keep it fast paces but also organized.

# North Star
High-quality, easy to enter benchmark for LLM agents, testing their in-context-learning capabilities on a variety of interactive environments.


# Structure:

```
Balrog
├── README.md               # Documentation of the repository
├── config/                 # Config folder
│   ├── eval.yaml           # Base evaluation config file
├── balrog/               # Main code of the 
│   ├── agents/             # Agent implementations with LangChain (naive agent for now)
│   ├── environments/       # Environments folder with a unified env loader
│   └── prompt_builder/     # History, Zero-shot, and VLM prompt builders
│   ├── evaluator.py        # File with main evaluator class
├── external/               # External submodules
│   ├── nle-language-wrap   # Modified language wrapper (to be moved inside nle in environments)
└── eval.py                 # Entry point of the benchmark
```

The idea is to that people will interact through the eval.py, whose structure should more or less be:

```
evaluator_manager = EvaluatorManager(config, original_cwd=original_cwd)
agent_factory = AgentFactory(config)

# Run experiments
evaluator_manager.run(agent_factory)

overall_summary = collect_and_summarize_results(evaluator_manager.output_dir, config)
print_summary_table(overall_summary)
```

Ideally we should in the future also support interaction with the benchmark purely from command line with an evaluation harness similar to SWEbench or llm-eval-harness

# Environments:
1. NetHack
2. Craftax -> TODO
3. MiniHack -> TODO
4. BabyAI -> TODO

# Installation
```
conda create --y --name balrog python=3.10
conda activate balrog
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
pip install wandb
pip install pytest
```

### pre-commit installation and setup 
```
pip install black isort flake8 pre-commit
pre-commit install
pre-commit run --all-files
``` 

# Create a SECRETS file

```txt
OPENAI_API_KEY=<KEY>
GEMINI_API_KEY=<KEY>
ANTHROPIC_API_KEY=<KEY>
DEFAULT_ORG=
```

# Run

Spin up a vllm server (if on another GPU, consider tunneling) :
```
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct
```
The run eval.py. If you are on the same machine as the vllm server, simply run:
```
python eval.py
```

If you are on a different machine, and are doing tunneling:
```
python eval.py base_url=/your/vllm/server/baseurl
```

### In Context Learning
We use expert demonstrations for ICL
Download and unzip them

    curl -L -o demos.zip 'https://drive.google.com/uc?export=download&id=11vYFclIY4RoJ6Ha7I5rWhuA5lQnhDtPL'
    unzip demos.zip
