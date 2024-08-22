# In-Context-Learning benchmark for LLM agents

![Alt text](assets/figures/balrog.jpg)

### Collaboration guide:
When implementing majour features:
1. Branch out from develop
2. Implement feature
3. Once ready, open a PR from your feature branch to develop
4. Solve conflicts and ask for review
5. Once reviewers approve, merge (replying with LGTM is fine fow now)

Let's try to keep it fast paces but also organized.

# North Star
High-quality, easy to enter benchmark for LLM agents, testing their in-context-learning capabilities on a variety of interactive environments.


# Structure:

```
ICL-bench
├── README.md               # Documentation of the repository
├── config/                 # Config folder
│   ├── eval.yaml           # Base evaluation config file
├── iclbench/               # Main code of the 
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
# Load configuration
config = OmegaConf.load("config/eval.yaml")

# Instantiate LLM client
client = OpenAI(api_key="EMPTY", base_url=config.base_url)

# Instantiate environment
env = make_envs(**config.env_kwargs)

agent = create_agent_class(client, config)

# Instantiate evaluator and run the evaluation
evaluator = Evaluator(env, agent, config)
results = evaluator.run()

# Save results
evaluator.save_results(results, config.get("savedir", "eval.json"))
```

Ideally we should in the future also support interaction with the benchmark purely from command line with an evaluation harness similar to SWEbench or llm-eval-harness

# Environments:
1. NetHack
2. Craftax
3. MiniHack
4. BabyAI

# Installation
```
conda create --y --name iclbench python=3.10
conda activate iclbench
pip install nle==0.9.0
cd external/nle-language-wrapper
pip install -e .
```
