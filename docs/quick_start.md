# Quickstart


Experiments are run using the eval.py script. Simply run:
```bash
python eval.py envs="babyai,nle" base_url=<your_vllm_server_base_url>
```
Where evaluation environments are specified in a comma separated list. Experiment results are saved to the `./results` directory.

By default, eval.py assumes the LLM follows the OpenAI interface. This makes it compatible with your self-hosted VLLM server.

Spin up a vllm server (if on another GPU, consider tunneling) :
```bash
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct
```

## In Context Learning

We use expert demonstrations for ICL
Download and unzip them

    curl -L -o demos.zip 'https://drive.google.com/uc?export=download&id=11vYFclIY4RoJ6Ha7I5rWhuA5lQnhDtPL'
    unzip demos.zip