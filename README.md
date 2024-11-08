# BALROG: Benchmarking Agenting LLM/VLM Reasoning On Games

<p align="center">
  <a href="https://balrogai.com">
    <img src="assets/figures/balrog_1.png" width="50%" alt="BALROG Agent" />
  </a>
</p>

---
<p align="center">
Code for our paper BALROG: Benchmarking Agentic LLM/VLM Reasoning On Games


# Insallation
We advise using conda for the installation
```
conda create -n balrog python=3.10 -y
conda activate balrog

git clone https://github.com/balrog-ai/BALROG.git
cd BALROG
pip install -e .
balrog-post-install
pytest tests/test_evaluation.py
```

# Evaluate your agent
For a simple tutorial on how to evaluate and create custom agents, check out the [tutorial](https://github.com/balrog-ai/BALROG/blob/main/assets/evaluation.md)