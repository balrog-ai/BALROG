import os
from fmrl.environments.nle import NLELanguageWrapper
from fmrl.prompt_builder import ChatPromptBuilder
from nle.nethack import ACTIONS
from nle_language_wrapper import NLELanguageWrapper
import wandb
import gym
from functools import partial
from omegaconf import OmegaConf
import logging
from fmrl.agents import NaiveAgent
from queue import Empty
import multiprocessing
from queue import Empty
import numpy as np

def make_env(obs_style):
    env = gym.make('NetHackChallenge-v0')
    return NLELanguageWrapper(env, obs_style)
    
def make_prompt_builder(strategy):
    ACTION_NAMES = [
        action_strs[0]
        for action, action_strs in NLELanguageWrapper.all_nle_action_map.items()
        if action in ACTIONS
    ]
    ACTIONS_LIST_STR = ",\n".join(ACTION_NAMES)
    INSTRUCTION_PROMPT = f"""
    You are an agent playing NetHack. In a moment I will present you an observation. Only output an action from the following list:
    {ACTIONS_LIST_STR}.

    For example, a valid output would simply be "{ACTION_NAMES[0]}" or "{ACTION_NAMES[1]}".
    You can only output one action at a time. The goal is to maximize the reward.
    Don't just output the example actions above, output the action that you think will maximize the reward.
    """.strip()
    
    if strategy == "simple":
        raise NotImplementedError
    elif strategy == "chat":
        return ChatPromptBuilder(INSTRUCTION_PROMPT)
    else:
        raise ValueError(f"Unknown prompt_builder_strategy: {strategy}, choices are [\"simple\", \"chat\"]")

def agent_worker(task_queue, results_queue):
    while True:
        try:
            agent_args = task_queue.get()
            agent = NaiveAgent(**agent_args)
            agent.run()
            results_queue.put(agent.get_metrics())
        except Empty:
            break
        
# want some of that jax innit
def tree_map(func, *trees):
    if not trees:
        raise ValueError("At least one tree must be provided")
    
    first_tree = trees[0]
    
    if isinstance(first_tree, dict):
        return {k: tree_map(func, *(t[k] for t in trees)) for k in first_tree}
    elif isinstance(first_tree, (list, tuple)):
        return type(first_tree)(tree_map(func, *(t[i] for t in trees)) for i in range(len(first_tree)))
    elif isinstance(first_tree, np.ndarray):
        return func(*trees)
    else:
        return func(*trees)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    
    config = OmegaConf.to_container(OmegaConf.load("eval_config.yaml"))
    if config["savedir"] is None:
        config["savedir"] = os.path.join(".", "outputs", config["model_id"])
        
    default_payload = default_payload={
        "model": config["model_id"],
        **config["generate_kwargs"],
    }
    
    worker_args = {
        "make_env": partial(make_env, obs_style=config["obs_style"]),
        "url": "http://localhost:8000/v1/chat/completions",
        "make_prompt_builder": partial(make_prompt_builder, strategy=config["prompt_builder_strategy"]),
        "default_payload": default_payload,
    }
    
    env = worker_args["make_env"]()
    
    NUM_RUNS = 2
    NUM_WORKERS = 1
    
    if NUM_WORKERS > 1:
        task_queue = multiprocessing.Queue()
        results_queue = multiprocessing.Queue()
        
        for _ in range(NUM_RUNS):
            task_queue.put(worker_args)
        
        processes = []
        for _ in range(NUM_WORKERS):
            p = multiprocessing.Process(target=agent_worker, args=(task_queue, results_queue))
            processes.append(p)
            p.start()
        
        results = []
        while len(results) < NUM_RUNS:
            result = results_queue.get()
            results.append(result)
            
        for p in processes:
            p.join()
            
        # something, something collect stats here
        results_tree = tree_map(lambda *x: np.stack(x), *results)
        print(results_tree["episode_return"].mean())
        # for result in results:
    else:
        results = []
        for _ in range(NUM_RUNS):
            agent = NaiveAgent(**worker_args)
            agent.run()
            results.append(agent.get_metrics())
            
        for result in results:
            print("====")
            print(result)
        # print(results)
        