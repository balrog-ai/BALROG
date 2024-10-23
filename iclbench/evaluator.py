import copy
import json
import csv
import logging
import multiprocessing
import os
import random
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from iclbench.agents.icl import ICLAgent
from iclbench.dataset import InContextDataset
from iclbench.environments import make_env


class Evaluator:
    def __init__(self, env_name, config, original_cwd=""):
        self.env_name = env_name.strip()  # Ensure no leading/trailing whitespace
        self.config = config
        self.tasks = config.tasks[f"{self.env_name}_tasks"]

        self.num_episodes = config.eval.num_episodes[self.env_name]
        self.num_workers = config.eval.num_workers
        self.max_steps_per_episode = config.eval.max_steps_per_episode

        self.dataset = InContextDataset(self.config, self.env_name, original_cwd=original_cwd)

    def load_in_context_learning_episode(self, i, task, agent, episode_log):
        demo_config = copy.deepcopy(self.config)
        demo_task = self.dataset.demo_task(task)
        demo_path = self.dataset.demo_path(i, demo_task, demo_config)
        self.dataset.override_incontext_config(demo_config, demo_path)
        env = make_env(self.env_name, demo_task, demo_config)
        recorded_actions = self.dataset.load_incontext_actions(demo_path)

        seed = demo_config.envs.env_kwargs.seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        env.seed(seed=seed)

        obs = env.reset()
        for action in recorded_actions:
            text_action = env.get_text_action(action)

            agent.update_icl_observation(obs)
            agent.update_icl_action(text_action)

            if self.config.eval.save_trajectories:
                episode_log["trajectory"].append((obs["text"]["long_term_context"], text_action))
            episode_log["action_frequency"][text_action] += 1

            obs, reward, done, info = env.step(text_action)

            if done:
                break

        if not done:
            print("warning: icl trajectory ended without done")

        agent.wrap_episode()

    def run_episode(self, task, agent, process_num=None, position=0, episode_idx=0):
        env = make_env(self.env_name, task, self.config)
        agent.reset()

        seed = self.config.envs.env_kwargs.seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        env.seed(seed=seed)

        obs = env.reset()
        episode_log = {
            "task": task,
            "action_frequency": defaultdict(int),
            "input_tokens": 0,
            "output_tokens": 0,
        }

        instructions = None
        if self.env_name == "babyai":
            instructions = obs["mission"]
        agent.prompt_builder.update_instruction_prompt(env.get_instruction_prompt(instructions=instructions))

        episode_return = 0.0

        max_steps_per_episode = env.max_steps if self.max_steps_per_episode is None else self.max_steps_per_episode

        # Create a unique CSV filename for this episode
        csv_filename = os.path.join(self.env_name, task, f"{task}_run_{episode_idx:02d}.csv")
        Path(csv_filename).parent.mkdir(exist_ok=True, parents=True)

        # Open the CSV file and write the header
        with open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Step", "Observation", "Action", "Reasoning", "Reward", "Done"])

            # If the agent is an ICLAgent, load the in-context learning episode
            if isinstance(agent, ICLAgent):
                for icl_episode in range(self.config.eval.icl_episodes):
                    self.load_in_context_learning_episode(icl_episode, task, agent, episode_log)

                if self.config.agent.cache_icl and self.config.client.client_name == "gemini":
                    agent.cache_icl()

            pbar_desc = f"Task: {task}, Proc: {process_num}"
            pbar = tqdm(
                total=max_steps_per_episode,
                desc=pbar_desc,
                position=position,
                leave=True,  # Keep the progress bar after completion
                dynamic_ncols=True,
            )

            action = None
            for step in range(max_steps_per_episode):
                response = agent.act(obs, prev_action=action)
                action = env.check_action_validity(response.completion)
                reasoning = response.reasoning if hasattr(response, "reasoning") else ""

                episode_log["action_frequency"][action] += 1
                episode_log["input_tokens"] += response.input_tokens
                episode_log["output_tokens"] += response.output_tokens

                obs, reward, done, info = env.step(action)
                episode_return += reward

                # Write the step data to the CSV file
                csv_writer.writerow([step, obs["text"]["long_term_context"], action, reasoning, reward, done])

                pbar.update(1)

                if done:
                    logging.info(f"Episode done with reward: {episode_return}")
                    episode_log["done"] = True
                    if pbar.n < pbar.total:
                        pbar.update(pbar.total - pbar.n)
                    pbar.set_postfix_str("DONE")
                    break

            if pbar.n < pbar.total:
                pbar.update(pbar.total - pbar.n)
            if "done" not in episode_log:
                pbar.set_postfix_str("DONE")
            pbar.close()

            episode_log["episode_return"] = episode_return
            episode_log["num_steps"] = step + 1
            episode_log["failed_candidates"] = env.failed_candidates
            episode_log.update(env.get_stats())

            episode_log["process_num"] = process_num

            # Write a separator row
            csv_writer.writerow([])
            csv_writer.writerow(["Episode Summary"])
            for key, value in episode_log.items():
                if isinstance(value, dict):
                    value_str = json.dumps(value)
                else:
                    value_str = str(value)
                csv_writer.writerow([key, value_str])

        return episode_log

    def run(self, agent_factory):
        if self.num_workers > 1:
            results = self._run_parallel(agent_factory)
        else:
            results = self._run_sequential(agent_factory)

        summary = self._save_results(results, self.env_name)
        return summary

    def _run_sequential(self, agent_factory):
        results = defaultdict(list)
        total_episodes = len(self.tasks) * self.num_episodes
        with tqdm(total=total_episodes, desc="Evaluating Episodes") as pbar:
            for task in self.tasks:
                for episode_idx in range(self.num_episodes):
                    agent = agent_factory.create_agent()
                    episode_log = self.run_episode(task, agent, episode_idx=episode_idx)
                    results[task].append(episode_log)
                    pbar.update(1)
        return results

    def _run_parallel(self, agent_factory):
        task_queue = multiprocessing.Queue()
        results_queue = multiprocessing.Queue()

        # Create a multiprocessing context with spawn
        ctx = multiprocessing.get_context("fork")

        # Create a list of all tasks to be executed
        all_tasks = [(task, episode_idx) for task in self.tasks for episode_idx in range(self.num_episodes)]

        # Initially fill the task queue with tasks up to the number of workers
        for item in all_tasks[: self.num_workers]:
            task_queue.put(item)

        # Assign unique positions for progress bars
        positions = list(range(self.num_workers))

        processes = []
        for idx in range(self.num_workers):
            position = positions[idx]
            p = ctx.Process(
                target=self._worker,
                args=(task_queue, results_queue, agent_factory, position),
            )
            processes.append(p)
            p.start()

        results = defaultdict(list)
        tasks_completed = 0
        tasks_queued = self.num_workers

        total_tasks = len(all_tasks)

        with tqdm(total=total_tasks, desc="Evaluating Episodes") as pbar:
            while tasks_completed < total_tasks:
                result = results_queue.get()
                if "error" in result:
                    logging.error(
                        f"Error in task {result['task']} processed by {result['process_num']}: {result['error']}"
                    )
                    logging.error(f"Traceback:\n{result['traceback']}")
                else:
                    results[result["task"]].append(result)
                tasks_completed += 1

                # Update progress bar
                pbar.update(1)
                pbar.set_description(f"Last task: {result['task']}, Process: {result.get('process_num', 'N/A')}")

                # Queue another task if there are any left
                if tasks_queued < len(all_tasks):
                    task_queue.put(all_tasks[tasks_queued])
                    tasks_queued += 1

        # Signal workers to stop
        for _ in range(self.num_workers):
            task_queue.put(None)

        for p in processes:
            p.join()

        return results

    def _worker(self, task_queue, results_queue, agent_factory, position):
        agent = agent_factory.create_agent()
        process_num = multiprocessing.current_process().name
        while True:
            item = task_queue.get()
            if item is None:
                break
            try:
                task, episode_idx = item
                result = self.run_episode(
                    task, agent, process_num=process_num, position=position + 1, episode_idx=episode_idx
                )
                result["process_num"] = process_num  # Include process number in result
                results_queue.put(result)
            except Exception as e:
                tb = traceback.format_exc()
                logging.error(f"Error in worker processing task {task}: {e}\n{tb}")
                results_queue.put({"task": task, "error": str(e), "traceback": tb, "process_num": process_num})

    def _save_results(self, results, env_name):
        total_progression = 0.0
        total_count = 0

        env_summary = {}

        for task, runs in results.items():
            task_progression = sum(run.get("progression", 0.0) for run in runs)
            task_count = len(runs)
            avg_task_progression = task_progression / task_count if task_count else 0

            env_summary[task] = {
                "progression_percentage": 100 * avg_task_progression,
                "episodes_played": task_count,
            }

            total_progression += task_progression
            total_count += task_count

        overall_avg_progression = total_progression / total_count if total_count else 0

        data = {
            "progression_percentage": 100 * overall_avg_progression,
            "episodes_played": total_count,
            "tasks": env_summary,
            "input_tokens": sum(run["input_tokens"] for task_runs in results.values() for run in task_runs),
            "output_tokens": sum(run["output_tokens"] for task_runs in results.values() for run in task_runs),
        }

        filename = os.path.join(env_name, f"{env_name}_summary.json")
        Path(filename).parent.mkdir(exist_ok=True, parents=True)
        with open(filename, "w") as file:
            json.dump(data, file, indent=4)
        logging.info(f"Results saved for {env_name} in {filename}")

        return data
