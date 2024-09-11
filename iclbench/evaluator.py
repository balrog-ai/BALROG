import os
import time
import logging
import json
import multiprocessing
from queue import Empty
from iclbench.environments import make_env, get_tasks
from collections import defaultdict


class Evaluator:
    def __init__(self, env_name, agent_factory, config):
        self.env_name = env_name
        self.config = config
        self.tasks = get_tasks(env_name)

        self.agent_factory = agent_factory

        self.num_episodes = config.num_episodes
        self.num_workers = config.num_workers
        self.max_steps_per_episode = config.max_steps_per_episode

    def run_episode(self, task):
        env = make_env(self.env_name, task, self.config)
        agent = self.agent_factory()

        obs = env.reset()
        episode_log = {
            "task": task,
            "trajectory": [],
            "action_frequency": defaultdict(int),
        }

        instructions = None
        if self.env_name == "babyai":
            instructions = obs["mission"]
        agent.prompt_builder.update_instruction_prompt(
            env.get_instruction_prompt(instructions=instructions)
        )

        episode_return = 0.0

        action = None
        for step in range(self.max_steps_per_episode):

            action = agent.act(obs, prev_action=action)
            action = env.check_action_validity(action)
            if self.config.save_trajectories:
                episode_log["trajectory"].append((obs["text"][0], action))
            episode_log["action_frequency"][action] += 1

            obs, reward, done, info = env.step(action)
            episode_return += reward

            if done:
                logging.info(f"Episode done with reward: {episode_return}")
                episode_log["done"] = True
                break

        episode_log["episode_return"] = episode_return
        episode_log["num_steps"] = step + 1
        episode_log["failed_candidates"] = env.failed_candidates
        episode_log.update(env.get_stats())

        return episode_log

    def run(self):
        if self.num_workers > 1:
            results = self._run_parallel()
        else:
            results = self._run_sequential()

        summary = self._save_results(results, self.env_name)
        return summary

    def _run_sequential(self):
        results = defaultdict(list)
        for task in self.tasks:
            for _ in range(self.num_episodes):
                episode_log = self.run_episode(task)
                results[task].append(episode_log)
        return results

    def _run_parallel(self):
        task_queue = multiprocessing.Queue()
        results_queue = multiprocessing.Queue()

        for _ in range(self.num_episodes):
            task_queue.put(None)  # We can pass any required args here

        processes = []
        env_tasks = [task * self.num_episodes for task in self.tasks]

        for idx in range(self.num_workers):
            p = multiprocessing.Process(
                target=self._worker, args=(env_tasks[idx], task_queue, results_queue)
            )
            processes.append(p)
            p.start()

        results = []
        while len(results) < self.num_episodes:
            results.append(results_queue.get())

        for p in processes:
            p.join()

        return results

    def _worker(self, env_task, task_queue, results_queue):
        while True:
            try:
                _ = task_queue.get(timeout=1)
                result = self.run_episode(env_task)
                results_queue.put(result)
            except Empty:
                break

    def _save_results(self, results, env_name):

        progression = 0.0
        count = 0

        env_summary = defaultdict(list)

        for task, result in results.items():
            task_folder = os.path.join(env_name, task)
            os.makedirs(task_folder, exist_ok=True)
            task_progression = 0.0
            task_count = 0
            for idx, run in enumerate(result):
                progression += run["progression"]
                count += 1
                task_progression += run["progression"]
                task_count += 1
                filename = os.path.join(task_folder, f"run_{idx:02d}.json")
                with open(filename, "w") as file:
                    json.dump(run, file, indent=4)
            env_summary[task] = (task_progression / task_count, task_count)

        data = {
            "progression_percentage": 100 * progression / count,
            "episodes_played": count,
            "tasks": {
                task: {"progression_percentage": 100 * prog, "episodes_played": cnt}
                for task, (prog, cnt) in env_summary.items()
            },
        }

        filename = os.path.join(env_name, "summary.json")
        with open(filename, "w") as file:
            json.dump(data, file, indent=4)
        logging.info(f"Results saved for {env_name} in {filename}")

        return data


def summarize_env_progressions(results_summaries: defaultdict) -> float:
    average_progression = 0.0
    for _, results in results_summaries.items():
        average_progression += int(results["progression_percentage"])
    average_progression /= len(results_summaries)

    results_summaries["Final score"] = average_progression

    with open("summary.json", "w") as f:
        json.dump(results_summaries, f)
    logging.info(f"Results saved in summary.json")

    return average_progression
