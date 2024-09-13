import os
import logging
import json
import multiprocessing
from iclbench.environments import make_env, get_tasks
from collections import defaultdict


class Evaluator:
    def __init__(self, env_name, config):
        self.env_name = env_name
        self.config = config
        self.tasks = get_tasks(env_name)

        self.num_episodes = config.num_episodes
        self.num_workers = config.num_workers
        self.max_steps_per_episode = config.max_steps_per_episode

    def run_episode(self, task, agent):
        env = make_env(self.env_name, task, self.config)
        agent.reset()

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

    def run(self, agent_factory):
        if self.num_workers > 1:
            results = self._run_parallel(agent_factory)
        else:
            results = self._run_sequential(agent_factory)

        summary = self._save_results(results, self.env_name)
        return summary

    def _run_sequential(self, agent_factory):
        results = defaultdict(list)
        for task in self.tasks:
            for _ in range(self.num_episodes):
                agent = agent_factory()
                episode_log = self.run_episode(task, agent)
                results[task].append(episode_log)
        return results

    def _run_parallel(self, agent_factory):
        task_queue = multiprocessing.Queue()
        results_queue = multiprocessing.Queue()

        # Create a multiprocessing context with spawn
        ctx = multiprocessing.get_context("spawn")

        # Create a list of all tasks to be executed
        all_tasks = [task for task in self.tasks for _ in range(self.num_episodes)]

        # Initially fill the task queue with tasks up to the number of workers
        for task in all_tasks[: self.num_workers]:
            task_queue.put(task)

        processes = []
        for _ in range(self.num_workers):
            p = ctx.Process(
                target=self._worker, args=(task_queue, results_queue, agent_factory)
            )
            processes.append(p)
            p.start()

        results = defaultdict(list)
        tasks_completed = 0
        tasks_queued = self.num_workers

        while tasks_completed < len(all_tasks):
            result = results_queue.get()
            results[result["task"]].append(result)
            tasks_completed += 1

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

    def _worker(self, task_queue, results_queue, agent_factory):
        agent = agent_factory.create_agent()
        while True:
            task = task_queue.get()
            if task is None:
                break
            try:
                result = self.run_episode(task, agent)
                results_queue.put(result)
            except Exception as e:
                logging.error(f"Error in worker: {e}")
                results_queue.put({"task": task, "error": str(e)})

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
                filename = os.path.join(task_folder, f"{task}_run_{idx:02d}.json")
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

        filename = os.path.join(env_name, f"{env_name}_summary.json")
        with open(filename, "w") as file:
            json.dump(data, file, indent=4)
        logging.info(f"Results saved for {env_name} in {filename}")

        return data
