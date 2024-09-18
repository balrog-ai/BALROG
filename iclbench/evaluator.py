import os
import logging
import json
import multiprocessing
from iclbench.environments import make_env, get_tasks
from collections import defaultdict
from tqdm import tqdm


class Evaluator:
    def __init__(self, env_name, config):
        self.env_name = env_name.strip()  # Ensure no leading/trailing whitespace
        self.config = config
        self.tasks = get_tasks(self.env_name)

        self.num_episodes = config.num_episodes
        self.num_workers = config.num_workers
        self.max_steps_per_episode = config.max_steps_per_episode

    def run_episode(self, task, agent, process_num=None, position=0):
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

        pbar_desc = f"Task: {task}, Proc: {process_num}"
        pbar = tqdm(
            total=self.max_steps_per_episode,
            desc=pbar_desc,
            position=position,
            leave=True,  # Keep the progress bar after completion
            dynamic_ncols=True,
        )

        action = None
        for step in range(self.max_steps_per_episode):
            action = agent.act(obs, prev_action=action)
            action = env.check_action_validity(action)
            if self.config.save_trajectories:
                episode_log["trajectory"].append(
                    (obs["text"]["long_term_context"], action)
                )
            episode_log["action_frequency"][action] += 1

            obs, reward, done, info = env.step(action)

            episode_return += reward

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
                for _ in range(self.num_episodes):
                    agent = agent_factory.create_agent()
                    episode_log = self.run_episode(task, agent)
                    results[task].append(episode_log)
                    pbar.update(1)
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
                results[result["task"]].append(result)
                tasks_completed += 1

                # Update progress bar
                pbar.update(1)
                pbar.set_description(
                    f"Last task: {result['task']}, Process: {result.get('process_num', 'N/A')}"
                )

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
            task = task_queue.get()
            if task is None:
                break
            try:
                result = self.run_episode(
                    task, agent, process_num=process_num, position=position + 1
                )
                result["process_num"] = process_num  # Include process number in result
                results_queue.put(result)
            except Exception as e:
                logging.error(f"Error in worker: {e}")
                results_queue.put(
                    {"task": task, "error": str(e), "process_num": process_num}
                )

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
                progression += run.get("progression", 0.0)
                count += 1
                task_progression += run.get("progression", 0.0)
                task_count += 1
                filename = os.path.join(task_folder, f"{task}_run_{idx:02d}.json")
                with open(filename, "w") as file:
                    json.dump(run, file, indent=4)
            env_summary[task] = (
                task_progression / task_count if task_count else 0,
                task_count,
            )

        data = {
            "progression_percentage": 100 * progression / count if count else 0,
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
