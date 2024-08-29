import pdb
import logging
import json
import multiprocessing
from queue import Empty
from iclbench.environments import make_env, get_tasks


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

        instructions = None
        if self.env_name == "babyai":
            instructions = obs["mission"]
        agent.prompt_builder.update_instruction_prompt(
            env.get_instruction_prompt(instructions=instructions)
        )

        episode_return = 0.0

        action = None
        for _ in range(self.max_steps_per_episode):
            action = agent.act(obs, prev_action=action)
            action = env.check_action_validity(action)
            obs, reward, done, _ = env.step(action)
            episode_return += reward

            if done:
                print("Episode done with reward:", episode_return)
                break

        return {
            "episode_return": episode_return,
            **agent.get_metrics(),
            **env.get_stats(),
        }

    def check_action_validity(self, env, action):
        valid_action = None
        for choice in action.choices:
            candidate_action = (
                choice.text
                if not hasattr(choice, "message")
                else choice.message.content
            )
            if candidate_action in env.language_action_space:
                valid_action = candidate_action
                break
        if not valid_action:
            valid_action = env.default_action
            logging.warn(
                f'Failed to generate a valid action. Output: "{action.choices}".\
                    Selecting default action "{valid_action}".'
            )
#             self.agent.failed_generation_counter += 1
        return valid_action

    def run(self):
        if self.num_workers > 1:
            return self._run_parallel()
        else:
            return self._run_sequential()

    def _run_sequential(self):
        results = []
        for task in self.tasks:
            for _ in range(self.num_episodes):
                results.append(self.run_episode(task))
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

    def save_results(self, results, filename):
        with open(filename, "w") as file:
            json.dump(results, file, indent=4)
