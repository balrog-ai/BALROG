import json
import logging
import os
import math
from collections import defaultdict
from omegaconf import OmegaConf
from pathlib import Path

import google.generativeai as genai
import openai
import wandb


def collect_and_summarize_results(output_dir, config):
    results_summaries = defaultdict(list)

    # Collect per-episode results
    for env_name in os.listdir(output_dir):
        env_dir = os.path.join(output_dir, env_name)
        if not os.path.isdir(env_dir):
            continue

        # Recursively traverse directories under env_dir
        for root, dirs, files in os.walk(env_dir):
            for filename in files:
                if filename.endswith(".json") and not filename.endswith("_summary.json"):
                    json_filepath = os.path.join(root, filename)
                    with open(json_filepath, "r") as f:
                        episode_log = json.load(f)
                        results_summaries[env_name].append(episode_log)

    # Summarize results per environment and overall
    total_progressions = []
    total_envs = 0
    overall_total_input_tokens = 0
    overall_total_output_tokens = 0
    overall_env_summaries = {}

    for env_name, episodes in results_summaries.items():
        env_episode_progress = []
        env_total_steps = 0
        env_total_input_tokens = 0
        env_total_output_tokens = 0
        env_total_episodes = len(episodes)
        env_tasks = defaultdict(list)

        for episode_log in episodes:
            task_name = episode_log.get("task")
            env_tasks[task_name].append(episode_log)
            episode_progress = episode_log.get("progression", 0.0)
            env_episode_progress.append(episode_progress)
            env_total_steps += episode_log.get("num_steps", 0)
            env_total_input_tokens += episode_log.get("input_tokens", 0)
            env_total_output_tokens += episode_log.get("output_tokens", 0)

        # Calculate mean and standard error for the environment
        env_avg_progress = sum(env_episode_progress) / env_total_episodes if env_total_episodes else 0.0
        env_std_dev = (
            math.sqrt(sum((x - env_avg_progress) ** 2 for x in env_episode_progress) / env_total_episodes)
            if env_total_episodes > 1
            else 0.0
        )
        env_std_error = env_std_dev / math.sqrt(env_total_episodes) if env_total_episodes > 1 else 0.0

        # Collect progression percentages for overall summary
        total_progressions.extend(env_episode_progress)

        # Update overall totals
        total_envs += 1
        overall_total_input_tokens += env_total_input_tokens
        overall_total_output_tokens += env_total_output_tokens

        env_task_summaries = {}
        for task_name, task_runs in env_tasks.items():
            task_episode_progress = [run.get("progression", 0.0) for run in task_runs]
            task_count = len(task_runs)
            avg_task_progress = sum(task_episode_progress) / task_count if task_count else 0.0
            task_std_dev = (
                math.sqrt(sum((x - avg_task_progress) ** 2 for x in task_episode_progress) / task_count)
                if task_count > 1
                else 0.0
            )
            task_std_error = task_std_dev / math.sqrt(task_count) if task_count > 1 else 0.0

            env_task_summaries[task_name] = {
                "progression_percentage": 100 * avg_task_progress,
                "standard_error": 100 * task_std_error,
                "episodes_played": task_count,
            }

        avg_steps = env_total_steps / env_total_episodes if env_total_episodes else 0.0

        env_summary = {
            "progression_percentage": 100 * env_avg_progress,
            "standard_error": 100 * env_std_error,
            "average_steps": avg_steps,
            "episodes_played": env_total_episodes,
            "tasks": env_task_summaries,
            "input_tokens": env_total_input_tokens,
            "output_tokens": env_total_output_tokens,
        }

        # Save environment summary
        env_summary_filename = os.path.join(output_dir, env_name, f"{env_name}_summary.json")
        Path(env_summary_filename).parent.mkdir(parents=True, exist_ok=True)
        with open(env_summary_filename, "w") as f:
            json.dump(env_summary, f, indent=4)
        logging.info(f"Results saved for {env_name} in {env_summary_filename}")

        # Collect environment summaries for overall summary
        overall_env_summaries[env_name] = {
            "progression_percentage": env_summary["progression_percentage"],
            "standard_error": env_summary["standard_error"],
            "episodes_played": env_summary["episodes_played"],
        }

    # Calculate overall mean and standard error
    total_episodes = len(total_progressions)
    overall_avg_progression = sum(total_progressions) / total_episodes if total_episodes > 0 else 0.0
    overall_std_dev = (
        math.sqrt(sum((x - overall_avg_progression) ** 2 for x in total_progressions) / total_episodes)
        if total_episodes > 1
        else 0.0
    )
    overall_std_error = overall_std_dev / math.sqrt(total_episodes) if total_episodes > 1 else 0.0

    summary = {
        "Final score": 100 * overall_avg_progression,
        "standard_error": 100 * overall_std_error,
        "environments": overall_env_summaries,
        "total_input_tokens": overall_total_input_tokens,
        "total_output_tokens": overall_total_output_tokens,
        "client": OmegaConf.to_container(config.client, resolve=True),
        "agent": OmegaConf.to_container(config.agent, resolve=True),
    }

    # Save overall summary
    summary_filename = os.path.join(output_dir, "summary.json")
    with open(summary_filename, "w") as f:
        json.dump(summary, f, indent=4)
    logging.info(f"Overall results saved in {summary_filename}")

    return summary


def print_summary_table(summary):
    print("\nSummary of Results:")
    print(f"Overall Average Progression: {summary['Final score']:.2f}% ± {summary['standard_error']:.2f}%")
    print("Per-Environment Results:")
    for env_name, env_data in summary["environments"].items():
        print(
            f"  {env_name}: {env_data['progression_percentage']:.2f}% ± {env_data['standard_error']:.2f}%, Episodes: {env_data['episodes_played']}"
        )


def wandb_save_artifact(config):
    # Initialize wandb run
    wandb.init(project=config.wandb.project_name, entity=config.wandb.entity_name)

    with open("./summary.json", "r") as f:
        json_data = json.load(f)

    wandb.log(json_data)

    wandb.finish()


def load_secrets(file_path):
    secrets = {}
    with open(file_path) as f:
        for line in f:
            key, value = line.strip().split("=", 1)
            secrets[key] = value
    return secrets


def setup_environment(
    openai_tag: str = "OPENAI_API_KEY",
    gemini_tag: str = "GEMINI_API_KEY",
    anthropic_tag: str = "ANTHROPIC_API_KEY",
    replicate_tag: str = "REPLICATE_API_KEY",
    organization: str = None,
    original_cwd: str = "",
):
    secrets = load_secrets(os.path.join(original_cwd, "SECRETS"))
    genai.configure(api_key=secrets[gemini_tag])
    os.environ["ANTHROPIC_API_KEY"] = secrets[anthropic_tag]
    os.environ["REPLICATE_API_TOKEN"] = secrets[replicate_tag]
    os.environ["OPENAI_API_KEY"] = secrets[openai_tag]
    if organization is not None:
        openai.organization = secrets[organization]
