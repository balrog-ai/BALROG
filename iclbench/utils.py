import json
import logging
import os
from collections import defaultdict
from omegaconf import OmegaConf

import google.generativeai as genai
import openai
import wandb


def summarize_env_progressions(results_summaries: defaultdict, config) -> float:
    average_progression = 0.0
    for _, results in results_summaries.items():
        average_progression += int(results["progression_percentage"])
    average_progression /= len(results_summaries)

    results_summaries["Final score"] = average_progression
    results_summaries["client"] = OmegaConf.to_container(config.client, resolve=True)
    results_summaries["agent"] = OmegaConf.to_container(config.agent, resolve=True)

    with open("summary.json", "w") as f:
        json.dump(results_summaries, f)
    logging.info("Results saved in summary.json")

    return average_progression


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
