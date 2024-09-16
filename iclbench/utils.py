import os
import json
import logging
from collections import defaultdict
import wandb


def summarize_env_progressions(results_summaries: defaultdict, config) -> float:
    average_progression = 0.0
    for _, results in results_summaries.items():
        average_progression += int(results["progression_percentage"])
    average_progression /= len(results_summaries)

    results_summaries["Final score"] = average_progression
    results_summaries["model"] = config.model_id

    with open("summary.json", "w") as f:
        json.dump(results_summaries, f)
    logging.info(f"Results saved in summary.json")

    return average_progression


def wandb_save_artifact(config):
    # Initialize wandb run
    wandb.init(project=config.wandb.project_name, entity=config.wandb.entity_name)

    with open("./summary.json", "r") as f:
        json_data = json.load(f)

    wandb.log(json_data)

    wandb.finish()
