import os
from fmrl.prompt_builder import HumanHistoryPromptBuilder
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
import pandas as pd
import sys
import multiprocessing
import traceback


def ascii_render(chars):
    rows, cols = chars.shape
    result = ""
    for i in range(rows):
        line = ""
        for j in range(cols):
            entry = chr(chars[i, j])
            line += entry

        if len(line) > 80:
            trimmed_line = line[:80].rstrip()
            if line[80:].strip():
                result += line + "\n"
            else:
                result += trimmed_line + "\n"
        else:
            result += line + "\n"

    lines = result.split("\n")

    if len(lines) > 22 and "St:" in lines[22] and "Dx:" in lines[22]:
        parts = lines[22].split("[")
        if len(parts) > 1:
            subparts = parts[1].split("the")
            if len(subparts) > 1:
                lines[22] = parts[0] + "[Agent the" + subparts[1]

    result = "\n".join(lines)
    return result


NO_ACTIONS = {
    "message to message": " ",
    "action triggering message": " ",
    "acting on a message": " ",
    "unknown action": " ",
    "unknown": " ",
    "unknown selection": " ",
    "inventory item not found": "?",
}


def clean_action(action):
    if action in NO_ACTIONS:
        return NO_ACTIONS[action]
    else:
        return action


def postprocess_human(data, history):
    summary = data["summary"]

    prompt_builder = HumanHistoryPromptBuilder(
        max_length=16000,
        max_history=history,
        summary=summary,
    )

    tty_chars = data["tty_chars"]
    tty_actions = data["action"]
    tty_cursor = data["tty_cursor"]
    tty_inventory = data["inventory"]
    samples = []

    for i in tqdm(range(tty_actions.shape[0] - 2)):
        inventory = tty_inventory[i]
        render = ascii_render(tty_chars[i])

        split_render = render.split("\n")
        if "#      " in split_render[0]:
            continue
        elif "# " in split_render[0]:
            split_render[0] = " " * 80

        render = "\n".join(render.split("\n"))

        cursor = f"{np.array2string(tty_cursor[i])}"
        action = clean_action(tty_actions[i])

        prompt_builder.update_history(inventory, render, cursor, action)
        samples.append({"text": prompt_builder.get_prompt() + "### Response:" + action})

    return samples


def load_and_process_game_id(path, game_id, history):
    try:
        data = np.load(f"{path}/{game_id}.npz")
        samples = postprocess_human(data, history)
        df = pd.DataFrame(samples)
        csv_file = f"human_dataset_{game_id}.csv"
        df.to_csv(csv_file, index=False, escapechar="\\")
        print(f"Wrote {csv_file}")
        return csv_file
    except Exception as e:
        print(f"Failed to process game ID {game_id}: {e}")
        traceback.print_exc()
        return None


def merge_csv_files(csv_files, output_file):
    combined_df = pd.concat(
        [pd.read_csv(file) for file in csv_files if file is not None]
    )
    combined_df.to_csv(output_file, index=False, escapechar="\\")
    print("Merging CSV files")
    for file in csv_files:
        if file is not None:
            os.remove(file)
    print(f"Merged CSV files into {output_file}")


def load_dataset_multiprocessing(path, game_ids, history, num_processes):
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(
            load_and_process_game_id, [(path, game_id, history) for game_id in game_ids]
        )
    merge_csv_files(results, "human_dataset.csv")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "config/generate_human_dataset.yaml"

    config = OmegaConf.load(config_file)

    if config.gameids:
        with open(config.gameids, "r") as file:
            gameids = [int(line.strip()) for line in file]
    else:
        gameids = [1]

    gameids = gameids[: config.num_games]

    print(gameids)
    load_dataset_multiprocessing(
        config.directory, gameids, config.history, num_processes=config.num_processes
    )
