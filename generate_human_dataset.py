import os
import pickle
from fmrl.prompt_builder import HumanHistoryPromptBuilder
import argparse
import numpy as np
from tqdm import tqdm
from idm.inverse_dynamics.utils import obs_to_message


def render_human_data(render, inventory, cursor):
    return


def ascii_render(chars):
    rows, cols = chars.shape
    result = ""
    for i in range(rows):
        line = ""
        for j in range(cols):
            entry = chr(chars[i, j])
            line += entry

        # If the line is longer than 80 characters, check for trailing spaces to strip
        if len(line) > 80:
            trimmed_line = line[:80].rstrip()
            if line[80:].strip():  # Check if there's more text after column 80
                result += line + "\n"
            else:
                result += trimmed_line + "\n"
        else:
            result += line + "\n"

    # Split the result into lines for further processing
    lines = result.split("\n")

    # Check if the status line exists and modify it
    if len(lines) > 22 and "St:" in lines[22] and "Dx:" in lines[22]:
        parts = lines[22].split("[")
        if len(parts) > 1:
            subparts = parts[1].split("the")
            if len(subparts) > 1:
                lines[22] = parts[0] + "[Agent the" + subparts[1]

    # Join the lines back together
    result = "\n".join(lines)


NO_ACTIONS = {
    "message to message": "",
    "action triggering message": "",
    "acting on a message": "",
    "unknown action": "",
    "unknown": "",
    "unknown selection": "",
    "inventory item not found": "?",
}


def clean_action(action):
    if action in NO_ACTIONS:
        return NO_ACTIONS[action]
    else:
        return action


def postprocess_human(data):

    summary = data["summary"]

    prompt_builder = HumanHistoryPromptBuilder(
        max_length=16000,
        max_history=8,
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


import pandas as pd


def load_dataset(path, game_ids):
    samples = []

    for game_id in game_ids:
        print(path)

        data = np.load(f"{path}/{game_id}.npz")
        samples.extend(postprocess_human(data))

    df = pd.DataFrame(samples)
    df.to_csv(f"human_dataset.csv", index=False, escapechar="\\")
    print("WROTE CSV")


import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default="human_labeled")
    parser.add_argument("--gameids", type=str, default="human_labeled/gameids.txt")

    args = parser.parse_args()
    path = args.directory

    if args.gameids:
        # Read the gameids from a txt file, they are on different lines each
        with open(args.gameids, "r") as file:
            gameids = [int(line.strip()) for line in file]
    else:
        gameids = [1]

    print(gameids)
    load_dataset(path, gameids)
