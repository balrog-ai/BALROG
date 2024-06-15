import os
import pickle
from fmrl.prompt_builder import HumanHistoryPromptBuilder
import argparse
import numpy as np
from tqdm import tqdm


def render_human_data(render, inventory, cursor):
    return


def ascii_render(chars):
    rows, cols = chars.shape
    result = ""
    for i in range(rows):
        for j in range(cols):
            entry = chr(chars[i, j])
            result += entry
        result += "\n"
    return result


def postprocess_human(data):

    summary = data["summary"]

    prompt_builder = HumanHistoryPromptBuilder(
        max_length=8000,
        max_history=32,
        summary=summary,
    )

    tty_chars = data["tty_chars"]
    tty_actions = data["action"]
    tty_cursor = data["tty_cursor"]
    tty_inventory = data["inventory"]
    samples = []

    for i in tqdm(range(tty_chars.shape[0] - 2)):
        inventory = tty_inventory[i]
        render = ascii_render(tty_chars[i])
        cursor = f"Cursor: {np.array2string(tty_cursor[i])}"
        action = tty_actions[i]

        prompt_builder.update_history(inventory, render, cursor, action)
        samples.append({"text": prompt_builder.get_prompt() + "### Response:" + action})

    return samples


import pandas as pd


def load_dataset(path, game_ids):
    samples = []

    for game_id in game_ids:
        print(path)

        data = np.load(f"{path}/{game_id}_data.npz")
        samples.extend(postprocess_human(data))

    df = pd.DataFrame(samples)
    df.to_csv(f"{game_id}_timesteps.csv", index=False)
    print("WROTE CSV")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="autoascend",
    )
    parser.add_argument(
        "--gameids",
        type=str,
    )

    args = parser.parse_args()
    path = args.directory

    if args.gameids:
        # Read the gameids from a txt file, they are on different lines each
        with open(args.gameids, "r") as file:
            gameids = [int(line.strip()) for line in file]
    else:
        gameids = [1]

    load_dataset(path, gameids)
