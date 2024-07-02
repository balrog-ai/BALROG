import os
from fmrl.prompt_builder import VLMHistoryPromptBuilder
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
import pandas as pd
import sys
import multiprocessing
import traceback

from PIL import Image
from fmrl.renderer import ImageRenderer
from idm.inverse_dynamics.utils import ascii_render, find_n_of_m_end

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


def process_chars(chars, ascii=False):
    """
    Process chars for the VLM

    Args:
        chars (numpy array): The characters from the observation
    Returns:
        message (str): The message from the observation
        menu (str): The menu from the observation
        stats (str): The stats from the observation
    """
    if ascii:
        ascii_map = chars
    else:
        ascii_map = ascii_render(chars)

    lines = ascii_map.split("\n")

    stats = ""
    if len(lines) > 22 and "St:" in lines[22] and "Dx:" in lines[22]:
        parts = lines[22].split("[")
        if len(parts) > 1:
            subparts = parts[1].split("the")
            if len(subparts) > 1:
                lines[22] = parts[0] + "[Agent the" + subparts[1]
        stats = lines[22] + "\n" + lines[23] + "\n"

    ascii_map = "\n".join(lines)
    message = lines[0]

    menu = ""
    if find_n_of_m_end(ascii_map):
        idx = 0
        for idx, char in enumerate(message):
            start_idx = idx
            if char.isalpha():
                break

        line_idx = 0
        while line_idx <= 24:
            menu += lines[line_idx][start_idx:].strip() + "\n"
            if find_n_of_m_end(lines[line_idx]):
                break
            line_idx += 1

    return message.strip(), menu, stats


def postprocess_human(data, history, game_id):

    prompt_builder = VLMHistoryPromptBuilder(
        max_history=history,
    )

    tty_chars = data["tty_chars"]
    tty_colors = data["tty_colors"]
    tty_actions = data["action"]
    tty_cursor = data["tty_cursor"]
    tty_inventory = data["inventory"]
    samples = []

    image_renderer = ImageRenderer()

    for i in tqdm(range(tty_actions.shape[0] - 2)):
        inventory = tty_inventory[i]
        message, menu, stats = process_chars(tty_chars[i])

        observation = dict(
            tty_chars=tty_chars[i],
            tty_colors=np.where(
                tty_colors[i] > 15,
                0,
                tty_colors[
                    i
                ],  # If the color > 15, set foreground to black and background to the color - 16
            ),
            tty_cursor=tty_cursor[i],
            tty_background=np.clip(tty_colors - 16, 0, 15)[i],
        )

        image_array = image_renderer.render(observation)
        image_path = f"dataset/images/{game_id}/{i}.png"
        os.makedirs(f"dataset/images/{game_id}", exist_ok=True)
        Image.fromarray(image_array).save(image_path)

        cursor = f"{np.array2string(tty_cursor[i])}"
        action = clean_action(tty_actions[i])

        prompt_builder.update_history(
            inventory, message, menu, stats, cursor, action, image_path
        )
        prompt, image_path = prompt_builder.get_prompt()
        samples.append({"text": prompt + "### Response:" + action, "image": image_path})

    return samples


def load_and_process_game_id(path, game_id, history):
    try:
        data = np.load(f"{path}/{game_id}.npz")
        samples = postprocess_human(data, history, game_id)
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
    print(gameids)
    load_dataset_multiprocessing(
        config.directory, gameids, config.history, num_processes=config.num_processes
    )
