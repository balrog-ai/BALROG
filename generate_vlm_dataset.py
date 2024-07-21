import os
from fmrl.prompt_builder import VLMHistoryPromptBuilder
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
import json
import sys
import multiprocessing
import traceback

from datasets import Dataset, Features, Value, Image as HuggingFaceImage
from PIL import Image
from fmrl.renderer import ImageRenderer
from idm.inverse_dynamics.utils import ascii_render, find_n_of_m_end

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

def process_chars(chars, ascii=False):
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
        stats = lines[22].strip() + "\n" + lines[23].strip() + "\n"

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

def postprocess_human(data, history, image_history, game_id):

    prompt_builder = VLMHistoryPromptBuilder(
        max_history=history,
        image_history=image_history
    )

    tty_chars = data["tty_chars"]
    tty_colors = data["tty_colors"]
    tty_actions = data["action"]
    tty_cursor = data["tty_cursor"]
    tty_inventory = data["inventory"]
    
    ids = []
    prompts = []
    actions = []
    image_1_list = []
    image_2_list = []

    image_renderer = ImageRenderer()

    for i in tqdm(range(tty_actions.shape[0] - 2)):

        if i >= 50:
            break
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
        image = Image.fromarray(image_array)

        cursor = f"{np.array2string(tty_cursor[i])}"
        action = clean_action(tty_actions[i])

        prompt_builder.update_history(
            inventory, message, menu, stats, cursor, action, image  # Pass image
        )
        prompt, images = prompt_builder.get_prompt()  # Adjusted here
        ids.append(f"{game_id}_{i}")
        prompts.append(prompt)
        actions.append(action)

        # Handle two images
        image_1 = images[0] if len(images) > 0 else None
        image_2 = images[1] if len(images) > 1 else None
        image_1_list.append(image_1)
        image_2_list.append(image_2)

    dataset_dict = {
        "id": ids,
        "prompt": prompts,
        "action": actions,
        "image_1": image_1_list,
        "image_2": image_2_list,
    }

    return dataset_dict

def load_and_process_game_id(path, game_id, history, image_history):
    try:
        data = np.load(f"{path}/{game_id}.npz")
        samples = postprocess_human(data, history, image_history, game_id)
        return samples
    except Exception as e:
        print(f"Failed to process game ID {game_id}: {e}")
        traceback.print_exc()
        return None

def merge_datasets(datasets, output_file):
    dataset_dict = {
        "id": [],
        "prompt": [],
        "action": [],
        "image_1": [],
        "image_2": [],
    }
    
    count = 0
    for ds in datasets:
        if ds is not None:
            count += 1
            dataset_dict["id"].extend(ds["id"])
            dataset_dict["prompt"].extend(ds["prompt"])
            dataset_dict["action"].extend(ds["action"])
            dataset_dict["image_1"].extend(ds["image_1"])
            dataset_dict["image_2"].extend(ds["image_2"])
                
    print(f"Processed {count} games")

    # Define the features with two image columns
    features = Features({
        "id": Value("string"),
        "prompt": Value("string"),
        "action": Value("string"),
        "image_1": HuggingFaceImage(),
        "image_2": HuggingFaceImage()
    })

    dataset = Dataset.from_dict(dataset_dict, features=features)
    dataset.save_to_disk(output_file)
    dataset.push_to_hub(f"pagli98/{output_file}", private=False)

    print(f"Merged datasets into {output_file}")

def load_dataset_multiprocessing(path, game_ids, history, image_history, num_processes):
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(
            load_and_process_game_id, [(path, game_id, history, image_history) for game_id in game_ids]
        )
    merge_datasets(results, "human_dataset.hf")

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
    gameids = gameids[:config.max_games]
    print(gameids)
    load_dataset_multiprocessing(
        config.directory, gameids, config.history, config.image_history, num_processes=config.num_processes
    )