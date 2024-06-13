import os
import pickle
from fmrl.environments.nle.utils import text_render, render_human_data, ascii_render
from fmrl.prompt_builder import ConcatHistoryPromptBuilder
from generate_episodes import nle_action_textmap
import argparse
import numpy as np
from tqdm import tqdm


def load_episode(args, path, game_id):
    if args.dataset_type == "autoascend":
        with open(os.path.join(path, f"{game_id}_summary.pkl"), "rb") as file:
            summary = pickle.load(file)
        with open(os.path.join(path, f"{game_id}_data.pkl"), "rb") as file:
            data = pickle.load(file)
    elif args.dataset_type == "human":
        print(game_id)
        path = "/Users/davidepaglieri/Desktop/repos/fmrl/idm"
        data = np.load(f"{path}/{game_id}_data.npz")
        summary = None  # To be created if needed. (log dump)

    return summary, data


def postprocess(summary, timesteps):
    # TODO: different prompting strategies here
    prompt_builder = ConcatHistoryPromptBuilder(
        max_length=8000,
    )

    samples = []

    for timestep in timesteps:
        # TODO: different observation rendering strategies here

        obs = text_render(timestep)
        action = timestep["action"]

        prompt_builder.update_observation(obs)
        prompt_builder.update_action(action)

        samples.append(
            {
                "prompt": prompt_builder.get_prompt(),
                "completion": nle_action_textmap[action],
            }
        )

    return samples


def postprocess_human(data):
    # TODO: different prompting strategies here
    prompt_builder = ConcatHistoryPromptBuilder(
        max_length=8000,
    )

    tty_chars = data["tty_chars"]
    tty_actions = data["action"]
    tty_cursor = data["tty_cursor"]
    tty_inventory = data["inventory"]
    samples = []

    for i in tqdm(range(tty_chars.shape[0] - 2)):
        inventory = tty_inventory[i]
        render = ascii_render(tty_chars[i])
        cursor = np.array2string(tty_cursor[i])
        action = tty_actions[i]

        obs = render_human_data(render, inventory, cursor)

        prompt_builder.update_observation(obs)
        prompt_builder.update_action(action)

        samples.append(
            {
                "prompt": prompt_builder.get_prompt(),
                "completion": nle_action_textmap.get(action, " "),
            }
        )
        if i == 1000:
            print(obs)
            break

    return samples


def load_dataset(args, path, game_ids):
    samples = []

    for game_id in game_ids:

        if args.dataset_type == "autoascend":
            summary, data = load_episode(args, path, game_id)
            samples.extend(postprocess(args, summary, data))

        elif args.dataset_type == "human":
            summary, data = load_episode(args, path, game_id)
            samples.extend(postprocess_human(data))

    with open("samples.pkl", "wb") as file:
        pickle.dump(samples, file)


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
    file = args.directory

    if args.gameids:
        # Read the gameids from a txt file, they are on different lines each
        with open(args.gameids, "r") as file:
            gameids = [int(line.strip()) for line in file]
    else:
        gameids = [1]

    load_dataset(args, file, gameids)
