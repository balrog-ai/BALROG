import itertools
import re
from collections import defaultdict

import crafter
import gym
import numpy as np
from PIL import Image
from scipy import ndimage

from balrog.environments import Strings

ACTIONS = [
    "Noop",
    "Move West",
    "Move East",
    "Move North",
    "Move South",
    "Do",
    "Sleep",
    "Place Stone",
    "Place Table",
    "Place Furnace",
    "Place Plant",
    "Make Wood Pickaxe",
    "Make Stone Pickaxe",
    "Make Iron Pickaxe",
    "Make Wood Sword",
    "Make Stone Sword",
    "Make Iron Sword",
]

id_to_item = [0] * 19


dummyenv = crafter.Env()
for name, ind in itertools.chain(dummyenv._world._mat_ids.items(), dummyenv._sem_view._obj_ids.items()):
    name = (
        str(name)[str(name).find("objects.") + len("objects.") : -2].lower() if "objects." in str(name) else str(name)
    )
    id_to_item[ind] = name
player_idx = id_to_item.index("player")
del dummyenv

vitals = [
    "health",
    "food",
    "drink",
    "energy",
]

rot = np.array([[0, -1], [1, 0]])
directions = ["front", "right", "back", "left"]


def describe_inventory(info):
    result = ""

    status_str = "Your status:\n{}".format("\n".join(["- {}: {}/9".format(v, info["inventory"][v]) for v in vitals]))
    result += status_str + "\n\n"

    inventory_str = "\n".join(
        ["- {}: {}".format(i, num) for i, num in info["inventory"].items() if i not in vitals and num != 0]
    )
    inventory_str = (
        "Your inventory:\n{}".format(inventory_str) if inventory_str else "You have nothing in your inventory."
    )
    result += inventory_str  # + "\n\n"

    return result.strip()


REF = np.array([0, 1])


def rotation_matrix(v1, v2):
    dot = np.dot(v1, v2)
    cross = np.cross(v1, v2)
    rotation_matrix = np.array([[dot, -cross], [cross, dot]])
    return rotation_matrix


def describe_loc_precise(ref, P):
    """
    Describe the location of P relative to ref.
    Example: `1 step south and 4 steps west`
    """
    desc = []

    def distange_to_string(distance, direction):
        return f"{abs(distance)} step{'s' if abs(distance) > 1 else ''} {direction}"

    if ref[1] > P[1]:
        desc.append(distange_to_string(ref[1] - P[1], "north"))
    elif ref[1] < P[1]:
        desc.append(distange_to_string(ref[1] - P[1], "south"))
    if ref[0] > P[0]:
        desc.append(distange_to_string(ref[0] - P[0], "west"))
    elif ref[0] < P[0]:
        desc.append(distange_to_string(ref[0] - P[0], "east"))

    return " and ".join(desc) if desc else "at your location"


def describe_loc_old(ref, P):
    desc = []
    if ref[1] > P[1]:
        desc.append("north")
    elif ref[1] < P[1]:
        desc.append("south")
    if ref[0] > P[0]:
        desc.append("west")
    elif ref[0] < P[0]:
        desc.append("east")

    distance = abs(ref[1] - P[1]) + abs(ref[0] - P[0])
    distance_str = f"{distance} step{'s' if distance > 1 else ''} to your {'-'.join(desc)}"

    return distance_str


def get_edge_items(semantic, item_idx):
    item_mask = semantic == item_idx
    not_item_mask = semantic != item_idx
    item_edge = ndimage.binary_dilation(not_item_mask) & item_mask
    return item_edge


def describe_env(
    info,
    unique_items=True,
    precise_location=False,
    skip_items=[],
    edge_only_items=[],
):
    assert info["semantic"][info["player_pos"][0], info["player_pos"][1]] == player_idx
    semantic = info["semantic"][
        info["player_pos"][0] - info["view"][0] // 2 : info["player_pos"][0] + info["view"][0] // 2 + 1,
        info["player_pos"][1] - info["view"][1] // 2 + 1 : info["player_pos"][1] + info["view"][1] // 2,
    ]
    center = np.array([info["view"][0] // 2, info["view"][1] // 2 - 1])
    result = ""
    describe_loc = describe_loc_precise if precise_location else describe_loc_old
    obj_info_list = []

    facing = info["player_facing"]
    max_x, max_y = semantic.shape
    target_x = center[0] + facing[0]
    target_y = center[1] + facing[1]

    if 0 <= target_x < max_x and 0 <= target_y < max_y:
        target_id = semantic[int(target_x), int(target_y)]
        target_item = id_to_item[target_id]

        # skip grass, sand or path so obs here, since we are not displaying them
        if target_id in [id_to_item.index(o) for o in skip_items]:
            target_item = "nothing"

        obs = "You face {} at your front.".format(target_item)
    else:
        obs = "You face nothing at your front."

    # Edge detection
    edge_masks = {}
    for item_name in edge_only_items:
        item_idx = id_to_item.index(item_name)
        edge_masks[item_idx] = get_edge_items(semantic, item_idx)

    for i in range(semantic.shape[0]):
        for j in range(semantic.shape[1]):
            idx = semantic[i, j]
            if idx == player_idx:
                continue

            # only display the edge of items that are in edge_only_items
            if idx in edge_masks and not edge_masks[idx][i, j]:
                continue

            # skip grass, sand or path so obs is not too long
            if idx in [id_to_item.index(o) for o in skip_items]:
                continue

            obj_info_list.append((id_to_item[idx], describe_loc(np.array([0, 0]), np.array([i, j]) - center)))

    def extract_numbers(s):
        """Extract all numbers from a string."""
        return [int(num) for num in re.findall(r"\d+", s)]

    # filter out items, so we only display closest item of each type
    if unique_items:
        closest_obj_info_list = defaultdict(str)
        for item_name, loc in obj_info_list:
            loc_dist = sum(extract_numbers(loc))
            current_dist = (
                sum(extract_numbers(closest_obj_info_list[item_name]))
                if closest_obj_info_list[item_name]
                else float("inf")
            )

            if loc_dist < current_dist:
                closest_obj_info_list[item_name] = loc
        obj_info_list = [(name, loc) for name, loc in closest_obj_info_list.items()]

    if len(obj_info_list) > 0:
        status_str = "You see:\n{}".format("\n".join(["- {} {}".format(name, loc) for name, loc in obj_info_list]))
    else:
        status_str = "You see nothing away from you."
    result += status_str + "\n\n"
    result += obs.strip()

    return result.strip()


def describe_act(action):
    result = ""

    action_str = action.replace("do_", "interact_")
    action_str = action_str.replace("move_up", "move_north")
    action_str = action_str.replace("move_down", "move_south")
    action_str = action_str.replace("move_left", "move_west")
    action_str = action_str.replace("move_right", "move_east")

    act = "You took action {}.".format(action_str)
    result += act

    return result.strip()


def describe_status(info):
    if info["sleeping"]:
        return "You are sleeping, and will not be able take actions until energy is full.\n\n"
    elif info["dead"]:
        return "You died.\n\n"
    else:
        return ""


def describe_frame(
    info,
    unique_items=True,
    precise_location=False,
    skip_items=[],
    edge_only_items=[],
):
    try:
        result = ""

        result += describe_status(info)
        result += "\n\n"
        result += describe_env(
            info,
            unique_items=unique_items,
            precise_location=precise_location,
            skip_items=skip_items,
            edge_only_items=edge_only_items,
        )
        result += "\n\n"

        return result.strip(), describe_inventory(info)
    except Exception:
        return "Error, you are out of the map.", describe_inventory(info)


class CrafterLanguageWrapper(gym.Wrapper):
    default_iter = 10
    default_steps = 10000

    def __init__(
        self,
        env,
        task="",
        max_episode_steps=2,
        unique_items=True,
        precise_location=False,
        skip_items=[],
        edge_only_items=[],
    ):
        super().__init__(env)
        self.score_tracker = 0
        self.language_action_space = Strings(ACTIONS)
        self.default_action = "Noop"
        self.max_steps = max_episode_steps
        self.achievements = None

        self.unique_items = unique_items
        self.precise_location = precise_location
        self.skip_items = skip_items
        self.edge_only_items = edge_only_items

    def get_text_action(self, action):
        return self.language_action_space._values[action]

    def _step_impl(self, action):
        obs, reward, done, info = super().step(action)
        # extra stuff for language wrapper
        aug_info = info.copy()
        aug_info["sleeping"] = self.env._player.sleeping
        aug_info["player_facing"] = self.env._player.facing
        aug_info["dead"] = self.env._player.health <= 0
        aug_info["unlocked"] = {
            name
            for name, count in self.env._player.achievements.items()
            if count > 0 and name not in self.env._unlocked
        }
        aug_info["view"] = self.env._view
        return obs, reward, done, aug_info

    def reset(self):
        self.env.reset()
        obs, reward, done, info = self._step_impl(0)
        self.score_tracker = 0
        self.achievements = None
        return self.process_obs(obs, info)

    def step(self, action):
        obs, reward, done, info = self._step_impl(self.language_action_space.map(action))
        self.score_tracker = self.update_progress(info)
        obs = self.process_obs(obs, info)
        return obs, reward, done, info

    def process_obs(self, obs, info):
        img = Image.fromarray(self.env.render()).convert("RGB")
        long_term_context, short_term_context = describe_frame(
            info,
            unique_items=self.unique_items,
            precise_location=self.precise_location,
            skip_items=self.skip_items,
            edge_only_items=self.edge_only_items,
        )

        return {
            "text": {
                "long_term_context": long_term_context,
                "short_term_context": short_term_context,
            },
            "image": img,
            "obs": obs,
        }

    def update_progress(self, info):
        self.score_tracker = 0 + sum([1.0 for k, v in info["achievements"].items() if v > 0])
        self.achievements = info["achievements"]
        return self.score_tracker

    def get_stats(self):
        return {
            "score": self.score_tracker,
            "progression": float(self.score_tracker) / 22.0,
            "achievements": self.achievements,
        }
