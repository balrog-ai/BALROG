import itertools

import crafter
import gym
import numpy as np
from PIL import Image

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


def describe_loc(ref, P):
    desc = []
    if ref[1] > P[1]:
        desc.append("north")
    elif ref[1] < P[1]:
        desc.append("south")
    if ref[0] > P[0]:
        desc.append("west")
    elif ref[0] < P[0]:
        desc.append("east")

    return "-".join(desc)


def describe_env(info):
    assert info["semantic"][info["player_pos"][0], info["player_pos"][1]] == player_idx
    semantic = info["semantic"][
        info["player_pos"][0] - info["view"][0] // 2 : info["player_pos"][0] + info["view"][0] // 2 + 1,
        info["player_pos"][1] - info["view"][1] // 2 + 1 : info["player_pos"][1] + info["view"][1] // 2,
    ]
    center = np.array([info["view"][0] // 2, info["view"][1] // 2 - 1])
    result = ""
    x = np.arange(semantic.shape[1])
    y = np.arange(semantic.shape[0])
    x1, y1 = np.meshgrid(x, y)
    loc = np.stack((y1, x1), axis=-1)
    dist = np.absolute(center - loc).sum(axis=-1)
    obj_info_list = []

    facing = info["player_facing"]
    max_y, max_x = semantic.shape
    target_x = center[0] + facing[0]
    target_y = center[1] + facing[1]

    if 0 <= target_x < max_x and 0 <= target_y < max_y:
        target_id = semantic[int(target_x), int(target_y)]
        target_item = id_to_item[target_id]
        obs = "You face {} at your front.".format(target_item)
    else:
        obs = "You face nothing at your front."

    for idx in np.unique(semantic):
        if idx == player_idx:
            continue

        smallest = np.unravel_index(np.argmin(np.where(semantic == idx, dist, np.inf)), semantic.shape)
        obj_info_list.append(
            (
                id_to_item[idx],
                dist[smallest],
                describe_loc(np.array([0, 0]), smallest - center),
            )
        )

    if len(obj_info_list) > 0:
        status_str = "You see:\n{}".format(
            "\n".join(["- {} {} steps to your {}".format(name, dist, loc) for name, dist, loc in obj_info_list])
        )
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


def describe_frame(info):
    try:
        result = ""

        result += describe_status(info)
        result += "\n\n"
        result += describe_env(info)
        result += "\n\n"

        return result.strip(), describe_inventory(info)
    except Exception:
        breakpoint()
        return "Error, you are out of the map."


class CrafterLanguageWrapper(gym.Wrapper):
    default_iter = 10
    default_steps = 10000

    def __init__(
        self,
        env,
        task="",
        max_episode_steps=2,
    ):
        super().__init__(env)
        self.score_tracker = 0
        self.language_action_space = Strings(ACTIONS)
        self.default_action = "Noop"
        self.max_steps = max_episode_steps
        self.achievements = None

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
        long_term_context, short_term_context = describe_frame(info)

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
