import random
from typing import Any, Dict

import gymnasium as gym
from nle import nle_language_obsv
from nle.language_wrapper.wrappers import nle_language_wrapper
from nle.nethack import USEFUL_ACTIONS
from PIL import Image

from balrog.environments import Strings
from balrog.environments.nle.progress import get_progress_system
from balrog.environments.nle.render import tty_render_image
from balrog.environments.nle.render_rgb import rgb_render_image


class NLELanguageWrapper(gym.Wrapper):
    def __init__(self, env, vlm=False):
        super().__init__(env)
        self.nle_language = nle_language_obsv.NLELanguageObsv()
        self.vlm = vlm
        self.done = False

        if not vlm:
            self.prompt_mode = "hybrid"
        else:
            self.prompt_mode = "language"

        self._setup_action_space()
        self.progress = get_progress_system(self.env)
        self.max_steps = self.env.unwrapped._max_episode_steps

    def _setup_action_space(self):
        """Determines the action set and initializes the action space."""
        env_id_lower = self.env.spec.id.lower()
        if "minihack" in env_id_lower:
            actions_to_descr = MINIHACK_ACTIONS_TO_DESCR
        elif "nethack" in env_id_lower:
            actions_to_descr = NLE_ACTIONS_TO_DESCR
        else:
            raise ValueError(f"Unsupported environment: {self.env.spec.id}")

        self._initialize_action_mappings(actions_to_descr)
        self.language_action_space = Strings(list(self.action_str_enum_map.keys()))

    def _initialize_action_mappings(self, actions_to_descr: Dict[str, str]):
        """Builds mappings from action strings to NLE enums and indices."""
        self.action_str_enum_map: Dict[str, Any] = {}
        self.action_enum_index_map: Dict[Any, int] = {}
        self.action_str_desc_map: Dict[str, str] = {}

        # Pre-calculate all possible action strings from the base environment
        all_action_strs = {
            action_str
            for action_strs in nle_language_wrapper.NLELanguageWrapper.all_nle_action_map.values()
            for action_str in action_strs
        }

        # Validate that all keys in our description map are valid actions
        missing_keys = [key for key in actions_to_descr if key not in all_action_strs]
        if missing_keys:
            raise KeyError(f"Action keys not found in NLE's action map: {', '.join(missing_keys)}")

        for action_enum in self.env.unwrapped.actions:
            action_index = self.env.unwrapped.actions.index(action_enum)
            possible_strs = nle_language_wrapper.NLELanguageWrapper.all_nle_action_map.get(action_enum, [])

            for action_str in possible_strs:
                if action_str in actions_to_descr:
                    self.action_str_enum_map[action_str] = action_enum
                    self.action_enum_index_map[action_enum] = action_index
                    self.action_str_desc_map[action_str] = actions_to_descr[action_str]

    def pre_reset(self):
        self.progress = get_progress_system(self.env)

    def reset(self, **kwargs):
        self.pre_reset()
        obs, info = self.env.reset(**kwargs)

        return self.post_reset(obs), info

    def post_reset(self, nle_obs):
        return self.nle_process_obs(nle_obs)

    def pre_step(self, action):
        nle_action_enum = self.action_str_enum_map[action]
        nle_action_idx = self.action_enum_index_map[nle_action_enum]

        return nle_action_idx

    def step(self, action):
        action = self.pre_step(action)

        obs, reward, term, trun, info = self.env.step(action)

        done = term or trun
        self.done = done if not self.done else self.done
        self.progress.update(obs, reward, self.done, info)

        return self.post_step(obs), reward, term, trun, info

    def post_step(self, nle_obs):
        return self.nle_process_obs(nle_obs)

    def render(self):
        mode = self.env.render_mode

        if mode == "tiles":
            return self.render_tiles_from_obs(self.env.unwrapped._last_obs)
        elif mode == "tty_image":
            return self.render_tty_from_obs(self.env.unwrapped._last_obs)
        else:
            return self.env.render()

    @property
    def default_action(self):
        if "minihack" in self.env.spec.id.lower():
            return "north"
        else:
            return "esc"

    def get_text_action(self, action):
        return self.action_str_enum_map[action]

    def nle_process_obs(self, nle_obs):
        img = Image.fromarray(self.render_tiles_from_obs(nle_obs)).convert("RGB") if self.vlm else None
        text = self.nle_obs_type(nle_obs)

        return {
            "text": text,
            "image": img,
            "obs": nle_obs,
        }

    def nle_obs_type(self, nle_obs):
        nle_obs = self.nle_obs_to_language(nle_obs)
        if self.prompt_mode == "language":
            return self.render_text(nle_obs)
        elif self.prompt_mode == "hybrid":
            return self.render_hybrid(nle_obs)
        else:
            raise ValueError(f'"{self.prompt_mode}" is not a valid prompt mode.')

    def render_tiles_from_obs(self, obs):
        # Custom tiles rendering from latest observation
        key_idx = self.env.unwrapped._observation_keys.index
        return rgb_render_image(obs[key_idx("glyphs")])

    def render_tty_from_obs(self, obs):
        key_idx = self.env.unwrapped._observation_keys.index
        return tty_render_image(obs[key_idx("tty_chars")], obs[key_idx("tty_colors")])

    def get_stats(self):
        return self.progress.__dict__

    def ascii_render(self, chars):
        rows, cols = chars.shape
        result = ""
        for i in range(rows):
            for j in range(cols):
                entry = chr(chars[i, j])
                result += entry
            result += "\n"
        return result

    def nle_obs_to_language(self, nle_obsv):
        """Translate NLE Observation into a language observation.
        Args:
            nle_obsv (dict): NLE observation from the base environment
        Returns:
            (dict): language observation
        """

        message = (
            nle_obsv["text_message"]
            if "text_message" in nle_obsv
            else self.nle_language.text_message(nle_obsv["tty_chars"]).decode("latin-1")
        )

        glyphs = nle_obsv["glyphs"]
        blstats = nle_obsv["blstats"]
        tty_cursor = nle_obsv["tty_cursor"]
        inv_strs = nle_obsv["inv_strs"]
        inv_letters = nle_obsv["inv_letters"]

        return {
            "text_glyphs": self.nle_language.text_glyphs(glyphs, blstats).decode("latin-1"),
            "text_message": message,
            "text_blstats": self.nle_language.text_blstats(blstats).decode("latin-1"),
            "text_inventory": self.nle_language.text_inventory(inv_strs, inv_letters).decode("latin-1"),
            "text_cursor": self.nle_language.text_cursor(glyphs, blstats, tty_cursor).decode("latin-1"),
            "tty_chars": nle_obsv["tty_chars"],
            "tty_cursor": nle_obsv["tty_cursor"],
        }

    def render_text(self, nle_obsv):
        long_term_observations = [
            ("text_message", "message"),
            ("text_glyphs", "language observation"),
            ("text_cursor", "cursor"),
        ]

        short_term_observations = [
            ("text_blstats", "statistics"),
            ("text_inventory", "inventory"),
        ]

        long_term_context = "\n".join([f"{name}:\n{nle_obsv[key]}\n" for key, name in long_term_observations])
        short_term_context = "\n".join([f"{name}:\n{nle_obsv[key]}\n" for key, name in short_term_observations])

        return {
            "long_term_context": long_term_context,
            "short_term_context": short_term_context,
        }

    def render_hybrid(self, nle_obsv):
        ascii_map = self.ascii_render(nle_obsv["tty_chars"])
        cursor = nle_obsv["tty_cursor"]
        cursor = f"(x={cursor[1]}, y={cursor[0]})"
        ascii_map = "\n".join(ascii_map.split("\n")[1:])  # remove first line

        nle_obsv["map"] = ascii_map
        nle_obsv["text_cursor"] = nle_obsv["text_cursor"] + "\n" + cursor

        long_term_observations = [
            ("text_message", "message"),
            ("text_glyphs", "language observation"),
            ("text_cursor", "cursor"),
            ("map", "map"),
        ]
        short_term_observation = [
            ("text_inventory", "inventory"),
        ]

        long_term_context = "\n".join([f"{name}:\n{nle_obsv[key]}\n" for key, name in long_term_observations])
        short_term_context = "\n".join([f"{name}:\n{nle_obsv[key]}\n" for key, name in short_term_observation])

        return {
            "long_term_context": long_term_context,
            "short_term_context": short_term_context,
        }


NLE_ACTIONS_TO_DESCR = {
    "north": "move north",
    "east": "move east",
    "south": "move south",
    "west": "move west",
    "northeast": "move northeast",
    "southeast": "move southeast",
    "southwest": "move southwest",
    "northwest": "move northwest",
    "far north": "move far north",
    "far east": "move far east",
    "far south": "move far south",
    "far west": "move far west",
    "far northeast": "move far northeast",
    "far southeast": "move far southeast",
    "far southwest": "move far southwest",
    "far northwest": "move far northwest",
    "up": "go up a staircase",
    "down": "go down a staircase (tip: you can only go down if you are standing on the stairs)",
    "wait": "rest one move while doing nothing",
    "more": "display more of the message (tip: ONLY ever use when current message ends with --More--)",
    "annotate": "leave a note about the level",
    "apply": "apply (use) a tool",
    "call": "name a monster or object, or add an annotation",
    "cast": "cast a spell",
    "close": "close an adjacent door",
    "open": "open an adjacent door",
    "dip": "dip an object into something",
    "drop": "drop an item",
    "droptype": "drop specific item types (specify in the next prompt)",
    "eat": "eat something (tip: replenish food when hungry)",
    "esc": "exit menu or message",
    "engrave": "engrave writing on the floor (tip: Elbereth)",
    "enhance": "advance or check weapons skills",
    "fire": "fire ammunition from quiver",
    "fight": "fight a monster (even if you only guess one is there)",
    "force": "force a lock",
    "inventory": "show your inventory",
    "invoke": "invoke ",
    "jump": "jump to a location",
    "kick": "kick an enemy or a locked door or chest",
    "look": "look at what is under you",
    "loot": "loot a box on the floor",
    "monster": "use a monster's special ability (when polymorphed)",
    "offer": "offer a sacrifice to the gods (tip: on an aligned altar)",
    # "overview": "display an overview of the dungeon",
    "pay": "pay your shopping bill",
    "pickup": "pick up things at the current location",
    "pray": "pray to the gods for help",
    "puton": "put on an accessory",
    "quaff": "quaff (drink) something",
    "quiver": "select ammunition for quiver",
    "read": "read a scroll or spellbook",
    "remove": "remove an accessory",
    "rub": "rub a lamp or a stone",
    "search": "search for hidden doors and passages",
    "swap": "swap wielded and secondary weapons",
    "takeoff": "take off one piece of armor",
    "takeoffall": "take off all armor",
    "teleport": "teleport to another level (if you have the ability)",
    "throw": "throw something (e.g. a dagger or dart)",
    "travel": "travel to a specific location on the map (tip: in the next action, specify > or < for stairs, { for fountain, and _ for altar)",
    "twoweapon": "toggle two-weapon combat",
    "untrap": "untrap something",
    "wear": "wear a piece of armor",
    "wield": "wield a weapon",
    "wipe": "wipe off your face",
    "zap": "zap a wand",
    "minus": "-",
    "space": " ",
    "apos": "'",
    "0": "0",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
}


MINIHACK_ACTIONS_TO_DESCR = {
    "north": "move north",
    "east": "move east",
    "south": "move south",
    "west": "move west",
    "northeast": "move northeast",
    "southeast": "move southeast",
    "southwest": "move southwest",
    "northwest": "move northwest",
    "far north": "move far north",
    "far east": "move far east",
    "far south": "move far south",
    "far west": "move far west",
    "far northeast": "move far northeast",
    "far southeast": "move far southeast",
    "far southwest": "move far southwest",
    "far northwest": "move far northwest",
    "up": "go up the stairs",
    "down": "go down the stairs",
    "wait": "rest one move while doing nothing",
    "more": "display more of the message",
    "apply": "apply (use) a tool",
    "close": "close an adjacent door",
    "open": "open an adjacent door",
    "eat": "eat something",
    "force": "force a lock",
    "kick": "kick an enemy or a locked door or chest",
    "loot": "loot a box on the floor",
    "pickup": "pick up things at the current location if there are any",
    "pray": "pray to the gods for help",
    "puton": "put on an accessory",
    "quaff": "quaff (drink) something",
    "search": "search for hidden doors and passages",
    "zap": "zap a wand",
}
