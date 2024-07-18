# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Renderer for NetHack with customizable foreground and background colors."""

import struct
from typing import Mapping, Tuple, Optional

from nle import _pynethack
import numpy as np
import pygame as pg

FOREGROUND_COLORS = [
    # Dark
    "#000000",
    "#770000",
    "#007700",
    "#aa7700",
    "#000077",
    "#aa00aa",
    "#00aaaa",
    "#ffffff",
    # Bright
    "#888888",
    "#ff0000",
    "#00ff00",
    "#ffff00",
    "#0000ff",
    "#ff00ff",
    "#00ffff",
    "#ffffff",
]

BACKGROUND_COLORS = dict(
    cursor="#555555",  # Gray.
    corpse="#0000aa",  # Blue.
    invis="#333333",  # Gray.
    detect="#333333",  # Gray.
    pet="#ffffff",  # White.
    ridden="#333333",  # Gray.
    statue="#333333",  # Gray.
    objpile="#333333",  # Gray.
    bwlava="#330000",  # Red.
    default="#000000",  # Black.
)

# Rendered tiles are cached in this dictionary to avoid re-rendering later.
TILE_CACHE = {}


def hex_string_to_color_tuple(hex_string: str) -> Tuple[int, int, int]:
    """Convert a hex string prefixed with '#' to RGB tuple."""
    color_three_bytes = int(hex_string[1:], 16)
    blue, green, red, _ = struct.unpack("4B", struct.pack("I", color_three_bytes))
    return (red, green, blue)


def get_special_background(special: int) -> str:
    """Color the character and background based on its special flags."""
    if bool(special & _pynethack.nethack.MG_CORPSE):
        return "corpse"
    elif bool(special & _pynethack.nethack.MG_INVIS):
        return "invis"
    elif bool(special & _pynethack.nethack.MG_DETECT):
        return "detect"
    elif bool(special & _pynethack.nethack.MG_PET):
        return "pet"
    elif bool(special & _pynethack.nethack.MG_RIDDEN):
        return "ridden"
    elif bool(special & _pynethack.nethack.MG_STATUE):
        return "statue"
    elif bool(special & _pynethack.nethack.MG_OBJPILE):
        return "objpile"
    elif bool(special & _pynethack.nethack.MG_BW_LAVA):
        return "bwlava"
    else:
        return "default"


def represent_char(c: int) -> int:
    """Change the representation of certain characters to curses UI forms."""
    if c == 0:  # Represent invalid characters with blank space.
        return ord(" ")
    elif c == ord("`"):  # Represent boulders as 0 for readability.
        return ord("0")
    else:
        return c


class ImageRenderer:
    """Renderer using headless PyGame to render TTY observations as images."""

    def __init__(self):
        pg.font.init()
        self._font = pg.font.Font("./Hack-Regular.ttf", 14)

        # Make color tuples from the hex strings above.
        self._colors_map = np.array(
            list(map(hex_string_to_color_tuple, FOREGROUND_COLORS))
        )

        self._bgcolors = {
            k: hex_string_to_color_tuple(v) for k, v in BACKGROUND_COLORS.items()
        }

        # `specials` is a uint8 bitfield. Map all possible values of `specials` to
        # the background color that the we should use for it.
        self._specials_map = np.zeros((256, 3))
        for s in range(256):
            self._specials_map[s] = self._bgcolors[get_special_background(s)]

    def _render_char(
        self, c: str, color: Tuple[int, int, int], background: Tuple[int, int, int]
    ) -> np.ndarray:
        """Render a single character to a patch. Patches are cached for reuse."""
        key = (c, tuple(color), tuple(background))
        if key not in TILE_CACHE:
            # Render this single character to a small image patch.
            single_character = pg.surfarray.array3d(
                self._font.render(c, True, color, background)
            ).swapaxes(0, 1)
            # Ensure that the patch is exactly 16x8.
            single_character_fixed_size = np.zeros((16, 8, 3), dtype=np.uint8)
            height, width = single_character.shape[:2]
            single_character_fixed_size[:height, :width] = single_character[:16, :8]
            TILE_CACHE[key] = single_character_fixed_size
        return TILE_CACHE[key]

    def render(
        self,
        obs: Mapping[str, np.ndarray],
        foreground_colors: Optional[np.ndarray] = None,
        background_colors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Render an image of the TTY, with colors potentially overridden.

        Args:
          obs: Standard NLE observation dict, or observation dict from the NAO
            dataset. Must contain key `tty_chars`.
          foreground_colors: Optional RGB array of color overrides for the chars. If
            not specified, colors will be derived from `tty_colors` instead.
          background_colors: Optional RGB array of color overrides for tile
            backgrounds. If not specified, background colors will be derived from
            `specials` or `tty_background` keys of the observation.

        Returns:
          Rendered image in the form of an RGB numpy array.
        """
        # If colors are not specified, extract them from the observation.
        if foreground_colors is None:
            foreground_colors = self._colors_map[np.clip(obs["tty_colors"], 0, 15)]

        if background_colors is None:
            if "specials" in obs:
                padded_specials = np.pad(obs["specials"], [[1, 2], [0, 1]])
                padded_specials = np.clip(padded_specials, 0, 255)
                background_colors = self._specials_map[padded_specials]

            elif "tty_background" in obs:
                background_colors = self._colors_map[
                    np.clip(obs["tty_background"], 0, 15)
                ]

            else:
                background_colors = np.zeros((24, 80, 3), dtype=np.uint8)

            if "tty_cursor" in obs:
                background_colors[
                    min(obs["tty_cursor"][1], 23), min(obs["tty_cursor"][0], 79)
                ] = self._bgcolors["cursor"]

        tty_chars = np.clip(obs["tty_chars"], 0, 255)

        def _render_pos(x, y):
            return self._render_char(
                chr(represent_char(tty_chars[y, x])),
                foreground_colors[y, x],
                background_colors[y, x],
            )

        return np.vstack(
            [np.hstack([_render_pos(x, y) for x in range(80)]) for y in range(1, 22, 1)]
        )
