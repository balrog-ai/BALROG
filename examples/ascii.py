import numpy as np
from gym import Wrapper
from gym import spaces
from nle.env import NLE
from nle.nethack import tty_render

# def ascii_render(chars):
#     rows, cols = chars.shape
#     result = ""
#     for i in range(rows):
#         result += "\n"
#         for j in range(cols):
#             entry = chr(chars[i, j])
#             result += entry
#     return result

# https://github.com/facebookresearch/nle/blob/main/nle/scripts/play.py
# def tty_render(chars, colors, cursor=None):
#     """Returns chars as string with ANSI escape sequences.

#     Args:
#       chars: A row x columns numpy array of chars.
#       colors: A numpy array of colors (0-15), same shape as chars.
#       cursor: An optional (row, column) index for the cursor,
#         displayed as underlined.

#     Returns:
#       A string with chars decorated by ANSI escape sequences.
#     """
#     rows, cols = chars.shape
#     if cursor is None:
#         cursor = (-1, -1)
#     cursor = tuple(cursor)
#     result = ""
#     for i in range(rows):
#         result += "\n"
#         for j in range(cols):
#             entry = "\033[%d;3%dm%s" % (
#                 # & 8 checks for brightness.
#                 bool(colors[i, j] & 8),
#                 colors[i, j] & ~8,
#                 chr(chars[i, j]),
#             )
#             if cursor != (i, j):
#                 result += entry
#             else:
#                 result += "\033[4m%s\033[0m" % entry
#     return result + "\033[0m"
 
if __name__=="__main__":    
    env = tasks.NetHackChallenge(
        **dict(
            # savedir="./experiment_outputs/dummy_ttyrec",
            character="@",
            max_episode_steps=100000000,
            # observation_keys=(
            #     "blstats",
            #     "tty_chars",
            #     "tty_cursor",
            #     "glyphs",
            #     "inv_strs",
            #     "inv_letters",
            # ),
            penalty_step=0.0,
            penalty_time=0.0,
            penalty_mode="constant",
            no_progress_timeout=100,
            # save_ttyrec_every=1,
        )
    )

    obs = env.reset()
    print(obs.keys())
    glyphs = obs["glyphs"]
    
    text = tty_render(obs["tty_chars"], obs["tty_colors"], obs["tty_cursor"])
    print(obs["tty_cursor"])
    print(text)
    # print(glyphs)
    # print(ascii_render(glyphs))