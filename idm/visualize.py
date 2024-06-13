import numpy as np
import os


def ascii_render(chars):
    rows, cols = chars.shape
    result = ""
    for i in range(rows):
        for j in range(cols):
            entry = chr(chars[i, j])
            result += entry
        result += "\n"
    return result


# The files are in the following format:

"""
session = np.load(filename)                 # Dict of observations
tty_chars = session['tty_chars']            # Shape = (T, 24, 80)
tty_colors = session['tty_colors']          # Shape = (T, 24, 80)
tty_actions = session['tty_actions']        # Shape = (T, 1)
tty_cursor = session['tty_cursor']          # Shape = (T, 2,)
"""


def main():

    file = "/Users/davidepaglieri/Desktop/repos/fmrl/idm/labeled_145461.npz"
    # Check all the files in the games folder

    data = np.load(file)

    tty_chars = data["tty_chars"]
    # tty_colors = data['tty_colors']
    # tty_background = data['tty_background']
    tty_actions = data["actions"]
    tty_cursor = data["tty_cursor"]
    tty_inventory = data["inventory"]

    with open("game.txt", "a") as f:
        for i in range(tty_chars.shape[0] - 2):
            if i >= 1000:
                break
            render = ascii_render(tty_chars[i])
            cursor = np.array2string(tty_cursor[i])
            action = tty_actions[i]
            inventory = tty_inventory[i]
            # print(type(action))
            f.write(inventory)
            f.write(render)
            f.write(cursor)
            f.write("ACTION:")
            f.write(action)
            f.write("\n")
            f.write("#" * 80)
            f.write("\n")


if __name__ == "__main__":
    main()
