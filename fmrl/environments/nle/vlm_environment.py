import numpy as np
from gym import spaces
from nle_language_wrapper import NLELanguageWrapper
from fmrl.renderer import ImageRenderer
from idm.inverse_dynamics.utils import ascii_render, find_n_of_m_end

from PIL import Image


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

class NLE_VLMWrapper(NLELanguageWrapper):
    def __init__(self, env, use_language_action=True):
        super().__init__(env, use_language_action)
        self.observation_space = spaces.Space()
        self.image_renderer = ImageRenderer()
    
    def nle_process_obsv(self, nle_obsv):

        text_obsv = super().nle_obsv_to_language(nle_obsv)
        inventory = text_obsv["text_inventory"]
        cursor = np.array2string(np.array((nle_obsv["tty_cursor"][1], nle_obsv["tty_cursor"][0])))
        
        observation = dict(
            tty_chars=nle_obsv["tty_chars"],
            tty_colors=np.where(
                nle_obsv["tty_colors"] > 15,
                0,
                nle_obsv["tty_colors"],  # If the color > 15, set foreground to black and background to the color - 16
            ),
            tty_cursor=nle_obsv["tty_cursor"],
            tty_background=np.clip(nle_obsv["tty_colors"] - 16, 0, 15),
        )

        image_array = self.image_renderer.render(observation)
        image = Image.fromarray(image_array)
        
        message, menu, stats = process_chars(nle_obsv["tty_chars"])

        return inventory, message, menu, stats, cursor, image