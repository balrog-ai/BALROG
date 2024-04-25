from PIL import Image, ImageDraw, ImageFont
import numpy as np
from nle_language_wrapper import NLELanguageWrapper

MAX_ACTION_LENGTH = max([len(action_strs[0]) for action, action_strs in NLELanguageWrapper.all_nle_action_map.items()] + [len("ACTION HISTORY"),])

def create_texture_map():
    COLORS = [
        "#000000",
        "#800000",
        "#008000",
        "#808000",
        "#000080",
        "#800080",
        "#008080",
        "#808080",  # - flipped these ones around
        "#C0C0C0",  # | the gray-out dull stuff
        "#FF0000",
        "#00FF00",
        "#FFFF00",
        "#0000FF",
        "#FF00FF",
        "#00FFFF",
        "#FFFFFF",
    ]
    
    # Load a font (using default font here)
    dummy_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    font = ImageFont.truetype("./Hack-Regular.ttf", 12)
    cell_width, cell_height = map(max, zip(*[dummy_draw.textbbox((0, 0), text=chr(i), font=font)[2:] for i in range(256)]))

    # Create an image
    img_width = cell_width * 64
    img_height = cell_height * 64
    img = Image.new('RGB', (img_width, img_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    for color_idx, color in enumerate(COLORS):
        x_offset = (color_idx % 4) * (cell_width * 16)
        y_offset = (color_idx // 4) * (cell_height * 16)
        for i in range(256):
            x = (i % 16) * cell_width + x_offset
            y = (i // 16) * cell_height + y_offset
            character = chr(i)
            _, _, text_width, _ = draw.textbbox((0, 0), text=character, font=font)
            draw.text((x + (cell_width - text_width) / 2, y + 1), character, font=font, fill=COLORS[color_idx])

    # # DEBUG: Draw grid lines for columns and rows
    # for col in range(65):  # +1 to close the grid on the right side
    #     draw.line([(col * cell_width, 0), (col * cell_width, img_height)], fill=(255, 0, 0))
    # for row in range(65):  # +1 to close the grid on the bottom side
    #     draw.line([(0, row * cell_height), (img_width, row * cell_height)], fill=(255, 0, 0))
    
    return img

def make_atlas():
    image = np.array(create_texture_map())
    cell_height = image.shape[0] // 64
    cell_width = image.shape[1] // 64
    texture_atlas = image.reshape(4,16,cell_height,4,16,cell_width,3).transpose(0,3,1,4,2,5,6).reshape(4096,cell_height,cell_width,3)
    return texture_atlas
    # new_image = np.zeros((cell_height, 4096*cell_width, 3), dtype=np.uint8)
    # for j in range(16):
    #     for i, ch in enumerate(image[j]):
    #         new_image[:, (i+j*256)*cell_width:((i+j*256)+1)*cell_width] = ch
    # import matplotlib.pyplot as plt
    # plt.imsave("new_image.png", new_image)

DEFAULT_TEXTURE_ATLAS = make_atlas()
    
def tty_render_image(tty_chars, tty_colors, tty_cursor=None, *, texture_atlas=None):
    if texture_atlas is None:
        texture_atlas = DEFAULT_TEXTURE_ATLAS
    tty_colors_masked = tty_colors & 15 # I don't know why sometimes color > 15 but this is effectively what the ASCII renderers do
    return texture_atlas[tty_colors_masked*256+tty_chars].transpose(0,2,1,3,4).reshape(tty_chars.shape[0]*texture_atlas.shape[1], tty_chars.shape[1]*texture_atlas.shape[2],3)

def tty_render_image_action_history(tty_chars, tty_colors, action_history, tty_cursor=None, *, texture_atlas=None):
    tty_chars_extended = np.pad(tty_chars, ((0, 0), (0, MAX_ACTION_LENGTH + 1)), mode='constant', constant_values=ord(' '))
    tty_chars_extended[:, tty_chars.shape[1]] = ord('|')
    tty_colors_extended = np.pad(tty_colors, ((0, 0), (0, MAX_ACTION_LENGTH + 1)), mode='constant', constant_values=0)
    tty_colors_extended[:, tty_colors.shape[1]] = 7
    
    def to_array(string):
        arr = np.array([ord(c) for c in string])
        if len(arr) < MAX_ACTION_LENGTH:
            arr = np.pad(arr, (0, MAX_ACTION_LENGTH - len(arr)), mode='constant', constant_values=ord(' '))
        return arr
    
    tty_chars_extended[0,tty_chars.shape[1]+1:] = to_array("ACTION HISTORY")
    tty_colors_extended[0,tty_colors.shape[1]+1:] = 7
    
    tty_chars_extended[1,tty_chars.shape[1]+1:] = to_array("==============")
    tty_colors_extended[1,tty_colors.shape[1]+1:] = 7
    
    for i, action in enumerate(action_history[-(tty_chars.shape[0]-2):][::-1], 2):
        tty_chars_extended[i,tty_chars.shape[1]+1:] =  to_array(action)
        if i == 2:
            tty_colors_extended[i,tty_colors.shape[1]+1:] = 15
        else:
            tty_colors_extended[i,tty_colors.shape[1]+1:] = 7
    return tty_render_image(tty_chars_extended, tty_colors_extended, tty_cursor, texture_atlas=texture_atlas)

if __name__ == "__main__":
    from nle.env import tasks
    from nle.nethack import tty_render
    
    create_texture_map().save("texture_map.png")
    
    env = tasks.NetHackChallenge(
        **dict(
            # savedir="./experiment_outputs/dummy_ttyrec",
            character="@",
            max_episode_steps=100000000,
            penalty_step=0.0,
            penalty_time=0.0,
            penalty_mode="constant",
            no_progress_timeout=100,
            # save_ttyrec_every=1,
        )
    )

    obs = env.reset()
    chars = obs["tty_chars"]
    colors = obs["tty_colors"]

    print(tty_render(chars, colors, obs["tty_cursor"]))
    Image.fromarray(tty_render_image_action_history(chars, colors, ["esc",] * 200 + ["north", "south", "east", "west"])).save("test.png")