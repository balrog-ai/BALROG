from nle.nethack import tty_render
from nle_language_wrapper.nle_language_obsv import NLELanguageObsv

nle_language = NLELanguageObsv()


def ascii_render(chars):
    rows, cols = chars.shape
    result = ""
    for i in range(rows):
        for j in range(cols):
            entry = chr(chars[i, j])
            result += entry
        result += "\n"
    return result


def tty_render(nle_obsv):
    """Returns chars as string with ANSI escape sequences.

    Args:
      chars: A row x columns numpy array of chars.
      colors: A numpy array of colors (0-15), same shape as chars.
      cursor: An optional (row, column) index for the cursor,
        displayed as underlined.

    Returns:
      A string with chars decorated by ANSI escape sequences.
    """

    chars = nle_obsv["tty_chars"]
    colors = nle_obsv["tty_colors"]
    cursor = nle_obsv["tty_cursor"]

    rows, cols = chars.shape
    # if cursor is None:
    #     cursor = (-1, -1)
    # cursor = tuple(cursor)
    result = ""
    for i in range(rows):
        result += "\n"
        for j in range(cols):
            result += chr(chars[i, j])
            # entry = "\033[%d;3%dm%s" % (
            #     # & 8 checks for brightness.
            #     bool(colors[i, j] & 8),
            #     colors[i, j] & ~8,
            #     chr(chars[i, j]),
            # )
            # if cursor != (i, j):
            #     result += entry
            # else:
            #     result += "\033[4m%s\033[0m" % entry
    # return result + "\033[0m"
    return result


def nle_obsv_to_language(nle_obsv):
    """Translate NLE Observation into a language observation.
    Args:
        nle_obsv (dict): NLE observation from the base environment
    Returns:
        (dict): language observation
    """
    glyphs = nle_obsv["glyphs"]
    blstats = nle_obsv["blstats"]
    tty_cursor = nle_obsv["tty_cursor"]
    inv_strs = nle_obsv["inv_strs"]
    inv_letters = nle_obsv["inv_letters"]
    tty_chars = nle_obsv["tty_chars"]
    return {
        "text_glyphs": nle_language.text_glyphs(glyphs, blstats).decode("latin-1"),
        "text_message": nle_language.text_message(tty_chars).decode("latin-1"),
        "text_blstats": nle_language.text_blstats(blstats).decode("latin-1"),
        "text_inventory": nle_language.text_inventory(inv_strs, inv_letters).decode(
            "latin-1"
        ),
        "text_cursor": nle_language.text_cursor(glyphs, blstats, tty_cursor).decode(
            "latin-1"
        ),
    }


def text_render(nle_obsv):
    key_name_pairs = [
        ("text_blstats", "statistics"),
        ("text_glyphs", "glyphs"),
        ("text_message", "message"),
        ("text_inventory", "inventory"),
        ("text_cursor", "cursor"),
    ]
    text_obsv = nle_obsv_to_language(nle_obsv)
    return "\n".join([f"{name}[\n{text_obsv[key]}\n]" for key, name in key_name_pairs])


if __name__ == "__main__":
    import nle
    import gym

    env = gym.make("NetHackChallenge-v0", no_progress_timeout=100)
    obs = env.reset()

    print(tty_render(obs))

    breakpoint()
