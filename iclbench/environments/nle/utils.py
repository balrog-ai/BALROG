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
        "text_inventory": nle_language.text_inventory(inv_strs, inv_letters).decode("latin-1"),
        "text_cursor": nle_language.text_cursor(glyphs, blstats, tty_cursor).decode("latin-1"),
    }


def render_text(nle_obsv):
    long_term_observations = [
        ("text_message", "message"),
        ("text_glyphs", "language observation"),
        ("text_cursor", "cursor"),
    ]
    text_obsv = nle_obsv_to_language(nle_obsv)

    short_term_observations = [
        ("text_blstats", "statistics"),
        ("text_inventory", "inventory"),
    ]

    long_term_context = "\n".join([f"{name}:\n{text_obsv[key]}\n" for key, name in long_term_observations])
    short_term_context = "\n".join([f"{name}:\n{text_obsv[key]}\n" for key, name in short_term_observations])

    return {
        "long_term_context": long_term_context,
        "short_term_context": short_term_context,
    }


def render_hybrid(nle_obsv):
    ascii_map = ascii_render(nle_obsv["tty_chars"])
    cursor = nle_obsv["tty_cursor"]
    cursor = f"(x={cursor[1]}, y={cursor[0]})"
    ascii_map = "\n".join(ascii_map.split("\n")[1:])  # remove first line

    text_obsv = nle_obsv_to_language(nle_obsv)
    text_obsv["map"] = ascii_map
    text_obsv["text_cursor"] = text_obsv["text_cursor"] + "\n" + cursor

    long_term_observations = [
        ("text_message", "message"),
        ("text_glyphs", "language observation"),
        ("text_cursor", "cursor"),
        ("map", "map"),
    ]
    short_term_observation = [
        ("text_inventory", "inventory"),
    ]

    long_term_context = "\n".join([f"{name}:\n{text_obsv[key]}\n" for key, name in long_term_observations])
    short_term_context = "\n".join([f"{name}:\n{text_obsv[key]}\n" for key, name in short_term_observation])

    return {
        "long_term_context": long_term_context,
        "short_term_context": short_term_context,
    }
