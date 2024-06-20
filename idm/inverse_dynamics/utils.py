import re
import difflib


DIRECTIONS = {
    "north": (0, -1),
    "northeast": (1, -1),
    "east": (1, 0),
    "southeast": (1, 1),
    "south": (0, 1),
    "southwest": (-1, 1),
    "west": (-1, 0),
    "northwest": (-1, -1),
}

ATTACKS = (
    "You kill",
    "You hit",
    "You miss",
    "You destroy",
    "Really attack",
    "strikes",
    "stuns",
    "scares",
    "sears",
    "stuns",
    "burns",
    "cancels",
)

CALL_MESSAGES = (
    "Call a",
    "Call an",
    "What do you want to name this",
    "What do you want to name these",
)

EATING = (
    "eating",
    "delicious",
    "tastes",
    "You finish eating",
    "Rotten",
    "opens like magic",
    "You resume your meal",
    "that must have been poisonous",
)

WHAT_DO_YOU_WANT_MESSAGES = {
    "What do you want to eat": "eat",
    "What do you want to wear": "wear",
    "What do you want to drop": "drop",
    "What do you want to take off": "takeoff",
    "What do you want to throw": "throw",
    "What do you want to put on": "puton",
    "What do you want to wield": "wield",
    "What do you want to use or apply": "apply",
    "What do you want to remove": "remove",
    "What do you want to dip": "dip",
    "What do you want to ready": "quiver",
    "What do you want to write with": "engrave",
    "What do you want to drink": "quaff",
    "What do you want to rub": "rub",
    "What do you want to zap": "zap",
    "What do you want to name": "name",
    "What do you want to look at": "look",
    "What do you want to sacrifice": "offer",
}

INVENTORY_LOOK_SYMBOLS = {
    "What do you want to eat": "?",
    "What do you want to wear": "?",
    "What do you want to drop": "*",
    "What do you want to take off": "?",
    "What do you want to throw": "*",
    "What do you want to put on": "?",
    "What do you want to wield": "?",
    "What do you want to use or apply": "?",
    "What do you want to remove": "?",
    "What do you want to dip": "*",
    "What do you want to ready": "?",
    "What do you want to write with": "*",
    "What do you want to drink": "?",
    "What do you want to rub": "?",
    "What do you want to zap": "?",
    "What do you want to name": "*",
    "What do you want to look at": "*",
    "What do you want to sacrifice": "?",
}

MENU_INTERACTION_MSG = ("What would you like to drop", "Pick up what?")

ANIMATION_FILLERS = (
    "You see here",
    "You hear",
    "You smell",
    "It is not so easy to",
    "drops",
    "picks up",
    "misses",
    "hits",
    "Welcome to",
    "Welcome again",
    "Your sacrifice is consumed",
    "You feel that",
)

NOT_FILLERS = ("The door closes", "The door resists")

REMOVE_MESSAGES = (
    "# loot",
    "# name",
    "# pray",
    "# chat",
    "# adjust",
    "# dip",
    "# pray",
    "# offer",
    "# jump",
    "# rub",
    "# untrap",
    "# ride",
    "# enhance",
)

SYMBOLS = {
    "fountain": "{",
    "altar": "_",
    "staircase up": "<",
    "staircase down": ">",
}


def ascii_render(chars):
    rows, cols = chars.shape
    result = ""
    for i in range(rows):
        for j in range(cols):
            entry = chr(chars[i, j])
            result += entry
        result += "\n"
    return result


def color_render(chars):
    rows, cols = chars.shape
    result = ""
    for i in range(rows):
        for j in range(cols):
            entry = str(chars[i, j])
            result += entry
        result += "\n"
    return result


def clean_string(input_string):
    # Remove leading "a " or "an " or any leading numbers
    cleaned_string = re.sub(
        r"^(a |an |\d+)", "", input_string, flags=re.IGNORECASE
    ).strip()

    # Remove any ending "s" character from the last word
    cleaned_string = re.sub(r"\b(\w+)s\b$", r"\1", cleaned_string, flags=re.IGNORECASE)

    return cleaned_string.strip()


def get_timestep(stats):
    """stats is of the form:
    Dlvl:1 $:9 HP:56(56) Pw:21(21) AC:1 Xp:6/429 T:9680 Weak
    Where T:9680 is the timestep
    """
    if "T:" not in stats:
        return ""
    return stats.split("T:")[1].split(" ")[0]


def get_hp(stats):
    """stats is of the form:
    Dlvl:1 $:9 HP:56(56) Pw:21(21) AC:1 Xp:6/429 T:9680 Weak
    Where HP:56(56) is the health. We want to have 56 as output
    """
    if "HP:" not in stats:
        return ""
    return stats.split("HP:")[1].split("(")[0]


def get_dlvl(stats):
    """stats is of the form:
    Dlvl:1 $:9 HP:56(56) Pw:21(21) AC:1 Xp:6/429 T:9680 Weak
    Where Dlvl:1 is the dungeon level. We want to have 1 as output
    """
    if "Dlvl:" not in stats:
        return ""
    return stats.split("Dlvl:")[1].split(" ")[0]


def obs_to_message(obs):
    return "".join([chr(char) for char in obs[0]])


def obs_to_stats(obs):
    return "".join([chr(char) for char in obs[23]])


def check_cursor(map, cursor, char):
    # Check if the cursor is on the character
    map_lines = map.split("\n")[1:22]
    x, y = cursor[0], cursor[1] - 1
    # First check if the x,y coordinates are within the map
    if 0 <= x < 80 and 0 <= y < 21:
        return map_lines[y][x] == char
    else:
        return False


def attack_in_message(attacks, message):
    for attack in attacks:
        if attack in message:
            return True
    return False


def menu_interaction(obs_a, obs_b, ascii=False):
    # Could be further improved to select only some submenus
    changes = get_changes(obs_a, obs_b, line_0_offset=True, ascii=ascii)

    multiple_changes = False
    if len(changes.split("\n")) > 1:
        multiple_changes = True

    minus_index = changes.find("-")
    plus_index = changes.find("+")

    if minus_index != -1 and (plus_index == -1 or minus_index < plus_index):
        item = changes.split("-")[0].strip()
        if multiple_changes:
            return "-"
        return item
    elif plus_index != -1 and (minus_index == -1 or plus_index < minus_index):
        item = changes.split("+")[0].strip()
        if multiple_changes:
            return "."
        return item
    else:
        return "unknown selection"


def get_changes(obs_a, obs_b, line_0_offset=False, ascii=False):
    if ascii:
        map_a = obs_a
        map_b = obs_b
    else:
        map_a = ascii_render(obs_a)
        map_b = ascii_render(obs_b)

    changes = []
    diff = difflib.ndiff(map_a.splitlines(), map_b.splitlines())
    for line in diff:
        if line.startswith("+ "):
            changes.append(line[2:])

    start_idx = 0
    if line_0_offset:
        # Get the index of the first character that is not a letter in line 0
        idx = 0
        if ascii:
            message_b = map_b.split("\n")[0]
        else:
            message_b = obs_to_message(obs_b)
        for idx, char in enumerate(message_b):
            start_idx = idx
            if char.isalpha():
                break
    else:
        idx = 0
        # Check if changes is empty
        if not changes:
            print("No changes found")
            return ""
        for char in changes[0]:
            if not char.isalpha():
                idx += 1
                start_idx = idx
            elif char.isalpha():
                break

    # BUG: nboundLocalError: local variable 'start_idx' referenced before assignment
    # Now fixed, but need to check when this happens to make sure it labels correctly
    idx = start_idx
    changes = [line[idx:] for line in changes]
    changes = "\n".join(changes)
    return changes


def find_single_option(message):
    return re.search(r"\[\s*([a-zA-Z])\s*or", message)


def get_menu_message(obs, ascii=False):
    if ascii:
        map = obs
        message = map.split("\n")[0]
    else:
        map = ascii_render(obs)
        message = obs_to_message(obs)

    idx = 0
    for idx, char in enumerate(message):
        start_idx = idx
        if char.isalpha():
            break

    line_idx = 0
    menu = ""
    while not find_n_of_m_end(map.split("\n")[line_idx]):
        menu += map.split("\n")[line_idx][start_idx:] + "\n"
        line_idx += 1

    return menu


def find_n_of_m_end(string):
    pattern = re.compile(r"\(\d+ of \d+\)|\(end\)", re.IGNORECASE)
    if pattern.search(string):
        return True
    else:
        return False


def find_n_of_m(string):
    pattern = re.compile(r"\((\d+\+? ?of \d+)\)", re.IGNORECASE)
    if pattern.search(string):
        return True
    else:
        return False


def find_n_of_m_end_patterns(string):
    pattern = re.compile(r"\(\d+ of \d+\)|\(end\)", re.IGNORECASE)
    matches = pattern.findall(string)
    if not matches:
        return ""
    return matches[0]


def find_n_of_m_patterns(message):
    """
    Finds and returns the (n of m) patterns in the given message.

    Args:
        message (str): The message to search for patterns.

    Returns:
        str: A str with the pattern instance found
    """
    pattern = re.compile(r"\((\d+\+? ?of \d+)\)", re.IGNORECASE)
    matches = pattern.findall(message)

    if not matches:
        return ""
    return matches[0]


def clean_n_m_end(message):
    return "\n".join(
        line
        for line in message.split("\n")
        if not re.search(r"\(\d+ of \d+\)|\(end\)", line.strip())
    )
