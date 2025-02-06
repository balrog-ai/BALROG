import enum

from .auto_more import AutoMore
from .base import NLELanguageWrapper


class Role(enum.Enum):
    ARCHEOLOGIST = "arc"
    BARBARIAN = "bar"
    CAVEMAN = "cav"
    HEALER = "hea"
    KNIGHT = "kni"
    MONK = "mon"
    PRIEST = "pri"
    RANGER = "ran"
    ROGUE = "rog"
    SAMURAI = "sam"
    TOURIST = "tou"
    VALKYRIE = "val"
    WIZARD = "wiz"


ACTIONS = {
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
    "overview": "display an overview of the dungeon",
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


def get_instruction_prompt(task=None):
    action_strings = ",\n".join(f"{action}: {description}" for action, description in ACTIONS.items())
    instruction_prompt = f"""
You are an agent playing NetHack. The following are the possible actions you can take in the game, followed by a short description of each action:

{action_strings}.

Tips:
- When the message asks for a completion, such as: "What do you want to eat? [d or ?*]", you should respond with a single character corresponding to the item you want to eat/use.
    - For example, "What do you want to eat? [dgh or ?*]" -> Possible answers are "d", "g", or "h" to eat the associated food.
- When the message asks for a direction, such as: "In what direction?" you should respond with a direction.
- When the message has --More-- at the end, your next action should be "more" to see the rest of the message.
- Explore the environment to find the stairs down to the next level.
- Always carefully read the last message to understand the current state of the game and decide your next action accordingly.
- If you keep moving in the same direction, you will eventually hit a wall and stop moving. Your message might be: "It's solid stone", or "It's a wall". Change your action to move in another direction to continue exploring the environment.
- Read the language observation carefully and look at ascii map or image observation provided to decide the next action to take and where to move next.
- You can attack monsters by moving into them.

In a moment I will present a history of actions and observations from the game.
Your goal is to get as far as possible in the game.

PLAY!
""".strip()

    return instruction_prompt
