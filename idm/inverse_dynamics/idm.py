import numpy as np
import re
from idm.inverse_dynamics.monsters import MONSTER_DICT
from idm.inverse_dynamics.utils import *
from idm.inverse_dynamics.inventory import *
import random


class IDM:
    # TODO: FINHISH MENU INTERACTIONS! THEY ARE IMPORTANT! (almost all done)
    # TODO: Finish dipping objects (almost done)
    # TODO: Really attack [yn] and such, if we want to say no, we should instead say esc (not n) (7175)
    # TODO: pick an object (look objects, move the cursor and not the player. / and ; commands)
    # TODO: apply magic marger, asks: "What do you want to write on?""
    # particularly we could merge the menu interaction of the bag of holding staff with this)
    # TODO: THROW ITEMS (Mjolrnir, arrows, daggers, etc) (Not always working)
    # TODO: Wear and takeoff. Wear armor it's impossible to know what was worn, so we can choose randomly
    # while when taking off we know what is being take off from a message
    # TODO: Remove all the (e+d) from the inventory messages, otherwise they are detected as a menu interaction
    # (which they are, but we cannot see them). They are detected as ACTION:(e
    # TODO: We have some more edge cases in (5817) in "Put in what?" When it changes page it gets confused
    # TODO: ADD ANNOTATE! WE CAN ATTOTATE DUNGEON LEVELS. (# annotate) Useful to teleport there later on I guess
    # TODO: finish what do you want to put on (not working yet, with rings/amulets we know what is being put on)
    # TODO: Dagger/weapon/arrow throwing needs to be improved! (12222)
    # TODO: Add # overview (to see the map levels)
    # TODO: Add movement actions when you fall into a pit! (17116)
    # TODO: Teleport actions
    # TODO: There is a difference between (?) and (*) when menu interacting! (*) is ALL items
    # while (?) is all the items of a certain type we are interacting with (armor for example)
    # TODO: In some cases though, we may want to use (*), for example to throw food rations to new pets
    # TODO: SOME GAMES USE SYMBOLS OTHER THAN THE DEFAULT @ FOR THE PLAYER... IGNORE THOSE GAMES
    # TODO: ADD USELESS FRAME REMOVAL (like all the "# " messages, some of the unknown actions and so on)
    # TODO: Movements on corpses are not always right! Depending on the ttyrec (290 of Machinespr)
    # THIS IS TRUE ONLY WHEN MOVING ON MULTIPLE OBJECTS! (like corpses with other items)
    # TODO: WEAR OBJECTS
    # TODO: TOO MANY SEARCH actions

    def __init__(self):
        self.last_direction = "N"
        self.inventory = Inventory()

    def process_game(self, data_file, output_file="labeled_game.txt"):
        data = np.load(data_file)
        print(type(data))
        self.tty_chars = data["tty_chars"]
        self.last_action = ""
        self.tty_cursor = data["tty_cursor"]
        self.timestep = 0
        self.player_position = self.tty_cursor[self.timestep]

        with open(f"{output_file}", "a") as f:
            listening = False

            for i in range(self.tty_chars.shape[0] - 1):

                ########### TIMESTEP ############

                # start_time = 1118
                # end_time = 1121

                # stats = "".join([chr(c) for c in self.tty_chars[i][23]])
                # timestep = get_timestep(stats) if "T:" in stats else ""

                # if timestep != "" and int(timestep) == start_time:
                #     listening = True

                # if not listening:
                #     continue

                # if listening and timestep != "" and int(timestep) > end_time:
                #     listening = False

                # print(timestep)
                ########### TIMESTEP ############

                render_a = ascii_render(self.tty_chars[i])
                cursor_a = np.array2string(self.tty_cursor[i])

                movement = self.detect_action(self.tty_chars, self.tty_cursor, i)
                # f.write(self.inventory.get_inventory())
                # f.write("\n")
                f.write(render_a)
                f.write(cursor_a)
                f.write("ACTION:")
                f.write(movement)
                f.write("\n")
                f.write("#" * 80)
                f.write("\n")
        return output_file

    def label_game(self, data_file, dlvl_cutoff=15):
        data = np.load(data_file)
        print(type(data))
        self.tty_chars = data["tty_chars"]
        self.last_action = ""
        self.tty_cursor = data["tty_cursor"]
        self.timestep = 0
        self.player_position = self.tty_cursor[self.timestep]

        actions = []
        inventory = []
        messages = []

        print(len(self.tty_chars))
        for i in range(self.tty_chars.shape[0] - 2):
            actions.append(self.detect_action(self.tty_chars, self.tty_cursor, i))
            inventory.append(self.inventory.get_inventory())
            messages.append(obs_to_message(self.tty_chars[i]))

            dlvl = get_dlvl(obs_to_stats(self.tty_chars[i]))
            if dlvl and int(dlvl) > dlvl_cutoff:
                break

        for message in messages:
            if "You are a" in message:
                summary = message.split("You are a")[1].split(".")[0].strip()
                break
        print(f"Labeled game {data_file} with {len(actions)} actions")
        return actions, inventory, summary

    def read_scroll(self, obs_a, obs_b, obs_c, obs_d):
        # Decent enough implementation. Could still be improved
        message_a = obs_to_message(obs_a)
        message_b = obs_to_message(obs_b)
        message_c = obs_to_message(obs_c)
        message_d = obs_to_message(obs_d)
        if self.last_action == "read" and (
            "As you read" in message_c or "As you read" in message_b
        ):
            if "Scrolls" in message_a:
                scrolls = get_menu_message(obs_a)
                scrolls = scrolls.split("\n", 1)[1]
                self.inventory.update_type("Scrolls", scrolls)
            elif find_single_option(message_a):  # Find single
                scroll = find_single_option(message_a).group(1)
                return scroll
            else:
                scrolls = self.inventory.get_inventory_type("Scrolls")

            item = ""
            if (
                "identify" in message_c
                or "identify" in message_d
                or " - " in message_c  # Lucky scroll the identifies everything!
                or " - " in message_d
            ):
                self.last_action = "menu interaction"
                item = self.inventory.get_inventory_item("identify")

            if "glows silver" in message_c or "glows silver" in message_d:
                item = self.inventory.get_inventory_item("enchant armor")

            if "blue" in message_c or "blue" in message_d:
                item = self.inventory.get_inventory_item("enchant weapon")

            if (
                "What do you want to charge?" in message_c
                or "What do you want to charge?" in message_d
            ):
                self.last_action = "menu interaction"
                # Need to complete the menu interaction, it willl decide what to charge
                item = self.inventory.get_inventory_item("scroll of charging")

            if "A map coalesces" in message_c or "A map coalesces" in message_d:
                item = self.inventory.get_inventory_item("magic mapping")

            if (
                "do you wish to genocide?" in message_c
                or "do you wish to to genocide?" in message_d
            ):
                self.last_action = "menu interaction"
                item = self.inventory.get_inventory_item("scroll of genocide")

            # TODO: Scroll of teleportation

            if len(item) == 1 and item.isalpha():
                return item
            else:
                scrolls = scrolls.splitlines()
                len_scrolls = max(len(scrolls) - 1, 0)
                if len_scrolls == 0:
                    return " "
                number = np.random.randint(0, len_scrolls)
                random_item = scrolls[number]
                # If the random item is longer than 0 return the first character
                if len(random_item) > 0:
                    return random_item[0]
                else:
                    print("Random item is empty")
                    return None
        return None

    def detect_action(self, obs, cursor, timestep):
        """
        Detects the action performed by the player given two observations and their cursors.

        Args:
            obs: The list of observations.
            cursor: The list of cursor positions
            colors: The list of colors
            timestep: The timestep of the observation

        Returns:
            A string representing the action performed by the player.
        """
        obs_a = obs[timestep]
        obs_b = obs[timestep + 1]

        # check if it's not out of bounds
        if timestep + 2 >= len(obs):
            obs_c = obs_b
        else:
            obs_c = obs[timestep + 2]

        if timestep + 3 >= len(obs):
            obs_d = obs_c
        else:
            obs_d = obs[timestep + 3]

        cursor_a = cursor[timestep]
        cursor_b = cursor[timestep + 1]

        x_movement = cursor_b[0] - cursor_a[0]
        y_movement = cursor_b[1] - cursor_a[1]

        message_a = obs_to_message(obs_a)
        message_b = obs_to_message(obs_b)

        ########### ATTACKS ############
        if (
            attack_in_message(ATTACKS, message_b)
            and not "digging" in message_b
            and not "Really attack" in message_a
        ):
            return self.find_attack(obs_a, obs_b, cursor_a)

        ########### DIGGING ############
        if "In what direction do you want to dig?" in message_a:
            action = self.detect_digging_direction(obs_a, obs_b, cursor_a)
            if action:
                return action
            else:
                return "unknown digging"
        if "You start digging" in message_b:
            action = self.detect_digging_direction(obs_a, obs_b, cursor_a)
            if action:
                return action
            else:
                return "unknown digging"

        ########### MESSAGES ###########

        if y_movement == 0 and cursor_a[1] == 0:
            message = self.detect_message_to_message(obs, cursor, timestep)
            if message:
                return message
            else:
                return "message to message"

        ########### TRIGGERING A MESSAGE ###########

        if y_movement != 0 and cursor_b[1] == 0:
            # THIS COULD BE A MOVEMENT/ACTION TRIGGERING A MESSAGE...
            action = self.detect_action_triggering_message(obs, timestep)
            if action:
                return action
            else:
                return "action triggering message"

        ########### ACTING ON A MESSAGE ###########
        if y_movement != 0 and cursor_a[1] == 0:
            return self.detect_acting_on_message(obs, cursor, timestep)

        ########### DOOR OPENING ###########
        if (
            "The door opens" in message_b
            or "The door is locked" in message_b
            or "That door is closed" in message_b
            or "In what direction?"
            in message_b  # Not sure whether this is good checking in message_b here
            or "You break open the lock" in message_b
        ):
            # print(message_b)
            # TODO: The kicking the door down is not well implemented yet. There might be more situations in which
            # we want to kick something else than a door! Examples are locked boxes/chests, but also monsters like ghosts!

            # Check door first
            action = self.find_door(obs_a, cursor_a)
            if action:
                return action

            # Otherwise check if we find an adjacent chest/box
            action = self.find_adjacent(obs_a, cursor_a, "(")
            if action:
                return action

            return "open door or chest"

        ########### OTHER ############

        # Name
        if "What do you want to name" in message_b:
            self.last_action = "call"
            return "name"

        # Wielding
        if (
            "What do you want to wield" in message_a
            or "What do you want to ready?" in message_a
        ):
            if " - " in message_b:
                return message_b.split(" - ")[0]
            elif "You are empty handed" in message_b:
                return "-"

        if "In what direction?" in message_a:
            if "You dig a hole through the floor" in message_b:
                return ">"
            elif "You dig a pit in the floor" in message_b:
                return "<"

        if "--More" in message_a:
            return "more"

        if "You have a little trouble lifting" in message_b:
            if not "Continue" in message_b:
                item = (
                    message_b.split("lifting")[1].split(".")[0].strip().replace(".", "")
                )
                self.inventory.add_item(item)
            return "pickup"

        if "Pick up what?" in message_b:
            # TODO: We have the MENU HERE
            if "Pick up what" in message_a:
                return menu_interaction(obs_a, obs_b)
            return "pickup"

        if "For you" in message_b and "zorkmid" in message_b:
            return "pickup"

        if "Pay? [yn]" in message_a and "You bought" in message_b:
            return "y"

        # Composit action. Pick up often times has a skip frame (the action is not seen on screen)
        if "Pick up what?" in message_a and "" in message_b.strip():
            self.last_action = "enter"
            return "more"
        if self.last_action == "enter" and " - " in message_b:
            self.inventory.add_item(message_b.split(".")[0].strip())
            return " "

        if "little trouble lifting" in message_a and "Continue?" in message_a:
            # Check whether item was picked up or not
            self.check_pickup(timestep)

        if "unlock it?" in message_a.lower() and "You succeed in" in message_b:
            return "y"

        if "Never mind" in message_b:
            return "esc"

        if (
            "What do you want to wear" in message_a
            and "You are now wearing" in message_b
        ):
            # self.inventory.print_inventory()
            item = message_b.split("You are now wearing ")[1].split(".")[0].strip()
            return self.inventory.get_inventory_item(item)

        if "into the fountain? [yn] (n)" in message_a:
            return "y"

        if "Sell it?" in message_a and "You sold" in message_b:
            return "y"

        ########### WHAT IS ############
        if "What do you want to look at" in message_b:
            return "whatis"

        if "What do you want to look at" in message_a:
            if "Please move the cursor to an unknown object" in message_b:
                self.last_action = "whatis"
                return "/"
            else:
                return "esc"

        if self.last_action == "whatis" and not message_b.strip():
            self.last_action = ""
            return "esc"

        ########### READ ############
        action = self.read_scroll(obs_a, obs_b, obs_c, obs_d)
        if action:
            return action

        ########### TRAVEL ############
        if "Where do you want to travel to" in message_b:
            return "travel"
        if "Where do you want to travel to" in message_a and any(
            symbol_name.lower() in message_b.lower() for symbol_name in SYMBOLS
        ):
            for symbol_name, symbol in SYMBOLS.items():
                if symbol_name.lower() in message_b.lower():
                    self.last_action = "travel"
                    return symbol

        if self.last_action == "travel" and not message_b.strip():
            self.last_action = ""
            return "."
        # Otherwise if messaged_b is empty, return esc
        if "Where do you want to travel to" in message_a and not message_b.strip():
            self.last_action = ""
            return "esc"

        ############ INVENTORY ############
        if (
            "Coins" in message_b
            or "Weapons" in message_b
            or "Armor" in message_b
            or "Amulets" in message_b
        ):
            if find_n_of_m_end(ascii_render(obs_c)):
                self.inventory.update_inventory_from_maps(obs_a, obs_b, delete_old=True)
            else:
                self.inventory.update_inventory_from_maps(obs_a, obs_b)
            self.last_action = "inventory"
            return "inventory"
        if self.last_action == "inventory" and find_n_of_m_end(ascii_render(obs_b)):
            self.inventory.update_inventory_from_maps(obs_a, obs_b)
            return ">"
        elif self.last_action == "inventory":
            self.last_action = ""

        ##################################

        if "You drop " in message_b and self.inventory:
            return self.inventory.get_inventory_item(
                message_b.split("You drop ")[1].strip()
            )

        if "Dlvl" in (obs_to_stats(obs_a)) and "Dlvl" in (obs_to_stats(obs_b)):
            if get_dlvl(obs_to_stats(obs_a)) != get_dlvl(obs_to_stats(obs_b)):
                if int(get_dlvl(obs_to_stats(obs_a))) > int(
                    get_dlvl(obs_to_stats(obs_b))
                ):
                    return "down"
                else:
                    return "up"

        ########### BAG INTERACTION ############
        if (
            "Do what with your bag" in message_a
            or "Do what with the bag" in message_a
            or "Do what with the chest" in message_a
        ):
            if "Put in what type" in message_b:
                return "i"
            elif "Take out what type" in message_b:
                return "o"

        ########### MENU INTERACTION ############
        map_a = ascii_render(obs_a)
        map_b = ascii_render(obs_b)

        if "--More--" in map_a:
            return "more"

        if (
            "Drop what type of items" in message_a
            and "What would you like to drop?" in message_b
        ):
            return "more"

        if (
            "Take out what type" in message_a and "Take out what type" in message_b
        ):  # First menu
            return menu_interaction(map_a, map_b, ascii=True)
        if (
            "Take out what type" in message_a and "Take out what?" in message_b
        ):  # Move to next menu
            self.last_action = "menu interaction"
            return "more"

        if "Put in what type" in message_a and "Put in what type" in message_b:
            return menu_interaction(map_a, map_b, ascii=True)
        if "Put in what type" in message_a and "Put in what?" in message_b:
            self.last_action = "menu interaction"
            return "more"

        if self.last_action == "menu interaction" and (  # second menu decision
            (find_n_of_m_patterns(map_a) == find_n_of_m_patterns(map_b))
            or (find_n_of_m_end(map_a) and find_n_of_m_end(map_b))
        ):

            if find_n_of_m_end(map_a) and not find_n_of_m_end(map_b):
                if "Never mind" in message_b:
                    self.last_action = ""
                    return "esc"
                else:
                    self.last_action = ""
                    return "more"
            else:
                return menu_interaction(map_a, map_b, ascii=True)

        if "Pick a skill to advance" in message_a:
            return "a"

        # Keep navigating the menu
        # When we change page of the bag of holding we cannot compare to the previous map
        if find_n_of_m(map_a) and find_n_of_m(map_b):
            pattern_a = find_n_of_m_patterns(map_a)
            pattern_b = find_n_of_m_patterns(map_b)
            if pattern_a != pattern_b:
                if int(pattern_b[0]) > int(pattern_a[0]):
                    return ">"
                elif int(pattern_b[0]) < int(pattern_a[0]):
                    return "<"
                elif pattern_a[1] == " " and pattern_b[1] == "+":
                    # TODO: ASK TIM! What is it doing here?
                    return ">"
                else:
                    return "more"

        ## CALL/NAME ##
        if find_n_of_m(map_a) and not find_n_of_m(map_b):
            message_c = obs_to_message(obs_c)
            if self.last_action == "call" and (
                any(
                    call_msg in message_b or call_msg in message_c
                    for call_msg in CALL_MESSAGES
                )
            ):
                self.last_action = ""
                # self.inventory.print_inventory()
                if "Call a" in message_b:
                    item = re.findall(r"Call a[n]?\s(.*?):", message_b)[0]
                elif "Call a" in message_c:
                    item = re.findall(r"Call a[n]?\s(.*?):", message_c)[0]
                elif "What do you want to name th" in message_b:
                    item = re.findall(
                        r"What do you want to name (this|these) (.*?)\?", message_b
                    )[0][1]
                    # Got an error on item = re.findall(...)
                    # IndexError: list index out of range
                elif "What do you want to name th" in message_c:
                    item = re.findall(
                        r"What do you want to name (this|these) (.*?)\?", message_c
                    )[0][1]
                return self.inventory.get_inventory_item(item)

            pattern_a = find_n_of_m_patterns(map_a)
            pattern_b = find_n_of_m_patterns(map_b)
            # Check that pattern_a is not of the type (n of n)
            self.last_action = ""
            if pattern_a[0] != pattern_a[3] and pattern_a != pattern_b:
                return "more"

        ########### REMOVE ############
        if (
            "What do you want to put on?" in message_a
            or "What do you want to remove?" in message_a
        ) and "towel" in message_b:
            return self.inventory.get_inventory_item("towel")

        ########### CAST SPELL ############
        # TODO: Plenty of spells to cast, we need to implement the menu interaction...
        # Not very easy to know which spell was cast actually...
        if "Choose which spell to cast" in message_b:
            self.last_action = "cast"
            return "cast"

        ########### DROP ############
        if "Drop what type of items?" in message_b:
            self.last_action = "menu interaction"
            return "droptype"

        ########### LOOT ############
        if "locked" in message_b and "loot it?" in message_a:
            return "y"

        ########### MOVEMENTS ############
        movement = self.detect_movement(obs_a, obs_b, cursor_a, cursor_b)
        if movement:
            return movement

        ########### ANIMATIONS/FILLERS ############
        if any(filler in message_b for filler in ANIMATION_FILLERS):
            return " "

        ########### SEARCH ############
        if (
            get_timestep(obs_to_stats(obs_a)) != ""
            and get_timestep(obs_to_stats(obs_b)) != ""
        ):
            if int(get_timestep(obs_to_stats(obs_a))) + 1 == int(
                get_timestep(obs_to_stats(obs_b))
            ):
                if "The door resists" in message_b:
                    action = self.find_door(obs_b, cursor_b)
                    return action if action else self.last_direction
                elif message_a.split():
                    return " "
                return "search"

    def check_pickup(self, timestep):
        obs_c = self.tty_chars[timestep + 2]

        message_c = obs_to_message(obs_c)

        if (
            "You have a little trouble lifting" in message_c
            and not "Continue" in message_c
        ):
            item = message_c.split("lifting")[1].split(".")[0].strip().replace(".", "")
            self.inventory.add_item(item)
            return "y"
        else:
            return "n"

    def find_attack(self, obs_a, obs_b, cursor_a):
        """
        Detects the direction of the attack performed by the player given two
        observations and their cursors.

        Args:
            obs_a: The first observation.
            obs_b: The second observation.
            cursor_a: The cursor position of the first observation.

        Returns:
            A string representing the direction of the attack performed by the player.
        """

        # TODO: There are situations in which the "it" monster (invisible I) could be detected just
        # because "it" is in the message (in any part of it). The monster "it" is an edge case

        map_a = ascii_render(obs_a)
        map_lines = map_a.split("\n")[1:22]

        if cursor_a[1] == 0:
            x, y = self.player_position[0], self.player_position[1] - 1
        else:
            x, y = cursor_a[0], cursor_a[1] - 1

        message_b = obs_to_message(obs_b)

        message_b = [
            message
            for message in message_b.split("  ")
            if attack_in_message(ATTACKS, message)
        ][0]

        for monster in MONSTER_DICT.keys():
            if monster in message_b:
                symbol = MONSTER_DICT[monster]["ascii"]
                for direction, (dx, dy) in DIRECTIONS.items():
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 80 and 0 <= ny < 21:
                        if map_lines[ny][nx] == symbol:
                            self.last_direction = direction
                            return direction

        return self.last_direction

    def detect_message_to_message(self, obs, cursor, timestep):

        obs_a = obs[timestep]
        obs_b = obs[timestep + 1]

        cursor_a = cursor[timestep]
        cursor_b = cursor[timestep + 1]

        if timestep + 2 >= len(obs):
            obs_c = obs_b
        else:
            obs_c = obs[timestep + 2]

        # obs_d
        if timestep + 3 >= len(obs):
            obs_d = obs_c
        else:
            obs_d = obs[timestep + 3]

        message_a = obs_to_message(obs_a)
        message_b = obs_to_message(obs_b)

        if "In what direction?" in message_a and "unlock it?" in message_b.lower():
            action = self.find_door(obs_a, self.player_position)
            if action:
                return action
            action = self.find_adjacent(obs_a, self.player_position, "(")
            if action:
                return action
            return "open door or chest"

        if "What do you want to dip" in message_b:
            self.last_action = "dip"
            return "dip"

        if "# untrap" in message_b:
            self.last_action = "untrap"
            return "untrap"

        if "# enhance" in message_b:
            return "enhance"

        if "# rub" in message_b or "# rub" in message_a:
            self.last_action = "rub"
            return "rub"

        if "# chat" in message_a:
            # To be completed!
            # Will ask in what direction
            return "chat"

        if "eat it?" in message_a and any(
            eat_message in message_b for eat_message in EATING
        ):
            return "y"

        if "Never mind" in message_b:
            return "esc"

        if self.last_action == "read" and "As you read" in message_b:
            return self.read_scroll(obs_a, obs_b, obs_c, obs_d)

        if "--More--" in message_a:
            return "more"

        if "# name" in message_b:
            return "name"

        if "Are you sure you want to pray?" in message_b:
            return "pray"

        if "What do you want to sacrifice?" in message_b:
            return "offer"

        if "# loot" in message_a and "loot it?" in message_b:
            return "loot"

        if "You write in the dust with your fingertip" in message_b:
            return "-"

        # Checking for patterns like: Y - a blessed white potion. And taking the first letter
        if re.findall(r"\b(\w) - [^-]+", message_b):
            return re.findall(r"\b(\w) - [^-]+", message_b)[0]

        if (
            "What do you want to name" in message_b
            or ("What do you want to write" in message_b and "[" not in message_b)
            or "What do you want to add to the writing" in message_b
            or "Call a" in message_b
            or "For what do you wish?" in message_b
            or "What type of scroll do you want to write" in message_b
            or "do you wish to genocide?" in message_b
            or ("Really attack" in message_a and "Really attack" in message_b)
        ):
            self.last_action = "writing"

            if (cursor_a[0] - 1) == cursor_b[0]:
                return "delete"
            elif cursor_a[0] > 79 or cursor_a[0] < 0:
                return ""
            return message_b[cursor_a[0]]

        if "Wiped out" in message_b:
            self.last_action = ""
            return "more"

        if "Do you want to add to the current engraving?" in message_b:
            self.last_action = "engrave"
            return "engrave"

        if "In what direction?" in message_a and (
            "lock it?" in message_b.lower() or "unlock it?" in message_b.lower()
        ):
            action = self.find_door(obs_a, cursor_a)
            if action:
                return action
            action = self.find_adjacent(obs_a, cursor_a, "(")
            if action:
                return action
            return " "

        if timestep + 2 >= len(obs):
            obs_c = obs_b
        else:
            obs_c = obs[timestep + 2]
        message_c = obs_to_message(obs_c)

        if (
            "What do you want to use or apply" in message_a
            and "Unlock it?" in message_c
        ):
            # The lockpick might have a similar message to the key too!
            return self.inventory.get_inventory_item("key")

        if "Do you want to add to the current engraving?" in message_a:
            if "You add" in message_b:
                return "y"
            elif "You wipe" in message_b:
                return "n"
            else:
                return " "

        if "loot it? [ynq]" in message_a:
            if "open" in message_b or "locked" in message_b:
                return "y"
            else:
                return "n"

        return None

    def find_adjacent(self, obs_a, cursor_a, symbol):
        map_a = ascii_render(obs_a)
        map_lines = map_a.split("\n")[1:22]

        if cursor_a[1] == 0:
            x, y = self.player_position[0], self.player_position[1] - 1
        else:
            x, y = cursor_a[0], cursor_a[1] - 1

        for direction, (dx, dy) in DIRECTIONS.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < 80 and 0 <= ny < 21:
                if map_lines[ny][nx] == symbol:
                    return direction
        return None

    def find_adjacent_change(self, obs_a, obs_b, cursor_a, symbol):
        # Similar to the above, but find if the symbol we are looking for just appeared
        # in obs_b, and was not there in obs_a
        map_a = ascii_render(obs_a)
        map_b = ascii_render(obs_b)
        map_lines_a = map_a.split("\n")[1:22]
        map_lines_b = map_b.split("\n")[1:22]

        if cursor_a[1] == 0:
            x, y = self.player_position[0], self.player_position[1] - 1
        else:
            x, y = cursor_a[0], cursor_a[1] - 1

        directions = list(DIRECTIONS.items())
        random.shuffle(directions)
        for direction, (dx, dy) in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 80 and 0 <= ny < 21:
                if map_lines_a[ny][nx] != symbol and map_lines_b[ny][nx] == symbol:
                    return direction
        return None

    def find_door(self, obs_a, cursor_a):
        # Check adjacent cells for a door '+'
        direction = self.find_adjacent(obs_a, cursor_a, "+")
        if direction is not None:
            return direction
        return None

    def kick_door(self, obs_a, obs_b, cursor_a):
        map_a = ascii_render(obs_a)
        map_lines = map_a.split("\n")[1:22]

        x, y = cursor_a[0], cursor_a[1] - 1

        # Check adjacent cells for a door '+'
        for direction, (dx, dy) in DIRECTIONS.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < 80 and 0 <= ny < 21:
                if map_lines[ny][nx] == "+":
                    return direction
        return None

    def detect_digging_direction(self, obs_a, obs_b, cursor_a):
        map_a = ascii_render(obs_a)
        map_b = ascii_render(obs_b)
        map_lines_a = map_a.split("\n")[1:22]
        map_lines_b = map_b.split("\n")[1:22]

        if cursor_a[1] == 0:
            x, y = self.player_position[0], self.player_position[1] - 1
        else:
            x, y = cursor_a[0], cursor_a[1] - 1

        for direction, (dx, dy) in DIRECTIONS.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < 80 and 0 <= ny < 21:
                if map_lines_a[ny][nx] == " " and map_lines_b[ny][nx] == "#":
                    return direction
                if (
                    map_lines_a[ny][nx] == "-" or map_lines_a[ny][nx] == "|"
                ) and map_lines_b[ny][nx] == ".":
                    return direction
        return None

    def detect_action_triggering_message(self, obs, timestep):
        """
        Function to detect the action that triggers an in-game message.

        Args:
            obs: The list of observations.
            timestep: The timestep of the observation.

        Returns:
            A string representing the action that triggers the message.
        """

        obs_a = obs[timestep]
        obs_b = obs[timestep + 1]
        # check it's not out of bounds
        if timestep + 2 >= len(obs):
            obs_c = obs_b
        else:
            obs_c = obs[timestep + 2]

        # obs_d
        if timestep + 3 >= len(obs):
            obs_d = obs_c
        else:
            obs_d = obs[timestep + 3]

        message_a = obs_to_message(obs_a)
        message_b = obs_to_message(obs_b)
        message_c = obs_to_message(obs_c)
        message_d = obs_to_message(obs_d)

        # print(self.last_action, "|", message_a, message_b, message_c)

        # Check stair movement
        if "You descend the stairs" in message_b:
            return "down"
        elif "You climb up the stairs" in message_b:
            return "up"
        elif (
            "eat it? [yn" in message_b
            or "You don't have anything to eat." in message_b
            or "What do you want to eat?" in message_b
        ):
            return "eat"
        elif "You find a hidden " in message_b:
            # This is not entirely correct... We should check how long is the player counting
            return "n20s"
        elif "What do you want to use or apply?" in message_b:
            self.last_action = "apply"
            return "apply"
        elif "What do you want to wield?" in message_b:
            # Add the acting on message to actually decide what to wield
            return "wield"
        elif "What do you want to wear?" in message_b:
            # Add the acting on message to actually decide what to wear
            self.last_action = "wear"
            return "wear"
        elif "What do you want to take off?" in message_b:
            # NEED TO ADD TAKE OFF INTERACTION! These may happen many timesteps after,
            # as it takes time to dress/undress in NetHack. WE actually have no information
            # On what is being worn, other than a "What do you want to wear?" message. We do however
            # have information of what is being taken off........
            return "takeoff"
        elif "What do you want to dip?" in message_b:
            return "dip"
        elif "What do you want to remove?" in message_b:
            return "remove"
        elif "Pick up what?" in message_b:
            return "pickup"
        elif "What do you want to read?" in message_b:
            self.last_action = "read"
            return "read"
        elif "What do you want to write with?" in message_b:
            return "engrave"
        elif "What do you want to drop?" in message_b:
            return "drop"
        elif "Drop what type of item?" in message_b:
            return "drop"
        elif "What do you want to drink?" in message_b:
            return "quaff"
        elif "What do you want to rub?" in message_b:
            return "rub"
        ######### ZAP WAND ############
        elif "What do you want to zap?" in message_b:
            self.last_action = "zap"
            return "zap"
        # elif self.last_action = ""

        ###############################
        elif "What do you want to put on?" in message_b:
            return "puton"
        elif "Choose which spell to cast" in message_b:
            return "cast"
        elif "What do you want to throw" in message_b:
            # Missing throwing food to tame pets! (The warhorse catches the apple.  The warhorse eats an uncursed apple.)
            self.last_action = "throw"
            return "throw"
        elif "loot it?" in message_b:
            # TODO: Add take out what to loot actions
            return "loot"
        elif "force its lock" in message_b:
            return "force"
        elif "You have a little trouble lifting" in message_b:
            return "pickup"
        elif "What do you want to ready?" in message_b:
            return "quiver"
        elif (
            "What do you want to wield" in message_a
            or "What do you want to ready?" in message_a
        ):
            if " - " in message_b:
                return message_b.split(" - ")[0]
        elif "loot it? [ynq]" in message_a:
            if "open" in message_b or "locked" in message_b:
                return "y"
            else:
                return "n"
        elif "In what direction?" in message_b:
            # Probably also "fire" and "zap" should be here
            if "WHAMMM!!!" in message_c or "As you kick the door" in message_c:
                return "kick"
            elif "THUD!" in message_c or "THUD!" in message_d:
                self.last_action = "kick"
                return "kick"
            elif "arrow" in message_c:
                return "fire"
            elif "You throw" in message_c or " hits " in message_c:
                self.player_position = self.tty_cursor[timestep]
                return "throw"
            elif (
                "The door closes" in message_c
                or "The door closes" in message_d
                or "The door resists" in message_c
                or "The door resists" in message_d
            ):
                self.last_action = "close"
                return "close"
        elif "Count: " in message_b:
            # TODO: # This should be modified. Sometimes instead of searching we just wait!
            # Wait and do nothing apparently is better to heal faster and not get hunger
            count = message_b.split("Count: ")[1].strip()
            return f"n{count}s"

        elif (
            "Weapons" in message_b
            or "Coins" in message_b
            or "Amulets" in message_b
            or "Armor" in message_b
        ):
            if find_n_of_m_end(ascii_render(obs_c)):
                self.inventory.update_inventory_from_maps(obs_a, obs_b, delete_old=True)
            else:
                self.inventory.update_inventory_from_maps(obs_a, obs_b)
            self.last_action = "inventory"
            return "inventory"
        elif (
            any(interaction in message_a for interaction in MENU_INTERACTION_MSG)
        ) and not any(
            (interaction in message_b for interaction in MENU_INTERACTION_MSG)
        ):
            return "more"

        elif "Sell it?" in message_a and "You sold" in message_b:
            return "y"

        ########### BOULDER ############
        if "You try to move the boulder, but in vain" in message_b:
            return self.find_adjacent(obs_a, self.tty_cursor[timestep], "0")

        ########### SCROLLS ############
        elif self.last_action == "read" and "As you read" in message_b:
            return self.read_scroll(obs_a, obs_b, obs_c, obs_d)

        ########### DIP ############
        elif "Dip" in message_b and (
            "[yn]" in message_b or "What do you want to dip a" in message_b
        ):
            if "[yn]" in message_b:
                pattern = r"(?i)dip (.*?) into"
                match = re.search(pattern, message_b)
                if match:
                    item_name = match.group(1).strip()
                    item = self.inventory.get_inventory_item(item_name)
                    # print(item, item_name)
                    return item
                else:
                    return None

        elif "What do you want to dip a" in message_b:
            if find_single_option(message_b):
                return find_single_option(message_b)
            # TODO: Dipping objects in potions for example
            pass
            # item = message_b.split("dip a ")[1].split(" into")[0].strip()
            # self.inventory.add_item(item)
            # return item

        ########### CALL/NAME ############
        elif "What do you want to name" in message_a:
            if "What do you want to name" in message_b:
                self.last_action = "call"
                return "i"
            if "What do you want to call" in message_b:
                self.last_action = "call"
                return "o"
            else:
                return "esc"

        elif self.last_action == "call" and (
            any(
                call_msg in message_b or call_msg in message_c
                for call_msg in CALL_MESSAGES
            )
        ):

            # self.inventory.print_inventory()
            if "Call a" in message_b:
                item = re.findall(r"Call a[n]?\s(.*?):", message_b)[0]
            elif "Call a" in message_c:
                item = re.findall(r"Call a[n]?\s(.*?):", message_c)[0]
            elif "What do you want to name th" in message_b:
                item = re.findall(
                    r"What do you want to name (this|these) (.*?)\?", message_b
                )[0][1]
            elif "What do you want to name th" in message_c:
                item = re.findall(
                    r"What do you want to name (this|these) (.*?)\?", message_c
                )[0][1]
            return self.inventory.get_inventory_item(item)

        ########### SHOP ############
        elif "Pay? [yn]" in message_b:
            self.last_action = "pay"
            return "pay"

        elif "You bought" in message_b:
            return "y"

        ########### ANIMATIONS ############
        if (
            message_a.strip() in message_b.strip()
        ) and not "In what direction" in message_b:
            return " "

        else:
            return None

    def find_coordinates(self, map, symbol, including_player=False):
        map_lines = map.split("\n")[1:22]
        player_cursor = (self.player_position[0], self.player_position[1] - 1)
        for y, line in enumerate(map_lines):
            for x, char in enumerate(line):
                if char == symbol:
                    if char == "@" and (x, y) == player_cursor and not including_player:
                        continue
                    return x, y
        return None

    def detect_ranged_attack_direction(self, obs, timestep, attack_type="throw"):
        if attack_type == "throw":
            msg = " hits "
        elif attack_type == "fire":
            msg = " hits "
        elif attack_type == "zap":
            msg = " zaps"  # maybe, not sure
        else:
            assert False, "Invalid attack type"

        # Check the next messages to see if the player hit something
        for i in range(5):
            message = obs_to_message(obs[timestep + i])
            if msg in message:
                for monster in MONSTER_DICT.keys():
                    if monster in message:
                        symbol = MONSTER_DICT[monster]["ascii"]
                        map = ascii_render(obs[timestep])
                        enemy_position = self.find_coordinates(map, symbol)
                        if not enemy_position:
                            return None
                        player_cursor = (
                            self.player_position[0],
                            self.player_position[1] - 1,
                        )
                        direction = self.find_direction(
                            player_cursor,
                            enemy_position,
                        )
                        if direction:
                            return direction

        return None

    def detect_acting_on_message(self, obs, cursor, timestep):
        """
        Detects the action performed by the player after a message requiring an action
        is displayed.
        Args:
            obs: The list of observations.
            cursor: The list of cursor positions.
            timestep: The timestep of the observation.
        Returns:
            A string representing the action performed by the player.
        """

        obs_a = obs[timestep]
        obs_b = obs[timestep + 1]
        if timestep + 2 >= len(obs):
            obs_c = obs_b
        else:
            obs_c = obs[timestep + 2]

        if timestep + 3 >= len(obs):
            obs_d = obs_c
        else:
            obs_d = obs[timestep + 3]
        cursor_a = cursor[timestep]
        cursor_b = cursor[timestep + 1]

        message_a = obs_to_message(obs_a)
        message_b = obs_to_message(obs_b)
        message_c = obs_to_message(obs_c)

        if "--More" in message_a:
            return "more"
        elif "What do you want to eat?" in message_a and any(
            eat_msg in message_b for eat_msg in EATING
        ):
            return self.inventory.get_inventory_item(message_b)
        elif any(
            what_do_you_want in message_a
            for what_do_you_want in WHAT_DO_YOU_WANT_MESSAGES.keys()
        ) and any(inventory_type in message_b for inventory_type in INVENTORY_TYPES):
            self.inventory.update_inventory_from_single_map(obs_b)
            for key, return_value in INVENTORY_LOOK_SYMBOLS.items():
                if key in message_a:
                    return return_value
            return "*"
            # return self.get_inventory_item(message_b)
        elif "Never mind." in message_b:
            return "esc"
        elif "You find a hidden " in message_b:
            return "20s"
        elif "What do you want to wield" in message_a:
            if " - " in message_b:
                return message_b.split(" - ")[0]
        elif (
            "WHAMMM!!!" in message_b
            or "As you kick the door" in message_b
            or "WHAMMM!!!" in message_c
        ):
            action = self.find_door(obs_a, cursor_b)
            if action:
                return action
        elif "THUD!" in message_b or "THUD!" in message_c:
            action = self.find_adjacent(obs_a, cursor_b, "(")
            if action:
                return action
        elif "You throw" in message_b:
            direction = self.detect_ranged_attack_direction(
                obs, timestep, attack_type="throw"
            )
            if direction:
                return direction
        elif "Do what?" in message_b:
            return ""
        elif (
            ("[yn]" in message_a or "[ynq]" in message_a)
            and ")" in message_b
            and len(message_b.split(") ")) > 1
            and not "save" in message_a
        ):
            return message_b.split(") ")[1].strip()
        elif "[rl]" in message_a:
            if "]" in message_b and len(message_b.split("] ")) > 1:
                return message_b.split("] ")[1].strip()
            elif "right" in message_b:
                return "r"
            elif "left" in message_b:
                return "l"
            else:
                return " "
        # Check if any of the eating messages are in the message
        elif "eat it? [ynq]" in message_a and any(
            eat_msg in message_b for eat_msg in EATING
        ):
            return "y"
        elif "eat it? [ynq]" in message_a and not any(
            eat_msg in message_b for eat_msg in EATING
        ):
            return "n"
        elif "little trouble lifting" in message_a and "Continue?" in message_a:
            return self.check_pickup(timestep)

        ############ BAG OF HOLDING/SACK ##############
        elif "What do you want to use or apply" in message_a and (
            "Do what with your bag" in message_b or "empty" in message_b
        ):
            if "bag of holding" in message_b.lower():
                item = "bag of holding"
            elif "sack" in message_b.lower():
                item = "sack"
            else:
                item = "bag"
            self.last_action = "menu interaction"
            # self.inventory.print_inventory()
            return self.inventory.get_inventory_item(item)
        elif "What do you want to use or apply" in message_a and (
            "You produce a strange whistling sound" in message_c
            or "You produce a high whistling sound" in message_c
        ):
            return self.inventory.get_inventory_item("whistle")
        elif "What do you want to use or apply" in message_a and (
            "Unlock it?" in message_c or "Unlock it it?" in obs_to_message(obs_d)
        ):
            # The lockpick might have a similar message to the key too!
            return self.inventory.get_inventory_item("key")
        elif (
            "What do you want to put on?" in message_a
            or "What do you want to remove?" in message_a
        ) and ("towel" in message_b or "towel" in message_c):
            return self.inventory.get_inventory_item("towel")

        elif "into the fountain? [yn] (n)" in message_a:
            return "y"

        elif (
            "What do you want to name" in message_a
            and not "What do you want to name" in message_b
        ):
            return "more"
        elif self.last_action == "writing" and not (
            "write" in message_b
            or "call" in message_b.lower()
            or "name" in message_b.lower()
        ):
            self.last_action = ""
            return "more"
        elif "What do you want to name" in message_b:
            return "name"

        elif (
            self.last_action == "untrap"
            and "In what direction?" in message_a
            and self.find_adjacent(obs_a, cursor_a, "^")
        ):
            self.last_action = ""
            return self.find_adjacent(obs_a, cursor_a, "^")

        elif self.last_action == "close":
            direction = None
            if "The door closes" in message_b:
                direction = self.find_door(obs_b, cursor_a)
            elif "The door closes" in message_c:
                direction = self.find_door(obs_c, cursor_a)
            if direction:
                self.last_action = ""
                return direction
            else:
                self.last_action = ""
                return " "

        elif (
            "What do you want to call" in message_a
            or "What do you want to name" in message_a
        ) and any(inv_type in message_b for inv_type in INVENTORY_TYPES.keys()):
            self.inventory.update_inventory_from_single_map(obs_b)
            self.last_action = "call"
            return "*"

        ############# SCROLLS ##############
        elif "What do you want to read" in message_a and (
            "Scrolls" in message_b or "Spellbooks" in message_b
        ):
            self.inventory.update_inventory_from_single_map(obs_b)
            self.last_action = "read"
            return "?"

        ############# OFFER ##############
        elif "What do you want to sacrifice?" in message_a:
            if find_single_option(message_a):
                return find_single_option(message_a).group(1)
            else:
                return "?"

        ############# RUB ##############
        elif "What do you want to rub?" in message_a:
            if find_single_option(message_a):
                return find_single_option(message_a).group(1)
            else:
                return self.inventory.get_inventory_item("lamp")

        ############# PRAY ##############
        elif (
            "Are you sure you want to pray?" in message_a
            and "You begin praying" in message_b
        ):
            return "y"

        ####################################
        elif "Do you want to add" in message_a and "You wipe" in message_b:
            return "n"

        elif "Sell it?" in message_a and "You sold" in message_b:
            return "y"

        elif "You succeed in" in message_b:
            self.last_action = ""
            return "y"

        return "acting on a message"

    def find_direction(self, cursor_a, cursor_b):
        x_movement = cursor_b[0] - cursor_a[0]
        y_movement = cursor_b[1] - cursor_a[1]

        if x_movement == 0 and y_movement > 0:
            return "S"
        elif x_movement == 0 and y_movement < 0:
            return "N"
        elif x_movement > 0 and y_movement == 0:
            return "E"
        elif x_movement < 0 and y_movement == 0:
            return "W"
        elif x_movement > 0 and y_movement > 0 and x_movement == y_movement:
            return "SE"
        elif x_movement > 0 and y_movement < 0 and x_movement == -y_movement:
            return "NE"
        elif x_movement < 0 and y_movement > 0 and x_movement == -y_movement:
            return "SW"
        elif x_movement < 0 and y_movement < 0 and x_movement == y_movement:
            return "NW"
        else:
            return "unknown"

    def detect_movement(self, obs_a, obs_b, cursor_a, cursor_b):
        # TODO: Movement on corpses, boulders, etc. We need to check the map for this.

        x_movement = cursor_b[0] - cursor_a[0]
        y_movement = cursor_b[1] - cursor_a[1]

        multiple_step = False
        if abs(x_movement) > 1 or abs(y_movement) > 1:
            if cursor_a[1] == 0 or cursor_a[1] == 23:
                x, y = self.player_position[0], self.player_position[1]
                x_movement = cursor_b[0] - x
                y_movement = cursor_b[1] - y

            else:
                x, y = cursor_a[0], cursor_a[1]

            if (
                y <= 20
                and check_cursor(ascii_render(obs_b), cursor_b, "@")
                and check_cursor(ascii_render(obs_a), (x, y), "@")
            ):
                multiple_step = True
                x_movement = max(-1, min(1, x_movement))
                y_movement = max(-1, min(1, y_movement))

        message_a = obs_to_message(obs_a)
        message_b = obs_to_message(obs_b)
        self.player_position = cursor_b if cursor_b[1] <= 20 else self.player_position
        action = self.last_direction
        if x_movement == 0 and y_movement == 1:
            action = "south"
        elif x_movement == 0 and y_movement == -1:
            action = "north"
        elif x_movement == 1 and y_movement == 0:
            action = "east"
        elif x_movement == -1 and y_movement == 0:
            action = "west"
        elif x_movement == 1 and y_movement == 1:
            action = "southeast"
        elif x_movement == 1 and y_movement == -1:
            action = "northeast"
        elif x_movement == -1 and y_movement == 1:
            action = "southwest"
        elif x_movement == -1 and y_movement == -1:
            action = "northwest"
        elif x_movement == 0 and y_movement == 0:
            if "You try to move the boulder, but in vain" in message_b:
                action = self.find_adjacent(obs_a, cursor_a, "0")
            elif self.detect_digging_direction(obs_a, obs_b, cursor_a):
                action = self.detect_digging_direction(obs_a, obs_b, cursor_a)
            elif "Autopickup" in message_b:
                return "autopickup"
            elif " - " in message_b and not find_n_of_m(ascii_render(obs_b)):
                item = message_b.replace(".", "")
                self.inventory.add_item(item)
                return "pickup"
            elif (
                find_n_of_m(ascii_render(obs_b))
                and not ("Put in what?" in message_a or "Take out what?" in message_a)
                and self.last_action == "inventory"
            ):
                self.inventory.update_inventory_from_maps(obs_a, obs_b)
                return " "
            elif any(eat_msg in message_b for eat_msg in EATING):
                if "eat it? [ynq]" in message_a:
                    return "y"
                if message_a.strip() in message_b:
                    return " "
                return self.inventory.get_inventory_item(message_b)
            elif "Never mind" in message_b:
                return "esc"
            elif "This door is locked" in message_b:
                return self.find_door(obs_a, cursor_a)
            # Looking for a hidden door! You can move into a wall to try to find a hidden door/passage
            elif (
                get_timestep(obs_to_stats(obs_a)) != ""
                and get_timestep(obs_to_stats(obs_b)) != ""
            ):
                if int(get_timestep(obs_to_stats(obs_a))) + 1 == int(
                    get_timestep(obs_to_stats(obs_b))
                ):
                    if "You find a hidden" in message_b:
                        return self.find_door(obs_a, cursor_a)
                    elif "You find a hidden passage":
                        return self.find_adjacent_change(obs_a, obs_b, cursor_a, "#")
                    elif "The door resists" in message_b:
                        return self.find_door(obs_a, cursor_a)
                    return "search"
            else:
                # TODO: ANIMATIONS WILL BE DETECTED AS NO MOVEMENTS! BUT NO ACTION IS TAKEN!
                # TODO: A lot of the no movement actions are just game messages being displayed in
                # the next screen (like): You hear a gurgling noise. or "You hear a door open."
                # We could probably remove all the "no movement detected" actions to " "
                return "no movement detected"
        else:
            if any(filler in message_b for filler in ANIMATION_FILLERS):
                return " "

            timestep_a = get_timestep(obs_to_stats(obs_a))
            timestep_b = get_timestep(obs_to_stats(obs_b))

            if timestep_a != "" and timestep_b != "":
                if int(timestep_a) == int(timestep_b) and (
                    cursor_a[0] == cursor_b[0] and cursor_a[1] == cursor_b[1]
                ):
                    return "search"
                elif int(timestep_a) + 1 == int(timestep_b):
                    return " "

            if self.last_action == "far movement":
                self.last_action = ""
                return " "

            if "Things that are here" in message_b:
                return " "

            # TODO: SOMETIMES THE UNKNOWN ACTION IS A MOVEMENT ACTION AFTER A FAR MOVEMENT... THESE
            # ARE QUITE TRICKY, THE CURSOR GOES TO [... 23]

            return "unknown action"

        self.last_direction = action

        if multiple_step:
            self.last_action = "far movement"
            return f"far {action}"
        return action
