import numpy as np
import difflib

# from utils import *

from inverse_dynamics.utils import *

INVENTORY_TYPES = {
    "Amulets": "",
    "Coins": "",
    "Weapons": "",
    "Armor": "",
    "Comestibles": "",
    "Scrolls": "",
    "Spellbooks": "",
    "Potions": "",
    "Rings": "",
    "Wands": "",
    "Tools": "",
    "Miscellaneous": "",
    "Gems/Stones": "",
}


class Inventory:
    def __init__(self, items=None):
        self.inventory = items if items else INVENTORY_TYPES
        self.raw_inventory_message = ""

    def _parse_inventory(self, message):
        inventory_dict = {}
        current_type = None

        # Initialize all inventory types in the dictionary
        for inv_type in INVENTORY_TYPES.keys():
            inventory_dict[inv_type] = []

        # Split the message into lines
        lines = message.split("\n")

        for line in lines:
            stripped_line = line.strip()

            # Check if the line is an inventory type
            if stripped_line in INVENTORY_TYPES.keys():
                current_type = stripped_line
            elif current_type:
                # If there is a current inventory type, add the line to its list
                inventory_dict[current_type].append(stripped_line)

        # Convert lists to strings, joined by newline character
        for inv_type in INVENTORY_TYPES.keys():
            inventory_dict[inv_type] = "\n".join(inventory_dict[inv_type]).strip()

        return inventory_dict

    def print_inventory(self):
        for inv_type, items in self.inventory.items():
            print(f"{inv_type}:\n{items}")

    def get_inventory(self):
        inventory = ""
        for inv_type, items in self.inventory.items():
            if items != "":
                inventory += f"{inv_type}\n{items}\n"
        return inventory

    def get_inventory_item(self, target_item):
        """
        Checks whether the item is in the inventory, if it is returns the corresponding letter.
        Args:
            item: The item to check.
        Returns:
            The letter corresponding to the item in the inventory.
        """
        self.raw_inventory_message = ""
        inventory = [item for key, item in self.inventory.items()]
        inventory = "\n".join(inventory)
        inventory = inventory.split("\n")
        inventory = [line.split("-") for line in inventory]
        inventory_items = [
            (line[0].strip(), line[1].strip()) for line in inventory if len(line) > 1
        ]
        # print(inventory_items)
        # print(target_item)

        for letter, item in inventory_items:
            item = clean_string(item)
            if item.lower() in target_item.lower():
                return letter

        for letter, item in inventory_items:
            item = clean_string(item)
            # print(target_item.lower(), item.lower())
            if target_item.lower() in item.lower():
                return letter

        # NOT WORKING PROPERLY YET. Reason being often times the items are not displayed in the inventory yet
        # This could be because the inventory is too long and not displayed withing 24 lines, or because objects
        # where picked up after the inventory was displayed and thus not saved in it...

        # TODO: THIS SHOULD BE REPLACED WITH A *, so that at least the agent will learn
        # to open the inventory and decide for itself what to do
        return "inventory item not found"

    def add_item(self, item):
        if self.inventory:
            # Check if item is already in the inventory. If it is, do not add it again
            if "gold pieces" in item:
                self.inventory["Coins"] = item
                return
            for inv_type, items in self.inventory.items():
                if item in items:
                    return
            self.inventory["Miscellaneous"] += f"\n{item}"
        else:
            self.inventory = {"Miscellaneous": item}

    def update_inventory_from_single_map(self, map, ascii=False):
        menu = get_menu_message(map, ascii=ascii)
        self.update_inventory_from_changes(changes=menu)

    def update_inventory_from_maps(self, obs_a, obs_b, delete_old=False, ascii=False):
        changes = get_changes(obs_a, obs_b, ascii=ascii)
        changes = clean_n_m_end(changes)
        self.update_inventory_from_changes(changes, delete_old=delete_old)

    def update_inventory_from_changes(self, changes, delete_old=False):
        if delete_old:
            self.raw_inventory_message = ""
            self.inventory = INVENTORY_TYPES

        self.raw_inventory_message += "\n"
        self.raw_inventory_message += changes
        new_inventory = self._parse_inventory(self.raw_inventory_message)

        # Override the new inventory items
        for inv_type, new_items in new_inventory.items():
            if new_items == "":
                continue
            else:
                self.inventory[inv_type] = new_items
        # Clean miscellaneous items
        if "Miscellaneous" in self.inventory:
            self.inventory["Miscellaneous"] = ""

    def update_type(self, inv_type, new_items):
        self.inventory[inv_type] = new_items

    def get_inventory_type(self, inv_type):
        return self.inventory[inv_type]


def main():
    inventory = Inventory()
    # TESTING


if __name__ == "__main__":
    main()
