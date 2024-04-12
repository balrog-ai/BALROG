import nle.nethack as nh

from autoascend import objects as O


class InventoryItems:
    def __init__(self, agent):
        self.agent = agent
        self._previous_inv_strs = None

        self._clear()

    def _clear(self):
        self.main_hand = None
        self.off_hand = None
        self.suit = None
        self.helm = None
        self.gloves = None
        self.boots = None
        self.cloak = None
        self.shirt = None

        self.total_weight = 0

        self.all_items = []
        self.all_letters = []

        self._recheck_containers = True

    def __iter__(self):
        return iter(self.all_items)

    def __str__(self):
        return (
                f'main_hand: {self.main_hand}\n'
                f'off_hand : {self.off_hand}\n'
                f'suit     : {self.suit}\n'
                f'helm     : {self.helm}\n'
                f'gloves   : {self.gloves}\n'
                f'boots    : {self.boots}\n'
                f'cloak    : {self.cloak}\n'
                f'shirt    : {self.shirt}\n'
                f'Items:\n' +
                '\n'.join([f' {l} - {i}' for l, i in zip(self.all_letters, self.all_items)])
        )

    def total_nutrition(self):
        ret = 0
        for item in self:
            if item.is_food():
                ret += item.object.nutrition * item.count
        return ret

    def free_slots(self):
        is_coin = any((isinstance(item, O.Coin) for item in self))
        return 52 + is_coin - len(self.all_items)

    def on_panic(self):
        self._previous_inv_strs = None
        self._clear()

    def update(self, force=False):
        if force:
            self._recheck_containers = True

        if force or self._previous_inv_strs is None or \
                (self.agent.last_observation['inv_strs'] != self._previous_inv_strs).any():
            self._clear()
            self._previous_inv_strs = self.agent.last_observation['inv_strs']
            previous_inv_strs = self._previous_inv_strs

            # For some reasons sometime the inventory entries in last_observation may be duplicated
            iterable = set()
            for item_name, category, glyph, letter in zip(
                    self.agent.last_observation['inv_strs'],
                    self.agent.last_observation['inv_oclasses'],
                    self.agent.last_observation['inv_glyphs'],
                    self.agent.last_observation['inv_letters']):
                item_name = bytes(item_name).decode().strip('\0')
                letter = chr(letter)
                if not item_name:
                    continue
                iterable.add((item_name, category, glyph, letter))
            iterable = sorted(iterable, key=lambda x: x[-1])

            assert len(iterable) == len(set(map(lambda x: x[-1], iterable))), \
                'letters in inventory are not unique'

            for item_name, category, glyph, letter in iterable:
                item = self.agent.inventory.item_manager.get_item_from_text(item_name, category=category,
                                                                            glyph=glyph if not nh.glyph_is_body(
                                                                                glyph) and not nh.glyph_is_statue(
                                                                                glyph) else None,
                                                                            position=None)

                self.all_items.append(item)
                self.all_letters.append(letter)

                if item.equipped:
                    for types, sub, name in [
                        ((O.Weapon, O.WepTool), None, 'main_hand'),
                        (O.Armor, O.ARM_SHIELD, 'off_hand'),  # TODO: twoweapon support
                        (O.Armor, O.ARM_SUIT, 'suit'),
                        (O.Armor, O.ARM_HELM, 'helm'),
                        (O.Armor, O.ARM_GLOVES, 'gloves'),
                        (O.Armor, O.ARM_BOOTS, 'boots'),
                        (O.Armor, O.ARM_CLOAK, 'cloak'),
                        (O.Armor, O.ARM_SHIRT, 'shirt'),
                    ]:
                        if isinstance(item.objs[0], types) and (sub is None or sub == item.objs[0].sub):
                            assert getattr(self, name) is None, ((name, getattr(self, name), item), str(self), iterable)
                            setattr(self, name, item)
                            break

                if item.is_possible_container() or (item.is_container() and self._recheck_containers):
                    self.agent.inventory.check_container_content(item)

                if (self.agent.last_observation['inv_strs'] != previous_inv_strs).any():
                    self.update()
                    return

                self.total_weight += item.weight()
                # weight is sometimes unambiguous for unidentified items. All exceptions:
                # {'helmet': 30, 'helm of brilliance': 50, 'helm of opposite alignment': 50, 'helm of telepathy': 50}
                # {'leather gloves': 10, 'gauntlets of fumbling': 10, 'gauntlets of power': 30, 'gauntlets of dexterity': 10}
                # {'speed boots': 20, 'water walking boots': 15, 'jumping boots': 20, 'elven boots': 15, 'fumble boots': 20, 'levitation boots': 15}
                # {'luckstone': 10, 'loadstone': 500, 'touchstone': 10, 'flint': 10}

            self._recheck_containers = False

    def get_letter(self, item):
        assert item in self.all_items, (item, self.all_items)
        return self.all_letters[self.all_items.index(item)]
