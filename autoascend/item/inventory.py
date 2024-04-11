import contextlib
import re
from functools import partial
from itertools import chain

import nle.nethack as nh
import numpy as np
from nle.nethack import actions as A

from autoascend import objects as O, utils
from autoascend.character import Character
from autoascend.exceptions import AgentPanic
from autoascend.glyph import G
from autoascend.item import ItemManager, Item, ContainerContent, check_if_triggered_container_trap, \
    find_equivalent_item, flatten_items
from autoascend.item.inventory_items import InventoryItems
from autoascend.strategy import Strategy


class Inventory:
    _name_to_category = {
        'Amulets': nh.AMULET_CLASS,
        'Armor': nh.ARMOR_CLASS,
        'Comestibles': nh.FOOD_CLASS,
        'Coins': nh.COIN_CLASS,
        'Gems/Stones': nh.GEM_CLASS,
        'Potions': nh.POTION_CLASS,
        'Rings': nh.RING_CLASS,
        'Scrolls': nh.SCROLL_CLASS,
        'Spellbooks': nh.SPBOOK_CLASS,
        'Tools': nh.TOOL_CLASS,
        'Weapons': nh.WEAPON_CLASS,
        'Wands': nh.WAND_CLASS,
        'Boulders/Statues': nh.ROCK_CLASS,
        'Chains': nh.CHAIN_CLASS,
        'Iron balls': nh.BALL_CLASS,
    }

    def __init__(self, agent):
        self.agent = agent
        self.item_manager = ItemManager(self.agent)
        self.items = InventoryItems(self.agent)

        self._previous_blstats = None
        self.items_below_me = None
        self.letters_below_me = None
        self.engraving_below_me = None

        self.skip_engrave_counter = 0

    def on_panic(self):
        self.items_below_me = None
        self.letters_below_me = None
        self.engraving_below_me = None
        self._previous_blstats = None

        self.item_manager.on_panic()
        self.items.on_panic()

    def update(self):
        self.item_manager.update()
        self.items.update()

        if self._previous_blstats is None or \
                (self._previous_blstats.y, self._previous_blstats.x, \
                 self._previous_blstats.level_number, self._previous_blstats.dungeon_number) != \
                (self.agent.blstats.y, self.agent.blstats.x, \
                 self.agent.blstats.level_number, self.agent.blstats.dungeon_number) or \
                (self.engraving_below_me is None or self.engraving_below_me.lower() == 'elbereth'):
            assume_appropriate_message = self._previous_blstats is not None and not self.engraving_below_me

            self._previous_blstats = self.agent.blstats
            self.items_below_me = None
            self.letters_below_me = None
            self.engraving_below_me = None

            self.get_items_below_me(assume_appropriate_message=assume_appropriate_message)

        assert self.items_below_me is not None and self.letters_below_me is not None and self.engraving_below_me is not None

    @contextlib.contextmanager
    def panic_if_items_below_me_change(self):
        old_items_below_me = self.items_below_me
        old_letters_below_me = self.letters_below_me

        def f(self):
            if (
                    [(l, i.text) for i, l in zip(old_items_below_me, old_letters_below_me)] !=
                    [(l, i.text) for i, l in zip(self.items_below_me, self.letters_below_me)]
            ):
                raise AgentPanic('items below me changed')

        fun = partial(f, self)

        self.agent.on_update.append(fun)

        try:
            yield
        finally:
            assert fun in self.agent.on_update
            self.agent.on_update.pop(self.agent.on_update.index(fun))

    ####### ACTIONS

    def wield(self, item, smart=True):
        if smart:
            if item is not None:
                item = self.move_to_inventory(item)

        if item is None:  # fists
            letter = '-'
        else:
            letter = self.items.get_letter(item)

        if item is not None and item.equipped:
            return True

        if self.agent.character.prop.polymorph:
            # TODO: depends on kind of a monster
            return False

        if (self.items.main_hand is not None and self.items.main_hand.status == Item.CURSED) or \
                (item is not None and item.objs[0].bi and self.items.off_hand is not None):
            return False

        with self.agent.atom_operation():
            self.agent.step(A.Command.WIELD)
            if "Don't be ridiculous" in self.agent.message:
                return False
            assert 'What do you want to wield' in self.agent.message, self.agent.message
            self.agent.type_text(letter)
            if 'You cannot wield a two-handed sword while wearing a shield.' in self.agent.message or \
                    'You cannot wield a two-handed weapon while wearing a shield.' in self.agent.message or \
                    ' welded to your hand' in self.agent.message:
                return False
            assert re.search(r'(You secure the tether\.  )?([a-zA-z] - |welds?( itself| themselves| ) to|'
                             r'You are already wielding that|You are empty handed|You are already empty handed)', \
                             self.agent.message), (self.agent.message, self.agent.popup)

        return True

    def wear(self, item, smart=True):
        assert item is not None

        if smart:
            item = self.move_to_inventory(item)
            # TODO: smart should be more than that (taking off the armor for shirts, etc)
        letter = self.items.get_letter(item)

        if item.equipped:
            return True

        for i in self.items:
            assert not isinstance(i, O.Armor) or i.sub != item.sub or not i.equipped, (i, item)

        with self.agent.atom_operation():
            self.agent.step(A.Command.WEAR)
            if "Don't even bother." in self.agent.message:
                return False
            assert 'What do you want to wear?' in self.agent.message, self.agent.message
            self.agent.type_text(letter)
            assert 'You finish your dressing maneuver.' in self.agent.message or \
                   'You are now wearing ' in self.agent.message or \
                   'Your foot is trapped!' in self.agent.message, self.agent.message

        return True

    def takeoff(self, item):
        # TODO: smart

        assert item is not None and item.equipped, item
        letter = self.items.get_letter(item)
        assert item.status != Item.CURSED, item

        equipped_armors = [i for i in self.items if i.is_armor() and i.equipped]
        assert item in equipped_armors

        with self.agent.atom_operation():
            self.agent.step(A.Command.TAKEOFF)

            is_take_off_message = lambda: \
                'You finish taking off ' in self.agent.message or \
                'You were wearing ' in self.agent.message or \
                'You feel that monsters no longer have difficulty pinpointing your location.' in self.agent.message

            if len(equipped_armors) > 1:
                if is_take_off_message():
                    raise AgentPanic('env did not ask for the item to takeoff')
                assert 'What do you want to take off?' in self.agent.message, self.agent.message
                self.agent.type_text(letter)
            if 'It is cursed.' in self.agent.message or 'They are cursed.' in self.agent.message:
                return False
            assert is_take_off_message(), self.agent.message

        return True

    def use_container(self, container, items_to_put, items_to_take, items_to_put_counts=None,
                      items_to_take_counts=None):
        assert container in self.items.all_items or container in self.items_below_me
        assert all((item in self.items.all_items for item in items_to_put))
        assert all((item in container.content.items for item in items_to_take))
        assert container.is_container()
        assert len(items_to_take) - len(items_to_put) <= self.items.free_slots()  # TODO: take counts into consideration
        assert not container.content.locked, container

        def gen():
            if ' vanished!' in self.agent.message:
                self.item_manager.container_contents.pop(container.container_id)
                raise AgentPanic('some items from the container vanished')
            if 'You carefully open ' in self.agent.single_message or 'You open ' in self.agent.single_message:
                yield ' '
            assert 'You have no free hand.' not in self.agent.single_message, 'TODO: handle it'
            assert 'Do what with ' in self.agent.single_popup[0]
            if items_to_put and items_to_take:
                yield 'r'
            elif items_to_put and not items_to_take:
                yield 'i'
            elif not items_to_put and items_to_take:
                yield 'o'
            else:
                assert 0
            if items_to_put:
                if 'Put in what type of objects?' in self.agent.single_popup[0]:
                    yield from 'a\r'
                assert 'Put in what?' in self.agent.single_popup[0], (
                    self.agent.single_message, self.agent.single_popup)
                yield from self._select_items_in_popup(items_to_put, items_to_put_counts)
            if items_to_take:
                while not self.agent.single_popup or self.agent.single_popup[0] not in [
                    'Take out what type of objects?', 'Take out what?']:
                    assert ' inside, you are blasted by a ' not in self.agent.message, self.agent.message
                    assert self.agent.single_message or self.agent.single_popup, (self.agent.message, self.agent.popup)
                    yield ' '
                if self.agent.single_popup[0] == 'Take out what type of objects?':
                    yield from 'a\r'
                assert 'Take out what?' in self.agent.single_popup[0]
                yield from self._select_items_in_popup(items_to_take, items_to_take_counts)

                if self.agent._observation['misc'][2]:
                    yield ' '
                while 'You have ' in self.agent.single_message and ' removing ' in self.agent.single_message and \
                        'Continue? [ynq] (q)' in self.agent.single_message:
                    yield 'y'

        with self.agent.atom_operation():
            # TODO: refactor: the same fragment is in check_container_content
            if container in self.items.all_items:
                self.agent.step(A.Command.APPLY)
                assert "You can't do that while carrying so much stuff." not in self.agent.message, self.agent.message
                self.agent.step(self.items.get_letter(container), gen())
            elif container in self.items_below_me:
                self.agent.step(A.Command.LOOT)
                while True:
                    assert 'Loot which containers?' not in self.agent.popup, self.agent.popup
                    assert 'Loot in what direction?' not in self.agent.message
                    if "You don't find anything here to loot." in self.agent.message:
                        raise AgentPanic('no container to loot')
                    r = re.findall(r'There is ([a-zA-z0-9# ]+) here\, loot it\? \[ynq\] \(q\)', self.agent.message)
                    assert len(r) == 1, self.agent.message
                    text = r[0]
                    it = self.item_manager.get_item_from_text(text,
                                                              position=(
                                                                  *self.agent.current_level().key(),
                                                                  self.agent.blstats.y,
                                                                  self.agent.blstats.x))
                    if it.container_id == container.container_id:
                        break
                    self.agent.step('n')

                self.agent.step('y', gen())
            else:
                assert 0

        for item in chain(self.items.all_items, self.items_below_me):
            if item.is_container() and item.container_id == container.container_id:
                self.check_container_content(item)

    def check_container_content(self, item):
        assert item.is_possible_container() or item.is_container()
        assert item in self.items.all_items or item in self.items_below_me

        is_bag_of_tricks = False
        if item.content is not None:
            content = item.content
            content.reset()
        else:
            content = ContainerContent()

        def gen():
            nonlocal content, is_bag_of_tricks

            if 'You carefully open ' in self.agent.single_message or 'You open ' in self.agent.single_message:
                yield ' '

            if 'It develops a huge set of teeth and bites you!' in self.agent.single_message:
                is_bag_of_tricks = True
                return

            if 'Hmmm, it turns out to be locked.' in self.agent.single_message or 'It is locked.' in self.agent.single_message:
                content.locked = True
                yield A.Command.ESC
                return

            if check_if_triggered_container_trap(self.agent.single_message):
                self.agent.stats_logger.log_event('triggered_undetected_trap')
                raise AgentPanic('triggered trap while looting')

            if 'You have no hands!' in self.agent.single_message or \
                    'You have no free hand.' in self.agent.single_message:
                return

            if ' vanished!' in self.agent.message:
                raise AgentPanic('some items from the container vanished')

            if 'cat' in self.agent.message and ' inside the box is ' in self.agent.message:
                raise AgentPanic('encountered a cat in a box')

            assert self.agent.single_popup, (self.agent.single_message)
            if '\no - ' not in '\n'.join(self.agent.single_popup):
                # ':' sometimes doesn't display items correctly if there's >= 22 items (the first page isn't shown)
                yield ':'
                if ' is empty' in self.agent.single_message:
                    return
                # if self.agent.single_popup and 'Contents of ' in self.agent.single_popup[0]:
                #     for text in self.agent.single_popup[1:]:
                #         if not text:
                #             continue
                #         content.items.append(self.item_manager.get_item_from_text(text, position=None))
                #     return
                assert 0, (self.agent.single_message, self.agent.single_popup)

            yield from 'o'
            if ' is empty' in self.agent.single_message and not self.agent.single_popup:
                return
            if self.agent.single_popup and self.agent.single_popup[0] == 'Take out what type of objects?':
                yield from 'a\r'
            if self.agent.single_popup and 'Take out what?' in self.agent.single_popup[0]:
                category = None
                while self.agent._observation['misc'][2]:
                    yield ' '
                assert self.agent.popup.count('Take out what?') == 1, self.agent.popup
                for text in self.agent.popup[self.agent.popup.index('Take out what?') + 1:]:
                    if not text:
                        continue
                    if text in self._name_to_category:
                        category = self._name_to_category[text]
                        continue
                    assert category is not None
                    assert text[1:4] == ' - '
                    text = text[4:]
                    content.items.append(self.item_manager.get_item_from_text(text, category=category, position=None))
                return

            assert 0, (self.agent.single_message, self.agent.single_popup)

        with self.agent.atom_operation():
            # TODO: refactor: the same fragment is in use_container
            if item in self.items.all_items:
                self.agent.step(A.Command.APPLY)
                if "You can't do that while carrying so much stuff." in self.agent.message:
                    return  # TODO: is not changing the content in this case a good way to handle this?
                self.agent.step(self.items.get_letter(item), gen())
                if 'You have no hands!' in self.agent.message:
                    return
            else:
                self.agent.step(A.Command.LOOT)
                while True:
                    if "You don't find anything here to loot." in self.agent.message:
                        raise AgentPanic('no container below me')
                    assert 'Loot which containers?' not in self.agent.popup, self.agent.popup
                    assert 'There is ' in self.agent.message and ', loot it?' in self.agent.message, self.agent.message
                    r = re.findall(r'There is ([a-zA-z0-9# ]+) here\, loot it\? \[ynq\] \(q\)', self.agent.message)
                    assert len(r) == 1, self.agent.message
                    text = r[0]
                    it = self.item_manager.get_item_from_text(text,
                                                              position=(
                                                                  *self.agent.current_level().key(),
                                                                  self.agent.blstats.y,
                                                                  self.agent.blstats.x))
                    if (item.container_id is not None and it.container_id == item.container_id) or \
                            (item.container_id is None and item.text == it.text):
                        break
                    self.agent.step('n')
                self.agent.step('y', gen())

            if is_bag_of_tricks:
                assert item.content is None
                raise AgentPanic('bag of tricks bites')

            if item in self.items.all_items and item.comment != item.container_id:
                self.call_item(item, item.container_id)

            if item.content is None:
                assert item.container_id is not None
                assert item.container_id not in self.item_manager.container_contents
                self.item_manager.container_contents[item.container_id] = content
                item.content = content

            # TODO: make it more elegant
            if len(item.glyphs) == 1 and item.glyphs[0] not in self.item_manager._is_not_bag_of_tricks:
                self.item_manager._is_not_bag_of_tricks.add(item.glyphs[0])
                self.item_manager.update_possible_objects(item)

    def _select_items_in_popup(self, items, counts=None):
        assert counts is None or len(counts) == len(items)
        items = list(items)
        while 1:
            if not self.agent.single_popup:
                raise AgentPanic('no popup, but some items were not selected yet')
            for line_i in range(len(self.agent.single_popup)):
                line = self.agent.single_popup[line_i]
                if line[1:4] != ' - ':
                    continue

                for item in items:
                    if item.text != line[4:]:
                        continue

                    i = items.index(item)
                    letter = line[0]

                    if counts is not None and counts[i] != item.count:
                        yield from str(counts[i])
                    yield letter

                    items.pop(i)
                    if counts is not None:
                        count = counts.pop(i)
                    else:
                        count = None
                    break

                if not items:
                    yield '\r'
                    return

            yield ' '
        assert not items

    def get_items_below_me(self, assume_appropriate_message=False):
        with self.agent.panic_if_position_changes():
            with self.agent.atom_operation():
                if not assume_appropriate_message:
                    self.agent.step(A.Command.LOOK)
                elif 'Things that are here:' in self.agent.popup or \
                        re.search('There are (several|many) objects here\.', self.agent.message):
                    # LOOK is necessary even when 'Things that are here' popup is present for some very rare cases
                    self.agent.step(A.Command.LOOK)

                if 'Something is ' in self.agent.message and 'You read: "' in self.agent.message:
                    index = self.agent.message.index('You read: "') + len('You read: "')
                    assert '"' in self.agent.message[index:]
                    engraving = self.agent.message[index: index + self.agent.message[index:].index('"')]
                    self.engraving_below_me = engraving
                else:
                    self.engraving_below_me = ''

                if 'Things that are here:' not in self.agent.popup and 'There is ' not in '\n'.join(self.agent.popup):
                    if 'You see no objects here.' in self.agent.message:
                        items = []
                        letters = []
                    elif 'You see here ' in self.agent.message:
                        item_str = self.agent.message[self.agent.message.index('You see here ') + len('You see here '):]
                        item_str = item_str[:item_str.index('.')]
                        items = [self.item_manager.get_item_from_text(item_str,
                                                                      position=(*self.agent.current_level().key(),
                                                                                self.agent.blstats.y,
                                                                                self.agent.blstats.x))]
                        letters = [None]
                    else:
                        items = []
                        letters = []
                else:
                    self.agent.step(A.Command.PICKUP)  # FIXME: parse LOOK output, add this fragment to pickup method
                    if 'Pick up what?' not in self.agent.popup:
                        if 'You cannot reach the bottom of the pit.' in self.agent.message or \
                                'You cannot reach the bottom of the abyss.' in self.agent.message or \
                                'You cannot reach the floor.' in self.agent.message or \
                                'There is nothing here to pick up.' in self.agent.message or \
                                ' solidly fixed to the floor.' in self.agent.message or \
                                'You read:' in self.agent.message or \
                                "You don't see anything in here to pick up." in self.agent.message or \
                                'You cannot reach the ground.' in self.agent.message or \
                                "You don't feel anything in here to pick up." in self.agent.message:
                            items = []
                            letters = []
                        else:
                            assert 0, (self.agent.message, self.agent.popup)
                    else:
                        lines = self.agent.popup[self.agent.popup.index('Pick up what?') + 1:]
                        category = None
                        items = []
                        letters = []
                        for line in lines:
                            if line in self._name_to_category:
                                category = self._name_to_category[line]
                                continue
                            assert line[1:4] == ' - ', line
                            letter, line = line[0], line[4:]
                            letters.append(letter)
                            items.append(self.item_manager.get_item_from_text(line, category,
                                                                              position=(
                                                                                  *self.agent.current_level().key(),
                                                                                  self.agent.blstats.y,
                                                                                  self.agent.blstats.x)))

                self.items_below_me = items
                self.letters_below_me = letters
                return items

    def pickup(self, items, counts=None):
        # TODO: if polyphormed, sometimes 'You are physically incapable of picking anything up.'
        if isinstance(items, Item):
            items = [items]
            if counts is not None:
                counts = [counts]
        if counts is None:
            counts = [i.count for i in items]
        assert len(items) > 0
        assert all(map(lambda item: item in self.items_below_me, items))
        assert len(counts) == len(items)
        assert sum(counts) > 0 and all((0 <= c <= i.count for c, i in zip(counts, items)))

        letters = [self.letters_below_me[self.items_below_me.index(item)] for item in items]
        screens = [max(self.letters_below_me[:self.items_below_me.index(item) + 1].count('a') - 1, 0) for item in items]

        with self.panic_if_items_below_me_change():
            self.get_items_below_me()

        one_item = len(self.items_below_me) == 1
        with self.agent.atom_operation():
            if one_item:
                assert all((s in [0, None] for s in screens))
                self.agent.step(A.Command.PICKUP)
                drop_count = items[0].count - counts[0]
            else:
                text = ' '.join((
                    ''.join([(str(count) if item.count != count else '') + letter
                             for letter, item, count, screen in zip(letters, items, counts, screens)
                             if count != 0 and screen == current_screen])
                    for current_screen in range(max(screens) + 1)))
                self.agent.step(A.Command.PICKUP, iter(list(text) + [A.MiscAction.MORE]))

            while re.search('You have [a-z ]+ lifting ', self.agent.message) and \
                    'Continue?' in self.agent.message:
                self.agent.type_text('y')
            if one_item and drop_count:
                letter = re.search(r'([a-zA-Z$]) - ', self.agent.message)
                assert letter is not None, self.agent.message
                letter = letter[1]

        if one_item and drop_count:
            self.drop(self.items.all_items[self.items.all_letters.index(letter)], drop_count, smart=False)

        self.get_items_below_me()

        return True

    def drop(self, items, counts=None, smart=True):
        if smart:
            items = self.move_to_inventory(items)

        if isinstance(items, Item):
            items = [items]
            if counts is not None:
                counts = [counts]
        if counts is None:
            counts = [i.count for i in items]
        assert all(map(lambda x: isinstance(x, (int, np.int32, np.int64)), counts)), list(map(type, counts))
        assert len(items) > 0
        assert all(map(lambda item: item in self.items.all_items, items))
        assert len(counts) == len(items)
        assert sum(counts) > 0 and all((0 <= c <= i.count for c, i in zip(counts, items)))

        letters = [self.items.all_letters[self.items.all_items.index(item)] for item in items]
        texts_to_type = [(str(count) if item.count != count else '') + letter
                         for letter, item, count in zip(letters, items, counts) if count != 0]

        if all((not i.can_be_dropped_from_inventory() for i in items)):
            return False

        def key_gen():
            if 'Drop what type of items?' in '\n'.join(self.agent.single_popup):
                yield 'a'
                yield A.MiscAction.MORE
            assert 'What would you like to drop?' in '\n'.join(self.agent.single_popup), \
                (self.agent.single_message, self.agent.single_popup)
            i = 0
            while texts_to_type:
                for text in list(texts_to_type):
                    letter = text[-1]
                    if f'{letter} - ' in '\n'.join(self.agent.single_popup):
                        yield from text
                        texts_to_type.remove(text)

                if texts_to_type:
                    yield A.TextCharacters.SPACE
                    i += 1

                assert i < 100, ('infinite loop', texts_to_type, self.agent.message)
            yield A.MiscAction.MORE

        with self.agent.atom_operation():
            self.agent.step(A.Command.DROPTYPE, key_gen())
        self.get_items_below_me()

        return True

    def move_to_inventory(self, items):
        # all items in self.items will be updated!

        if not isinstance(items, list):
            is_list = False
            items = [items]
        else:
            is_list = True

        moved_items = {item for item in items if item in self.items.all_items}

        if len(moved_items) != len(items):
            with self.agent.atom_operation():
                its = list(filter(lambda i: i in self.items_below_me, items))
                if its:
                    moved_items = moved_items.union(its)
                    self.pickup(its)
                for container in chain(self.items_below_me, self.items):
                    if container.is_container():
                        its = list(filter(lambda i: i in container.content.items, items))
                        if its:
                            moved_items = moved_items.union(its)
                            self.use_container(container, items_to_take=its, items_to_put=[])

                assert moved_items == set(items), ('TODO: nested containers', moved_items, items)

            # TODO: HACK
            self.agent.last_observation = self.agent.last_observation.copy()
            for key in ['inv_strs', 'inv_oclasses', 'inv_glyphs', 'inv_letters']:
                self.agent.last_observation[key] = self.agent._observation[key].copy()
            self.items.update(force=True)

            ret = []
            for item in items:
                ret.append(find_equivalent_item(item, filter(lambda i: i not in ret, self.items.all_items)))
        else:
            ret = items

        if not is_list:
            assert len(ret) == 1
            return ret[0]
        return ret

    def call_item(self, item, name):
        assert item in self.items.all_items, item
        letter = self.items.get_letter(item)
        with self.agent.atom_operation():
            self.agent.step(A.Command.CALL, iter(f'i{letter}#{name}\r'))
        return True

    def quaff(self, item, smart=True):
        return self.eat(item, quaff=True, smart=smart)

    def eat(self, item, quaff=False, smart=True):
        if smart:
            if not quaff and item in self.items_below_me:
                with self.agent.atom_operation():
                    self.agent.step(A.Command.EAT)
                    while '; eat it? [ynq]' in self.agent.message or \
                            '; eat one? [ynq]' in self.agent.message:
                        if f'{item.text} here; eat it? [ynq]' in self.agent.message or \
                                f'{item.text} here; eat one? [ynq]' in self.agent.message:
                            self.agent.type_text('y')
                            return True
                        self.agent.type_text('n')
                    # if "What do you want to eat?" in self.agent.message or \
                    #         "You don't have anything to eat." in self.agent.message:
                    raise AgentPanic('no such food is lying here')
                    assert 0, self.agent.message

            # TODO: eat directly from ground if possible
            item = self.move_to_inventory(item)

        assert item in self.items.all_items, item or item in self.items_below_me
        letter = self.items.get_letter(item)
        with self.agent.atom_operation():
            if quaff:
                def text_gen():
                    if self.agent.message.startswith('Drink from the fountain?'):
                        yield 'n'

                self.agent.step(A.Command.QUAFF, text_gen())
            else:
                self.agent.step(A.Command.EAT)
            if item in self.items.all_items:
                while re.search('There (is|are)[a-zA-Z0-9- ]* here; eat (it|one)\?', self.agent.message):
                    self.agent.type_text('n')
                self.agent.type_text(letter)
                return True

            elif item in self.items_below_me:
                while ' eat it? [ynq]' in self.agent.message or \
                        ' eat one? [ynq]' in self.agent.message:
                    if item.text in self.agent.message:
                        self.type_text('y')
                        return True
                if "What do you want to eat?" in self.agent.message or \
                        "You don't have anything to eat." in self.agent.message:
                    raise AgentPanic('no food is lying here')

                assert 0, self.agent.message

        assert 0

    ######## STRATEGIES helpers

    def get_best_melee_weapon(self, items=None, *, return_dps=False, allow_unknown_status=False):
        if self.agent.character.role == Character.MONK:
            return None

        if items is None:
            items = self.items
        # select the best
        best_item = None
        best_dps = utils.calc_dps(*self.agent.character.get_melee_bonus(None, large_monster=False))
        for item in flatten_items(items):
            if item.is_weapon() and \
                    (item.status in [Item.UNCURSED, Item.BLESSED] or
                     (allow_unknown_status and item.status == Item.UNKNOWN)):
                to_hit, dmg = self.agent.character.get_melee_bonus(item, large_monster=False)
                dps = utils.calc_dps(to_hit, dmg)
                # dps = item.get_dps(large_monster=False)  # TODO: what about monster size
                if best_dps < dps:
                    best_dps = dps
                    best_item = item
        if return_dps:
            return best_item, best_dps
        return best_item

    def get_ranged_combinations(self, items=None, throwing=True, allow_best_melee=False, allow_wielded_melee=False,
                                allow_unknown_status=False, additional_ammo=[]):
        if items is None:
            items = self.items
        items = flatten_items(items)
        launchers = [i for i in items if i.is_launcher()]
        ammo_list = [i for i in items if i.is_fired_projectile()]
        valid_combinations = []

        # TODO: should this condition be used here
        if any(l.equipped and l.status == Item.CURSED for l in launchers):
            launchers = [l for l in launchers if l.equipped]

        for launcher in launchers:
            for ammo in ammo_list + additional_ammo:
                if ammo.is_fired_projectile(launcher):
                    if launcher.status in [Item.UNCURSED, Item.BLESSED] or \
                            (allow_unknown_status and launcher.status == Item.UNKNOWN):
                        valid_combinations.append((launcher, ammo))

        if throwing:
            best_melee_weapon = None
            if not allow_best_melee:
                best_melee_weapon = self.get_best_melee_weapon()
            wielded_melee_weapon = None
            if not allow_wielded_melee:
                wielded_melee_weapon = self.items.main_hand
            valid_combinations.extend([(None, i) for i in items
                                       if i.is_thrown_projectile()
                                       and i != best_melee_weapon and i != wielded_melee_weapon])

        return valid_combinations

    def get_best_ranged_set(self, items=None, *, throwing=True, allow_best_melee=False,
                            allow_wielded_melee=False,
                            return_dps=False, allow_unknown_status=False, additional_ammo=[]):
        if items is None:
            items = self.items
        items = flatten_items(items)

        best_launcher, best_ammo = None, None
        best_dps = -float('inf')
        for launcher, ammo in self.get_ranged_combinations(items, throwing, allow_best_melee, allow_wielded_melee,
                                                           allow_unknown_status, additional_ammo):
            to_hit, dmg = self.agent.character.get_ranged_bonus(launcher, ammo)
            dps = utils.calc_dps(to_hit, dmg)
            if dps > best_dps:
                best_launcher, best_ammo, best_dps = launcher, ammo, dps
        if return_dps:
            return best_launcher, best_ammo, best_dps
        return best_launcher, best_ammo

    def get_best_armorset(self, items=None, *, return_ac=False, allow_unknown_status=False):
        if items is None:
            items = self.items
        items = flatten_items(items)

        best_items = [None] * O.ARM_NUM
        best_ac = [None] * O.ARM_NUM
        for item in items:
            if not item.is_armor() or not item.is_unambiguous():
                continue

            # TODO: consider other always allowed items than dragon hide
            is_dragonscale_armor = item.object.metal == O.DRAGON_HIDE

            allowed_statuses = [Item.UNCURSED, Item.BLESSED] + ([Item.UNKNOWN] if allow_unknown_status else [])
            if item.status not in allowed_statuses and not is_dragonscale_armor:
                continue

            slot = item.object.sub
            ac = item.get_ac()

            if self.agent.character.role == Character.MONK and slot == O.ARM_SUIT:
                continue

            if best_ac[slot] is None or best_ac[slot] > ac:
                best_ac[slot] = ac
                best_items[slot] = item

        if return_ac:
            return best_items, best_ac
        return best_items

    ######## LOW-LEVEL STRATEGIES

    def gather_items(self):
        return (
            self.pickup_and_drop_items()
                .before(self.check_containers())
                .before(self.wear_best_stuff())
                .before(self.wand_engrave_identify())
                .before(self.go_to_unchecked_containers())
                .before(self.check_items()
                        .before(self.go_to_item_to_pickup()).repeat().every(5)
                        .preempt(self.agent, [
                self.pickup_and_drop_items(),
                self.check_containers(),
            ])).repeat()
        )

    @utils.debug_log('inventory.arrange_items')
    @Strategy.wrap
    def arrange_items(self):
        yielded = False

        if self.agent.character.prop.polymorph:
            # TODO: only handless
            yield False

        while 1:
            items_below_me = list(filter(lambda i: i.shop_status == Item.NOT_SHOP, flatten_items(self.items_below_me)))
            forced_items = list(filter(lambda i: not i.can_be_dropped_from_inventory(), flatten_items(self.items)))
            assert all((item in self.items.all_items for item in forced_items))
            free_items = list(filter(lambda i: i.can_be_dropped_from_inventory(),
                                     flatten_items(sorted(self.items, key=lambda x: x.text))))
            all_items = free_items + items_below_me

            item_split = self.agent.global_logic.item_priority.split(
                all_items, forced_items, self.agent.character.carrying_capacity)

            assert all((container is None or container in self.items_below_me or container in self.items.all_items or \
                        (sum(item_split[container]) == 0 and not container.content.items)
                        for container in item_split)), 'TODO: nested containers'

            cont = False

            # put into containers
            for container in item_split:
                if container is not None:
                    counts = item_split[container]
                    indices = [i for i, item in enumerate(all_items) if item in self.items.all_items and counts[i] > 0]
                    if not indices:
                        continue
                    if not yielded:
                        yielded = True
                        yield True

                    self.use_container(container, [all_items[i] for i in indices], [],
                                       items_to_put_counts=[counts[i] for i in indices])
                    cont = True
                    break
            if cont:
                continue

            # drop on ground
            counts = item_split[None]
            indices = [i for i, item in enumerate(free_items) if
                       item in self.items.all_items and counts[i] != item.count]
            if indices:
                if not yielded:
                    yielded = True
                    yield True
                assert self.drop([free_items[i] for i in indices], [free_items[i].count - counts[i] for i in indices],
                                 smart=False)
                continue

            # take from container
            for container in all_items:
                if not container.is_container():
                    continue

                if container in item_split:
                    counts = item_split[container]
                    indices = [i for i, item in enumerate(all_items) if
                               item in container.content.items and counts[i] != item.count]
                    items_to_take_counts = [all_items[i].count - counts[i] for i in indices]
                else:
                    counts = np.array(list(item_split.values())).sum(0)
                    indices = [i for i, item in enumerate(all_items) if
                               item in container.content.items and counts[i] != 0]
                    items_to_take_counts = [counts[i] for i in indices]

                if not indices:
                    continue
                if not yielded:
                    yielded = True
                    yield True

                assert self.items.free_slots() > 0
                indices = indices[:self.items.free_slots()]

                self.use_container(container, [], [all_items[i] for i in indices],
                                   items_to_take_counts=items_to_take_counts)
                cont = True
                break
            if cont:
                continue

            # pick up from ground
            to_pickup = np.array([counts[len(free_items):] for counts in item_split.values()]).sum(0)
            assert len(to_pickup) == len(items_below_me)
            indices = [i for i, item in enumerate(items_below_me) if to_pickup[i] > 0 and item in self.items_below_me]
            if len(indices) > 0:
                assert self.items.free_slots() > 0
                indices = indices[:self.items.free_slots()]
                if not yielded:
                    yielded = True
                    yield True
                assert self.pickup([items_below_me[i] for i in indices], [to_pickup[i] for i in indices])
                continue

            break

        for container in item_split:
            for item, count in zip(all_items, item_split[container]):
                assert count == 0 or count == item.count
                assert count == 0 or item in (
                    container.content.items if container is not None else self.items.all_items)

        if not yielded:
            yield False

    def _determine_possible_wands(self, message, item):

        wand_regex = '[a-zA-Z ]+'
        floor_regex = '[a-zA-Z]+'
        mapping = {
            f"The engraving on the {floor_regex} vanishes!": ['cancellation', 'teleportation', 'make invisible'],
            # TODO?: cold,  # (if the existing engraving is a burned one)

            "A few ice cubes drop from the wand.": ['cold'],
            f"The bugs on the {floor_regex} stop moving": ['death', 'sleep'],
            f"This {wand_regex} is a wand of digging!": ['digging'],
            "Gravel flies up from the floor!": ['digging'],
            f"This {wand_regex} is a wand of fire!": ['fire'],
            "Lightning arcs from the wand. You are blinded by the flash!": ['lighting'],
            f"This {wand_regex} is a wand of lightning!": ['lightning'],
            f"The {floor_regex} is riddled by bullet holes!": ['magic missile'],
            f'The engraving now reads:': ['polymorph'],
            f"The bugs on the {floor_regex} slow down!": ['slow monster'],
            f"The bugs on the {floor_regex} speed up!": ['speed monster'],
            "The wand unsuccessfully fights your attempt to write!": ['striking'],

            # activated effects:
            "A lit field surrounds you!": ['light'],
            "You may wish for an object.": ['wishing'],
            "You feel self-knowledgeable...": ['enlightenment']  # TODO: parse the effect
            # TODO: "The wand is too worn out to engrave.": [None],  # wand is exhausted
        }

        for msg, wand_types in mapping.items():
            res = re.findall(msg, message)
            if len(res) > 0:
                assert len(res) == 1
                return [O.from_name(w, nh.WAND_CLASS) for w in wand_types]

        # TODO: "wand is cancelled (x:-1)" ?
        # TODO: "secret door detection self-identifies if secrets are detected" ?

        res = re.findall(f'Your {wand_regex} suddenly explodes!', self.agent.message)
        if len(res) > 0:
            assert len(res) == 1
            return None

        res = re.findall('The wand is too worn out to engrave.', self.agent.message)
        if len(res) > 0:
            assert len(res) == 1
            self.agent.inventory.call_item(item, 'EMPT')
            return None

        res = re.findall(f'{wand_regex} glows, then fades.', self.agent.message)
        if len(res) > 0:
            assert len(res) == 1
            return [p for p in O.possibilities_from_glyph(item.glyphs[0])
                    if p.name not in ['light', 'wishing']]
            # TODO: wiki says this:
            # return [O.from_name('opening', nh.WAND_CLASS),
            #         O.from_name('probing', nh.WAND_CLASS),
            #         O.from_name('undead turning', nh.WAND_CLASS),
            #         O.from_name('nothing', nh.WAND_CLASS),
            #         O.from_name('secret door detection', nh.WAND_CLASS),
            #         ]

        assert 0, message

    @utils.debug_log('inventory.wand_engrave_identify')
    @Strategy.wrap
    def wand_engrave_identify(self):
        if self.agent.character.prop.polymorph:
            yield False  # TODO: only for handless monsters (which cannot write)

        self.skip_engrave_counter -= 1
        if self.agent.character.prop.blind or self.skip_engrave_counter > 0:
            yield False
            return
        yielded = False
        for item in self.agent.inventory.items:
            if not isinstance(item.objs[0], O.Wand):
                continue
            if item.is_unambiguous():
                continue
            if self.agent.current_level().objects[self.agent.blstats.y, self.agent.blstats.x] not in G.FLOOR:
                continue
            if item.glyphs[0] in self.item_manager._already_engraved_glyphs:
                continue
            if len(item.glyphs) > 1:
                continue
            if item.comment == 'EMPT':
                continue

            if not yielded:
                yield True
            yielded = True
            self.skip_engrave_counter = 8

            with self.agent.atom_operation():
                wand_types = self._engrave_single_wand(item)

                if wand_types is None:
                    # there is a problem with engraving on this tile
                    continue

                self.item_manager._glyph_to_possible_wand_types[item.glyphs[0]] = wand_types
                self.item_manager._already_engraved_glyphs.add(item.glyphs[0])
                self.item_manager.possible_objects_from_glyph(item.glyphs[0])

            # uncomment for debugging (stopping when there is a new wand being identified)
            # print(len(self.item_manager.possible_objects_from_glyph(item.glyphs[0])))
            # print(self.item_manager._glyph_to_possible_wand_types)
            # input('==================3')

        if yielded:
            self.agent.inventory.items.update(force=True)

        if not yielded:
            yield False

    def _engrave_single_wand(self, item):
        """ Returns possible objects or None if current tile not suitable for identification."""

        def msg():
            return self.agent.message

        def smsg():
            return self.agent.single_message

        self.agent.step(A.Command.LOOK)
        if msg() != 'You see no objects here.':
            return None
        # if 'written' in msg() or 'engraved' in msg() or 'see' not in msg() or 'read' in msg():
        #     return None

        skip_engraving = [False]

        def action_generator():
            assert smsg().startswith('What do you want to write with?'), smsg()
            yield '-'
            # if 'Do you want to add to the current engraving' in smsg():
            #     yield 'q'
            #     assert smsg().strip() == 'Never mind.', smsg()
            #     skip_engraving[0] = True
            #     return
            if smsg().startswith('You wipe out the message that was written'):
                yield ' '
                skip_engraving[0] = True
                return
            if smsg().startswith('You cannot wipe out the message that is burned into the floor here.'):
                skip_engraving[0] = True
                return
            assert smsg().startswith('You write in the dust with your fingertip.'), smsg()
            yield ' '
            assert smsg().startswith('What do you want to write in the dust here?'), smsg()
            yield 'x'
            assert smsg().startswith('What do you want to write in the dust here?'), smsg()
            yield '\r'

        for _ in range(5):
            # write 'x' with finger in the dust
            self.agent.step(A.Command.ENGRAVE, additional_action_iterator=iter(action_generator()))

            if skip_engraving[0]:
                assert msg().strip().endswith('Never mind.') \
                       or 'You cannot wipe out the message that is burned into the floor here.' in msg(), msg()
                return None

            # this is usually true, but something unrelated like: "You hear crashing rock." may happen
            # assert msg().strip() in '', msg()

            # check if the written 'x' is visible when looking
            self.agent.step(A.Command.LOOK)
            if 'Something is written here in the dust.' in msg() \
                    and 'You read: "x"' in msg():
                break
            else:
                # this is usually true, but something unrelated like:
                #   "There is a doorway here.  Something is written here in the dust. You read: "4".
                #    You see here a giant rat corpse."
                # may happen
                # assert "You see no objects here" in msg(), msg()
                return None
        else:
            assert 0, msg()

        # try engraving with the wand
        letter = self.agent.inventory.items.get_letter(item)
        possible_wand_types = []

        def action_generator():
            assert smsg().startswith('What do you want to write with?'), smsg()
            yield letter
            if 'Do you want to add to the current engraving' in smsg():
                self.agent.type_text('y')
                # assert 'You add to the writing in the dust with' in smsg(), smsg()
                # self.agent.type_text(' ')
            r = self._determine_possible_wands(smsg(), item)
            if r is not None:
                possible_wand_types.extend(r)
            else:
                # wand exploded
                skip_engraving[0] = True

        self.agent.step(A.Command.ENGRAVE, additional_action_iterator=iter(action_generator()))

        if skip_engraving[0]:
            return None

        if 'Do you want to add to the current engraving' in smsg():
            self.agent.type_text('q')
            assert smsg().strip() == 'Never mind.', smsg()

        return possible_wand_types

    @utils.debug_log('inventory.wear_best_stuff')
    @Strategy.wrap
    def wear_best_stuff(self):
        yielded = False
        while 1:
            best_armorset = self.get_best_armorset()

            # TODO: twoweapon
            for slot, name in [(O.ARM_SHIELD, 'off_hand'), (O.ARM_HELM, 'helm'), (O.ARM_GLOVES, 'gloves'),
                               (O.ARM_BOOTS, 'boots'), (O.ARM_SHIRT, 'shirt'), (O.ARM_SUIT, 'suit'),
                               (O.ARM_CLOAK, 'cloak')]:
                if best_armorset[slot] == getattr(self.items, name) or \
                        (getattr(self.items, name) is not None and getattr(self.items, name).status == Item.CURSED):
                    continue
                additional_cond = True
                if slot == O.ARM_SHIELD:
                    additional_cond &= self.items.main_hand is None or not self.items.main_hand.objs[0].bi
                if slot == O.ARM_GLOVES:
                    additional_cond &= self.items.main_hand is None or self.items.main_hand.status != Item.CURSED
                if slot == O.ARM_SHIRT or slot == O.ARM_SUIT:
                    additional_cond &= self.items.cloak is None or self.items.cloak.status != Item.CURSED
                if slot == O.ARM_SHIRT:
                    additional_cond &= self.items.suit is None or self.items.suit.status != Item.CURSED

                if additional_cond:
                    if not yielded:
                        yielded = True
                        yield True
                    if (slot == O.ARM_SHIRT or slot == O.ARM_SUIT) and self.items.cloak is not None:
                        self.takeoff(self.items.cloak)
                        break
                    if slot == O.ARM_SHIRT and self.items.suit is not None:
                        self.takeoff(self.items.suit)
                        break
                    if getattr(self.items, name) is not None:
                        self.takeoff(getattr(self.items, name))
                        break
                    assert best_armorset[slot] is not None
                    self.wear(best_armorset[slot])
                    break
            else:
                break

        if not yielded:
            yield False

    @utils.debug_log('inventory.check_items')
    @Strategy.wrap
    def check_items(self):
        mask = utils.isin(self.agent.glyphs, G.OBJECTS, G.BODIES, G.STATUES)
        if not mask.any():
            yield False

        dis = self.agent.bfs()

        mask &= self.agent.current_level().item_count == 0
        if not mask.any():
            yield False

        mask &= dis > 0
        if not mask.any():
            yield False
        yield True

        nonzero_y, nonzero_x = (mask & (dis == dis[mask].min())).nonzero()
        i = self.agent.rng.randint(len(nonzero_y))
        target_y, target_x = nonzero_y[i], nonzero_x[i]

        with self.agent.env.debug_tiles(mask, color=(255, 0, 0, 128)):
            self.agent.go_to(target_y, target_x, debug_tiles_args=dict(color=(255, 0, 255), is_path=True))

    @utils.debug_log('inventory.go_to_unchecked_containers')
    @Strategy.wrap
    def go_to_unchecked_containers(self):
        mask = self.agent.current_level().item_count != 0
        if not mask.any():
            yield False

        dis = self.agent.bfs()
        mask &= dis > 0
        if not mask.any():
            yield False

        for y, x in zip(*mask.nonzero()):
            for item in self.agent.current_level().items[y, x]:
                if not item.is_possible_container():
                    mask[y, x] = False

        if not mask.any():
            yield False
        yield True

        nonzero_y, nonzero_x = (mask & (dis == dis[mask].min())).nonzero()
        i = self.agent.rng.randint(len(nonzero_y))
        target_y, target_x = nonzero_y[i], nonzero_x[i]

        with self.agent.env.debug_tiles(mask, color=(255, 0, 0, 128)):
            self.agent.go_to(target_y, target_x, debug_tiles_args=dict(color=(255, 0, 255), is_path=True))

    @utils.debug_log('inventory.check_containers')
    @Strategy.wrap
    def check_containers(self):
        yielded = False
        for item in self.agent.inventory.items_below_me:
            if item.is_possible_container():
                if not yielded:
                    yielded = True
                    yield True
                if item.is_chest() and not (item.is_unambiguous() and item.object.name == 'ice box'):
                    fail_msg = self.agent.untrap_container_below_me()
                    if fail_msg is not None and check_if_triggered_container_trap(fail_msg):
                        raise AgentPanic('triggered trap while looting')
                self.check_container_content(item)
        if not yielded:
            yield False

    @utils.debug_log('inventory.go_to_item_to_pickup')
    @Strategy.wrap
    def go_to_item_to_pickup(self):
        level = self.agent.current_level()
        dis = self.agent.bfs()

        # TODO: free (no charge) items
        mask = ~level.shop_interior & (dis > 0)
        if not mask.any():
            yield False

        mask[mask] = self.agent.current_level().item_count[mask] != 0

        items = {}
        for y, x in sorted(zip(*mask.nonzero()), key=lambda p: dis[p]):
            for i in level.items[y, x]:
                assert i not in items
                items[i] = (y, x)

        if not items:
            yield False

        items = {i: pos for item, pos in items.items() for i in flatten_items([item])}

        free_items = list(filter(lambda i: i.can_be_dropped_from_inventory(), flatten_items(self.items)))
        forced_items = list(filter(lambda i: not i.can_be_dropped_from_inventory(), flatten_items(self.items)))
        item_split = self.agent.global_logic.item_priority.split(
            free_items + list(items.keys()), forced_items,
            self.agent.character.carrying_capacity)
        counts = np.array(list(item_split.values())).sum(0)

        counts = counts[len(free_items):]
        assert len(counts) == len(items)
        if sum(counts) == 0:
            yield False
        yield True

        for (i, _), c in sorted(zip(items.items(), counts), key=lambda x: dis[x[0][1]]):
            if c != 0:
                target_y, target_x = items[i]
                break
        else:
            assert 0

        with self.agent.env.debug_tiles([(y, x) for _, (y, x) in items.items()], color=(255, 0, 0, 128)):
            self.agent.go_to(target_y, target_x, debug_tiles_args=dict(color=(255, 0, 255), is_path=True))

    @utils.debug_log('inventory.pickup_and_drop_items')
    @Strategy.wrap
    def pickup_and_drop_items(self):
        # TODO: free (no charge) items
        self.item_manager.price_identification()
        if self.agent.current_level().shop_interior[self.agent.blstats.y, self.agent.blstats.x]:
            yield False
        if len(self.items_below_me) == 0:
            yield False

        yield from self.arrange_items().strategy()
