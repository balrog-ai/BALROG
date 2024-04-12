import functools
import re

import nle.nethack as nh
from nle.nethack import actions as A

from autoascend import objects as O, utils
from autoascend.character import Character
from autoascend.glyph import MON
from autoascend.item import Item


class ContainerContent:
    def __init__(self):
        self.reset()

    def reset(self):
        self.items = []
        self.locked = False

    def __iter__(self):
        return iter(self.items)

    def weight(self):
        if self.locked:
            return 100000
        return sum((item.weight() for item in self.items))


class ItemManager:
    def __init__(self, agent):
        self.agent = agent
        self.object_to_glyph = {}
        self.glyph_to_object = {}
        self._last_object_glyph_mapping_update_step = None
        self._glyph_to_price_range = {}

        self._is_not_bag_of_tricks = set()

        # the container content should be edited instead of creating new one if exists.
        # Item.content keeps reference to it
        self.container_contents = {}  # container_id -> ContainerContent
        self._last_container_identifier = 0

        self._glyph_to_possible_wand_types = {}
        self._already_engraved_glyphs = set()

    def on_panic(self):
        self.update_object_glyph_mapping()

    def update(self):
        if self._last_object_glyph_mapping_update_step is None or \
                self._last_object_glyph_mapping_update_step + 200 < self.agent.step_count:
            self.update_object_glyph_mapping()

    def update_object_glyph_mapping(self):
        with self.agent.atom_operation():
            self.agent.step(A.Command.KNOWN)
            for line in self.agent.popup:
                if line.startswith('*'):
                    assert line[1] == ' ' and line[-1] == ')' and line.count('(') == 1 and line.count(')') == 1, line
                    name = line[1: line.find('(')].strip()
                    desc = line[line.find('(') + 1: -1].strip()

                    n_objs, n_glyphs = ItemManager.parse_name(name)
                    d_glyphs = O.desc_to_glyphs(desc, O.get_category(n_objs[0]))
                    assert d_glyphs
                    if len(n_objs) == 1 and len(d_glyphs) == 1:
                        obj, glyph = n_objs[0], d_glyphs[0]

                        assert glyph not in self.glyph_to_object or self.glyph_to_object[glyph] == obj
                        self.glyph_to_object[glyph] = obj
                        assert obj not in self.object_to_glyph or self.object_to_glyph[obj] == glyph
                        self.object_to_glyph[obj] = glyph

            self._last_object_glyph_mapping_update_step = self.agent.step_count

    def _get_new_container_identifier(self):
        ret = self._last_container_identifier
        self._last_container_identifier += 1
        return str(ret)

    def _buy_price_identification(self):
        if self.agent.blstats.charisma <= 5:
            charisma_multiplier = 2
        elif self.agent.blstats.charisma <= 7:
            charisma_multiplier = 1.5
        elif self.agent.blstats.charisma <= 10:
            charisma_multiplier = 1 + 1 / 3
        elif self.agent.blstats.charisma <= 15:
            charisma_multiplier = 1
        elif self.agent.blstats.charisma <= 17:
            charisma_multiplier = 3 / 4
        elif self.agent.blstats.charisma <= 18:
            charisma_multiplier = 2 / 3
        else:
            charisma_multiplier = 1 / 2

        if (self.agent.character.role == Character.TOURIST and self.agent.blstats.experience_level < 15) or \
                (self.agent.inventory.items.shirt is not None and self.agent.inventory.items.suit is None and
                 self.agent.inventory.items.cloak is None) or \
                (self.agent.inventory.items.helm is not None and self.agent.inventory.items.helm.is_unambiguous() and
                 self.agent.inventory.items.helm.object == O.from_name('dunce cap')):
            dupe_multiplier = (4 / 3, 4 / 3)
        elif self.agent.inventory.items.helm is not None and \
                any(((obj == O.from_name('dunce cap')) for obj in self.agent.inventory.items.helm.objs)):
            dupe_multiplier = (4 / 3, 1)
        else:
            dupe_multiplier = (1, 1)

        for item in self.agent.inventory.items_below_me:
            if item.shop_status == Item.FOR_SALE and not item.is_unambiguous() and len(item.glyphs) == 1 and \
                    isinstance(item.objs[0], (O.Armor, O.Ring, O.Wand, O.Scroll, O.Potion, O.Tool)):
                # TODO: not an artifact
                # TODO: Base price of partly eaten food, uncursed water, and (x:-1) wands is 0.
                assert item.price % item.count == 0
                low = int(item.price / item.count / charisma_multiplier / (4 / 3) / dupe_multiplier[0])
                if isinstance(item.objs[0], (O.Weapon, O.Armor)):
                    low = 0  # +10 zorkmoids for each point of enchantment

                if low <= 5:
                    low = 0
                high = int(item.price / item.count / charisma_multiplier / dupe_multiplier[1]) + 1
                if item.glyphs[0] in self._glyph_to_price_range:
                    l, h = self._glyph_to_price_range[item.glyphs[0]]
                    low = max(low, l)
                    high = min(high, h)
                assert low <= high, (low, high)
                self._glyph_to_price_range[item.glyphs[0]] = (low, high)

                # update mapping for that object
                self.possible_objects_from_glyph(item.glyphs[0])

    def price_identification(self):
        if self.agent.character.prop.hallu:
            return
        if not self.agent.current_level().shop_interior[self.agent.blstats.y, self.agent.blstats.x]:
            return

        self._buy_price_identification()

    def update_possible_objects(self, item):
        possibilities_from_glyphs = set.union(*(set(self.possible_objects_from_glyph(glyph)) for glyph in item.glyphs))
        item.objs = [o for o in item.objs if o in possibilities_from_glyphs]
        assert len(item.objs)

    def get_item_from_text(self, text, category=None, glyph=None, *, position):
        # position acts as a container identifier if the container is not called. If the item is in inventory set it to None

        if self.agent.character.prop.hallu:
            glyph = None

        try:
            objs, glyphs, count, status, modifier, *args = \
                self.parse_text(text, category, glyph)
            category = O.get_category(objs[0])
        except:
            # TODO: when blind, it may not work as expected, e.g. "a shield", "a gem", "a potion", etc
            if self.agent.character.prop.blind:
                obj = O.from_name('unknown')
                glyphs = O.possible_glyphs_from_object(obj)
                return Item([obj], glyphs, text=text)
            raise

        possibilities_from_glyphs = set.union(*(set(self.possible_objects_from_glyph(glyph)) for glyph in glyphs))
        objs = [o for o in objs if o in possibilities_from_glyphs]
        assert len(objs), ([O.objects[g - nh.GLYPH_OBJ_OFF].desc for g in glyphs],
                           [o.name for o in possibilities_from_glyphs], text)

        if status == Item.UNKNOWN and (
                self.agent.character.role == Character.PRIEST or
                (modifier is not None and category not in [nh.ARMOR_CLASS, nh.RING_CLASS])):
            # TODO: amulets of yendor
            status = Item.UNCURSED

        old_objs = None
        old_glyphs = None
        while old_objs != objs or old_glyphs != glyphs:
            old_objs = objs
            old_glyphs = glyphs

            objs = [o for o in objs if o not in self.object_to_glyph or self.object_to_glyph[o] in glyphs]
            glyphs = [g for g in glyphs if g not in self.glyph_to_object or self.glyph_to_object[g] in objs]
            if len(objs) == 1 and len(glyphs) == 1:
                assert glyphs[0] not in self.glyph_to_object or self.glyph_to_object[glyphs[0]] == objs[0]
                self.glyph_to_object[glyphs[0]] = objs[0]
                assert objs[0] not in self.object_to_glyph or self.object_to_glyph[objs[0]] == glyphs[0]
                self.object_to_glyph[objs[0]] = glyphs[0]
            elif len(objs) == 1 and objs[0] in self.object_to_glyph:
                glyphs = [self.object_to_glyph[objs[0]]]
            elif len(glyphs) == 1 and glyphs[0] in self.glyph_to_object:
                objs = [self.glyph_to_object[glyphs[0]]]

        item = Item(objs, glyphs, count, status, modifier, *args, text)

        if item.is_possible_container() or item.is_container():
            if item.comment:
                identifier = item.comment
            else:
                if position is not None:
                    identifier = position
                else:
                    identifier = self._get_new_container_identifier()
            item.container_id = identifier
            if identifier in self.container_contents:
                item.content = self.container_contents[identifier]
                if len(item.glyphs) == 1 and item.glyphs[0] not in self._is_not_bag_of_tricks:
                    self._is_not_bag_of_tricks.add(item.glyphs[0])
                    self.update_possible_objects(item)

        # FIXME: it gives a better score. Implement it in item equipping
        if item.status == Item.UNKNOWN:
            item.status = Item.UNCURSED

        return item

    def possible_objects_from_glyph(self, glyph):
        """ Get possible objects and update glyph_to_object and object_to_glyph when object isn't unambiguous.
        """
        assert nh.glyph_is_object(glyph)
        if glyph in self.glyph_to_object:
            return [self.glyph_to_object[glyph]]

        objs = []
        for obj in O.possibilities_from_glyph(glyph):
            if obj in self.object_to_glyph:
                continue
            if glyph in self._glyph_to_price_range and hasattr(obj, 'cost') and \
                    not self._glyph_to_price_range[glyph][0] <= obj.cost <= self._glyph_to_price_range[glyph][1]:
                continue
            if glyph in self._glyph_to_possible_wand_types and obj not in self._glyph_to_possible_wand_types[glyph]:
                continue
            if glyph in self._is_not_bag_of_tricks and obj.name == 'bag of tricks':
                continue
            objs.append(obj)

        if len(objs) == 1:
            self.glyph_to_object[glyph] = objs[0]
            assert objs[0] not in self.object_to_glyph
            self.object_to_glyph[objs[0]] = glyph

            # update objects with have the same possible glyph
            for g in O.possible_glyphs_from_object(objs[0]):
                self.possible_objects_from_glyph(g)
        assert len(objs), (O.objects[glyph - nh.GLYPH_OBJ_OFF].desc, self._glyph_to_price_range[glyph])
        return objs

    @staticmethod
    @utils.copy_result
    @functools.lru_cache(1024 * 1024)
    def parse_text(text, category=None, glyph=None):
        assert glyph is None or nh.glyph_is_normal_object(glyph), glyph

        if category is None and glyph is not None:
            category = ord(nh.objclass(nh.glyph_to_obj(glyph)).oc_class)
        assert glyph is None or category is None or category == ord(nh.objclass(nh.glyph_to_obj(glyph)).oc_class)

        assert category not in [nh.RANDOM_CLASS]

        matches = re.findall(
            r'^(a|an|the|\d+)'
            r'( empty)?'
            r'( (cursed|uncursed|blessed))?'
            r'( (very |thoroughly )?(rustproof|poisoned|corroded|rusty|burnt|rotted|partly eaten|partly used|diluted|unlocked|locked|wet|greased))*'
            r'( ([+-]\d+))? '
            r"([a-zA-z0-9-!'# ]+)"
            r'( \(([0-9]+:[0-9]+|no charge)\))?'
            r'( \(([a-zA-Z0-9; ]+(, flickering|, gleaming|, glimmering)?[a-zA-Z0-9; ]*)\))?'
            r'( \((for sale|unpaid), (\d+ aum, )?((\d+)[a-zA-Z- ]+|no charge)\))?'
            r'$',
            text)
        assert len(matches) <= 1, text
        assert len(matches), (text, len(text))

        (
            count,
            effects1,
            _, status,
            effects2, _, _,
            _, modifier,
            name,
            _, uses,
            _, info, _,
            _, shop_status, _, _, shop_price
        ) = matches[0]
        # TODO: effects, uses

        if info in {'being worn', 'being worn; slippery', 'wielded', 'chained to you'} or info.startswith(
                'weapon in ') or \
                info.startswith('tethered weapon in '):
            equipped = True
            at_ready = False
        elif info in {'at the ready', 'in quiver', 'in quiver pouch', 'lit'}:
            equipped = False
            at_ready = True
        elif info in {'', 'alternate weapon; not wielded', 'alternate weapon; notwielded'}:
            equipped = False
            at_ready = False
        else:
            assert 0, info

        if not shop_price:
            shop_price = 0
        else:
            shop_price = int(shop_price)

        count = int({'a': 1, 'an': 1, 'the': 1}.get(count, count))
        status = {'': Item.UNKNOWN, 'cursed': Item.CURSED, 'uncursed': Item.UNCURSED, 'blessed': Item.BLESSED}[status]
        # TODO: should be uses -- but the score is better this way
        if uses and status == Item.UNKNOWN:
            status = Item.UNCURSED
        modifier = None if not modifier else {'+': 1, '-': -1}[modifier[0]] * int(modifier[1:])
        monster_id = None

        if ' containing ' in name:
            # TODO: use number of items for verification
            name = name[:name.index(' containing ')]

        comment = ''
        naming = ''
        if ' named ' in name:
            # TODO: many of these are artifacts
            naming = name[name.index(' named ') + len(' named '):]
            name = name[:name.index(' named ')]
            if '#' in naming:
                # all given names by the bot, starts with #
                pos = naming.index('#')
                comment = naming[pos + 1:]
                naming = naming[:pos]
            else:
                comment = ''

        if shop_status == '':
            shop_status = Item.NOT_SHOP
        elif shop_status == 'for sale':
            shop_status = Item.FOR_SALE
        elif shop_status == 'unpaid':
            shop_status = Item.UNPAID
        else:
            assert 0, shop_status

        if name in ['potion of holy water', 'potions of holy water']:
            name = 'potion of water'
            status = Item.BLESSED
        elif name in ['potion of unholy water', 'potions of unholy water']:
            name = 'potion of water'
            status = Item.CURSED
        elif name in ['gold piece', 'gold pieces']:
            status = Item.UNCURSED

        # TODO: pass to Item class instance
        if name.startswith('tin of ') or name.startswith('tins of '):
            mon_name = name[len('tin of '):].strip()
            if mon_name.endswith(' meat'):
                mon_name = mon_name[:-len(' meat')]
            if mon_name.startswith('a '):
                mon_name = mon_name[2:]
            if mon_name.startswith('an '):
                mon_name = mon_name[3:]
            if mon_name == 'spinach':
                monster_id = None
            else:
                monster_id = nh.glyph_to_mon(MON.from_name(mon_name))
            name = 'tin'
        elif name.endswith(' corpse') or name.endswith(' corpses'):
            mon_name = name[:name.index('corpse')].strip()
            if mon_name.startswith('a '):
                mon_name = mon_name[2:]
            if mon_name.startswith('an '):
                mon_name = mon_name[3:]
            monster_id = nh.glyph_to_mon(MON.from_name(mon_name))
            name = 'corpse'
        elif name.startswith('statue of ') or name.startswith('statues of ') or \
                name.startswith('historic statue of ') or name.startswith('historic statues of '):
            if name.startswith('historic'):
                mon_name = name[len('historic statue of '):].strip()
            else:
                mon_name = name[len('statue of '):].strip()
            if mon_name.startswith('a '):
                mon_name = mon_name[2:]
            if mon_name.startswith('an '):
                mon_name = mon_name[3:]
            monster_id = nh.glyph_to_mon(MON.from_name(mon_name))
            name = 'statue'
        elif name.startswith('figurine of ') or name.startswith('figurines of '):
            mon_name = name[len('figurine of '):].strip()
            if mon_name.startswith('a '):
                mon_name = mon_name[2:]
            if mon_name.startswith('an '):
                mon_name = mon_name[3:]
            monster_id = nh.glyph_to_mon(MON.from_name(mon_name))
            name = 'figurine'
        elif name in ['novel', 'paperback', 'paperback book']:
            name = 'spellbook of novel'
        elif name.endswith(' egg') or name.endswith(' eggs'):
            monster_id = nh.glyph_to_mon(MON.from_name(name[:-len(' egg')].strip()))
            name = 'egg'
        elif name == 'worm teeth':
            name = 'worm tooth'

        dmg_bonus, to_hit_bonus = None, None

        if naming:
            if naming in ['Hachi', 'Idefix', 'Slasher', 'Sirius']:  # pet names
                pass
            elif name == 'corpse':
                pass
            elif name == 'spellbook of novel':
                pass
            else:
                name = naming

        if name == 'Excalibur':
            name = 'long sword'
            dmg_bonus = 5.5  # 1d10
            to_hit_bonus = 3  # 1d5
        elif name == 'Mjollnir':
            name = 'war hammer'
            dmg_bonus = 12.5  # 1d24
            to_hit_bonus = 3  # 1d5
        elif name == 'Cleaver':
            name = 'battle-axe'
            dmg_bonus = 3.5  # 1d6
            to_hit_bonus = 2  # 1d3
        elif name == 'Sting':
            name = 'elven dagger'
            dmg_bonus = 2.5  # TODO: x2
            to_hit_bonus = 3  # 1d5
        elif name == 'Grimtooth':
            name = 'orcish dagger'
            dmg_bonus = 3.5  # +1d6
            to_hit_bonus = 1.5  # 1d2
        elif name in ['Sunsword', 'Frost Brand', 'Fire Brand', 'Demonbane', 'Giantslayer']:
            name = 'long sword'
            dmg_bonus = 5.5  # TODO: x2
            to_hit_bonus = 3  # 1d5
        elif name == 'Vorpal Blade':
            name = 'long sword'
            dmg_bonus = 1
            to_hit_bonus = 3  # 1d5
        elif name == 'Orcrist':
            name = 'elven broadsword'
            dmg_bonus = 6  # TODO: x2
            to_hit_bonus = 3  # 1d5
        elif name == 'Magicbane':
            name = 'athame'
            dmg_bonus = 7.25  # TODO
            to_hit_bonus = 2  # 1d3
        elif name in ['Grayswandir', 'Werebane']:
            name = 'silver saber'
            dmg_bonus = 15  # TODO: x2 + 1d20
            to_hit_bonus = 3  # 1d5
        elif name == 'Stormbringer':
            name = 'runesword'
            dmg_bonus = 6  # 1d2+1d8
            to_hit_bonus = 3  # 1d5
        elif name == 'Snickersnee':
            name = 'katana'
            dmg_bonus = 4.5  # 1d8
            to_hit_bonus = 1  # +1
        elif name == 'Trollsbane':
            name = 'morning star'
            dmg_bonus = 5.25  # TODO: x2
            to_hit_bonus = 3  # 1d5
        elif name == 'Ogresmasher':
            name = 'war hammer'
            dmg_bonus = 3  # TODO: x2
            to_hit_bonus = 3  # 1d5
        elif name == 'Dragonbane':
            name = 'broadsword'
            dmg_bonus = 4.75  # TODO: x2
            to_hit_bonus = 3  # 1d5

        objs, ret_glyphs = ItemManager.parse_name(name)
        assert category is None or category == O.get_category(objs[0]), (text, category, O.get_category(objs[0]))

        if glyph is not None:
            assert glyph in ret_glyphs
            pos = O.possibilities_from_glyph(glyph)
            if objs[0].name not in ['elven broadsword', 'runed broadsword']:
                assert all(map(lambda o: o in pos, objs)), (objs, pos)
            ret_glyphs = [glyph]
            objs = sorted(set(objs).intersection(O.possibilities_from_glyph(glyph)))

        return (
            objs, ret_glyphs, count, status, modifier, equipped, at_ready, monster_id, shop_status, shop_price,
            dmg_bonus, to_hit_bonus, naming, comment, uses
        )

    @staticmethod
    @utils.copy_result
    @functools.lru_cache(1024 * 256)
    def parse_name(name):
        if name == 'wakizashi':
            name = 'short sword'
        elif name == 'ninja-to':
            name = 'broadsword'
        elif name == 'nunchaku':
            name = 'flail'
        elif name == 'shito':
            name = 'knife'
        elif name == 'naginata':
            name = 'glaive'
        elif name == 'gunyoki':
            name = 'food ration'
        elif name == 'osaku':
            name = 'lock pick'
        elif name == 'tanko':
            name = 'plate mail'
        elif name in ['pair of yugake', 'yugake']:
            name = 'pair of leather gloves'
        elif name == 'kabuto':
            name = 'helmet'
        elif name in ['flint stone', 'flint stones']:
            name = 'flint'
        elif name in ['unlabeled scroll', 'unlabeled scrolls', 'blank paper']:
            name = 'scroll of blank paper'
        elif name == 'eucalyptus leaves':
            name = 'eucalyptus leaf'
        elif name == 'pair of lenses':
            name = 'lenses'
        elif name.startswith('small glob'):
            name = name[len('small '):]
        elif name == 'knives':
            name = 'knife'

        # object identified (look on names)
        obj_ids = set()
        prefixes = [
            ('scroll of ', nh.SCROLL_CLASS),
            ('scrolls of ', nh.SCROLL_CLASS),
            ('spellbook of ', nh.SPBOOK_CLASS),
            ('spellbooks of ', nh.SPBOOK_CLASS),
            ('ring of ', nh.RING_CLASS),
            ('rings of ', nh.RING_CLASS),
            ('wand of ', nh.WAND_CLASS),
            ('wands of ', nh.WAND_CLASS),
            ('', nh.AMULET_CLASS),
            ('potion of ', nh.POTION_CLASS),
            ('potions of ', nh.POTION_CLASS),
            ('', nh.GEM_CLASS),
            ('', nh.ARMOR_CLASS),
            ('pair of ', nh.ARMOR_CLASS),
            ('', nh.WEAPON_CLASS),
            ('', nh.TOOL_CLASS),
            ('', nh.FOOD_CLASS),
            ('', nh.COIN_CLASS),
            ('', nh.ROCK_CLASS),
        ]
        suffixes = [
            ('s', nh.GEM_CLASS),
            ('s', nh.WEAPON_CLASS),
            ('s', nh.TOOL_CLASS),
            ('s', nh.FOOD_CLASS),
            ('s', nh.COIN_CLASS),
        ]
        for i in range(nh.NUM_OBJECTS):
            for pref, c in prefixes:
                if ord(nh.objclass(i).oc_class) == c:
                    obj_name = nh.objdescr.from_idx(i).oc_name
                    if obj_name and name == pref + obj_name:
                        obj_ids.add(i)

            for suf, c in suffixes:
                if ord(nh.objclass(i).oc_class) == c:
                    obj_name = nh.objdescr.from_idx(i).oc_name
                    if obj_name and (name == obj_name + suf or \
                                     (c == nh.FOOD_CLASS and \
                                      name == obj_name.split()[0] + suf + ' ' + ' '.join(obj_name.split()[1:]))):
                        obj_ids.add(i)

        # object unidentified (look on descriptions)
        appearance_ids = set()
        prefixes = [
            ('scroll labeled ', nh.SCROLL_CLASS),
            ('scrolls labeled ', nh.SCROLL_CLASS),
            ('', nh.ARMOR_CLASS),
            ('pair of ', nh.ARMOR_CLASS),
            ('', nh.WEAPON_CLASS),
            ('', nh.TOOL_CLASS),
            ('', nh.FOOD_CLASS),
            ('', nh.BALL_CLASS),
        ]
        suffixes = [
            (' amulet', nh.AMULET_CLASS),
            (' amulets', nh.AMULET_CLASS),
            (' gem', nh.GEM_CLASS),
            (' gems', nh.GEM_CLASS),
            (' stone', nh.GEM_CLASS),
            (' stones', nh.GEM_CLASS),
            (' potion', nh.POTION_CLASS),
            (' potions', nh.POTION_CLASS),
            (' spellbook', nh.SPBOOK_CLASS),
            (' spellbooks', nh.SPBOOK_CLASS),
            (' ring', nh.RING_CLASS),
            (' rings', nh.RING_CLASS),
            (' wand', nh.WAND_CLASS),
            (' wands', nh.WAND_CLASS),
            ('s', nh.ARMOR_CLASS),
            ('s', nh.WEAPON_CLASS),
            ('s', nh.TOOL_CLASS),
            ('s', nh.FOOD_CLASS),
        ]

        for i in range(nh.NUM_OBJECTS):
            for pref, c in prefixes:
                if ord(nh.objclass(i).oc_class) == c:
                    obj_descr = nh.objdescr.from_idx(i).oc_descr
                    if obj_descr and name == pref + obj_descr:
                        appearance_ids.add(i)

            for suf, c in suffixes:
                if ord(nh.objclass(i).oc_class) == c:
                    obj_descr = nh.objdescr.from_idx(i).oc_descr
                    if obj_descr and name == obj_descr + suf:
                        appearance_ids.add(i)

        appearance_ids = list(appearance_ids)
        assert len(appearance_ids) == 0 or len({ord(nh.objclass(i).oc_class) for i in appearance_ids}), name

        # assert (len(obj_ids) > 0) ^ (len(appearance_ids) > 0), (name, obj_ids, appearance_ids)
        if (len(obj_ids) > 0) == (len(appearance_ids) > 0):
            return [O.from_name('unknown')], O.possible_glyphs_from_object(O.from_name('unknown'))

        if obj_ids:
            assert len(obj_ids) == 1, name
            obj_id = list(obj_ids)[0]
            objs = [O.objects[obj_id]]
            glyphs = [i for i in range(nh.GLYPH_OBJ_OFF, nh.NUM_OBJECTS + nh.GLYPH_OBJ_OFF)
                      if O.objects[i - nh.GLYPH_OBJ_OFF] is not None and objs[0] in O.possibilities_from_glyph(i)]
        else:
            glyphs = [obj_id + nh.GLYPH_OBJ_OFF for obj_id in appearance_ids]
            obj_id = list(appearance_ids)[0]
            glyph = obj_id + nh.GLYPH_OBJ_OFF
            objs = sorted(set.union(*[set(O.possibilities_from_glyph(i)) for i in glyphs]))
            assert name == 'runed broadsword' or \
                   all(map(lambda i: sorted(O.possibilities_from_glyph(i + nh.GLYPH_OBJ_OFF)) == objs, appearance_ids)), \
                name

        return objs, glyphs
