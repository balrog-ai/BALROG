import nle.nethack as nh

from autoascend import objects as O
from autoascend.glyph import MON, WEA


class Item:
    # beatitude
    UNKNOWN = 0
    CURSED = 1
    UNCURSED = 2
    BLESSED = 3

    # shop status
    NOT_SHOP = 0
    FOR_SALE = 1
    UNPAID = 2

    def __init__(self, objs, glyphs, count=1, status=UNKNOWN, modifier=None, equipped=False, at_ready=False,
                 monster_id=None, shop_status=NOT_SHOP, price=0, dmg_bonus=None, to_hit_bonus=None,
                 naming='', comment='', uses=None, text=None):
        assert isinstance(objs, list) and len(objs) >= 1
        assert isinstance(glyphs, list) and len(glyphs) >= 1 and all((nh.glyph_is_object(g) for g in glyphs))
        assert isinstance(count, int)

        self.objs = objs
        self.glyphs = glyphs
        self.count = count
        self.status = status
        self.modifier = modifier
        self.equipped = equipped
        self.uses = uses
        self.at_ready = at_ready
        self.monster_id = monster_id
        self.shop_status = shop_status
        self.price = price
        self.dmg_bonus = dmg_bonus
        self.to_hit_bonus = to_hit_bonus
        self.naming = naming
        self.comment = comment
        self.text = text

        self.content = None  # for checked containers it will be set after the constructor
        self.container_id = None  # for containers and possible containers it will be set after the constructor

        self.category = O.get_category(self.objs[0])
        assert all((ord(nh.objclass(nh.glyph_to_obj(g)).oc_class) == self.category for g in self.glyphs))

    def display_glyphs(self):
        if self.is_corpse():
            assert self.monster_id is not None
            return [nh.GLYPH_BODY_OFF + self.monster_id]
        if self.is_statue():
            assert self.monster_id is not None
            return [nh.GLYPH_STATUE_OFF + self.monster_id]
        return self.glyphs

    def is_unambiguous(self):
        return len(self.objs) == 1

    def can_be_dropped_from_inventory(self):
        return not (
                (isinstance(self.objs[0], (O.Weapon, O.WepTool)) and self.status == Item.CURSED and self.equipped) or
                (isinstance(self.objs[0], O.Armor) and self.equipped) or
                (self.is_unambiguous() and self.object == O.from_name('loadstone') and self.status == Item.CURSED) or
                (self.category == nh.BALL_CLASS and self.equipped)
        )

    def weight(self, with_content=True):
        return self.count * self.unit_weight(with_content=with_content)

    def unit_weight(self, with_content=True):
        if self.is_corpse():
            return MON.permonst(self.monster_id).cwt

        if self.is_possible_container():
            return 100000

        if self.objs[0] in [
            O.from_name("glob of gray ooze"),
            O.from_name("glob of brown pudding"),
            O.from_name("glob of green slime"),
            O.from_name("glob of black pudding"),
        ]:
            assert self.is_unambiguous()
            return 10000  # weight is unknown

        weight = max((obj.wt for obj in self.objs))

        if self.is_container() and with_content:
            weight += self.content.weight()  # TODO: bag of holding

        return weight

    @property
    def object(self):
        assert self.is_unambiguous()
        return self.objs[0]

    ######## WEAPON

    def is_weapon(self):
        return self.category == nh.WEAPON_CLASS

    def get_weapon_bonus(self, large_monster):
        assert self.is_weapon()

        hits = []
        dmgs = []
        for weapon in self.objs:
            dmg = WEA.expected_damage(weapon.damage_large if large_monster else weapon.damage_small)
            to_hit = 1 + weapon.hitbon
            if self.modifier is not None:
                dmg += max(0, self.modifier)
                to_hit += self.modifier

            dmg += 0 if self.dmg_bonus is None else self.dmg_bonus
            to_hit += 0 if self.to_hit_bonus is None else self.to_hit_bonus

            dmgs.append(dmg)
            hits.append(to_hit)

        # assume the worse
        return min(hits), min(dmgs)

    def is_launcher(self):
        if not self.is_weapon() or not self.is_unambiguous():
            return False

        return self.object.name in ['bow', 'elven bow', 'orcish bow', 'yumi', 'crossbow', 'sling']

    def is_fired_projectile(self, launcher=None):
        if not self.is_weapon() or not self.is_unambiguous():
            return False

        arrows = ['arrow', 'elven arrow', 'orcish arrow', 'silver arrow', 'ya']

        if launcher is None:
            return self.object.name in (arrows + ['crossbow bolt'])  # TODO: sling ammo
        else:
            launcher_name = launcher.object.name
            if launcher_name == 'crossbow':
                return self.object.name == 'crossbow bolt'
            elif launcher_name == 'sling':
                # TODO: sling ammo
                return False
            else:  # any bow
                assert launcher_name in ['bow', 'elven bow', 'orcish bow', 'yumi'], launcher_name
                return self.object.name in arrows

    def is_thrown_projectile(self):
        if not self.is_weapon() or not self.is_unambiguous():
            return False

        # TODO: boomerang
        # TODO: aklys, Mjollnir
        return self.object.name in \
               ['dagger', 'orcish dagger', 'dagger silver', 'athame dagger', 'elven dagger',
                'worm tooth', 'knife', 'stiletto', 'scalpel', 'crysknife',
                'dart', 'shuriken']

    def __str__(self):
        if self.text is not None:
            return self.text
        return (f'{self.count}_'
                f'{self.status if self.status is not None else ""}_'
                f'{self.modifier if self.modifier is not None else ""}_'
                f'{",".join(list(map(lambda x: x.name, self.objs)))}'
                )

    def __repr__(self):
        return str(self)

    ######## ARMOR

    def is_armor(self):
        return self.category == nh.ARMOR_CLASS

    def get_ac(self):
        assert self.is_armor()
        return self.object.ac - (self.modifier if self.modifier is not None else 0)

    ######## WAND

    def is_wand(self):
        return isinstance(self.objs[0], O.Wand)

    def is_beam_wand(self):
        if not self.is_wand():
            return False
        beam_wand_types = ['cancellation', 'locking', 'make invisible',
                           'nothing', 'opening', 'polymorph', 'probing', 'slow monster',
                           'speed monster', 'striking', 'teleportation', 'undead turning']
        beam_wand_types = [O.from_name(w, nh.WAND_CLASS) for w in beam_wand_types]
        for obj in self.objs:
            if obj not in beam_wand_types:
                return False
        return True

    def is_ray_wand(self):
        if not self.is_wand():
            return False
        ray_wand_types = ['cold', 'death', 'digging', 'fire', 'lightning', 'magic missile', 'sleep']
        ray_wand_types = [O.from_name(w, nh.WAND_CLASS) for w in ray_wand_types]
        for obj in self.objs:
            if obj not in ray_wand_types:
                return False
        return True

    def wand_charges_left(self, item):
        assert item.is_wand()

    def is_offensive_usable_wand(self):
        if len(self.objs) != 1:
            return False
        if not self.is_ray_wand():
            return False
        if self.uses == 'no charges':
            # TODO: is it right ?
            return False
        if self.objs[0] == O.from_name('sleep', nh.WAND_CLASS):
            return False
        if self.objs[0] == O.from_name('digging', nh.WAND_CLASS):
            return False
        return True

    ######## FOOD

    def is_food(self):
        if isinstance(self.objs[0], O.Food):
            assert self.is_unambiguous()
            return True

    def nutrition_per_weight(self):
        # TODO: corpses/tins
        assert self.is_food()
        return self.object.nutrition / max(self.unit_weight(), 1)

    def is_corpse(self):
        if self.objs[0] == O.from_name('corpse'):
            assert self.is_unambiguous()
            return True
        return False

    ######## STATUE

    def is_statue(self):
        if self.objs[0] == O.from_name('statue'):
            assert self.is_unambiguous()
            return True
        return False

    ######## CONTAINER

    def is_chest(self):
        if self.is_unambiguous() and self.object.name == 'bag of tricks':
            return False
        assert self.is_possible_container() or self.is_container(), self.objs
        assert isinstance(self.objs[0], O.Container), self.objs
        return self.objs[0].desc != 'bag'

    def is_container(self):
        # don't consider bag of tricks as a container.
        # If the identifier doesn't exist yet, it's not consider a container
        return self.content is not None

    def is_possible_container(self):
        if self.is_container():
            return False

        if self.is_unambiguous() and self.object.name == 'bag of tricks':
            return False
        return any((isinstance(obj, O.Container) for obj in self.objs))

    def content(self):
        assert self.is_container()
        return self.content
