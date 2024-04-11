import functools

from .data import *
from .. import utils


@utils.copy_result
@functools.lru_cache(len(objects))
def possibilities_from_glyph(i):
    assert nh.glyph_is_object(i)
    obj_id = nh.glyph_to_obj(i)
    desc = nh.objdescr.from_idx(obj_id).oc_descr or nh.objdescr.from_idx(obj_id).oc_name
    cat = ord(nh.objclass(obj_id).oc_class)

    if cat == nh.WEAPON_CLASS:
        if desc == 'runed broadsword':
            return [objects[obj_id]]

        ret = [o for o in objects if o is not None and (o.desc or o.name) == desc]
        assert len(ret) == 1
        return ret

    if cat == nh.ARMOR_CLASS:
        # https://nethackwiki.com/wiki/Armor
        ambiguous_groups = [
            ('piece of cloth', 'opera cloak', 'ornamental cope', 'tattered cape'),
            ('plumed helmet', 'etched helmet', 'crested helmet', 'visored helmet'),
            ('old gloves', 'padded gloves', 'riding gloves', 'fencing gloves'),
            (
                'mud boots', 'buckled boots', 'riding boots', 'snow boots', 'hiking boots', 'combat boots',
                'jungle boots'),
        ]
        for group in ambiguous_groups:
            if desc in group:
                return [o for o in objects if o is not None and o.desc in group]

        # the item is unambiguous or is 'conical hat'
        ret = [o for o in objects if o is not None and (o.desc or o.name) == desc]
        if desc != 'conical hat':
            assert len(ret) == 1, ret
        return ret

    if cat in [nh.TOOL_CLASS, nh.FOOD_CLASS]:
        return [o for i, o in enumerate(objects) if o is not None and ord(nh.objclass(i).oc_class) == cat and \
                (o.desc or o.name) == desc]

    if cat == nh.GEM_CLASS:
        # https://nethackwiki.com/wiki/Gem
        desc2names = {
            'black': ['black opal', 'jet', 'obsidian', 'worthless piece of black glass'],
            'blue': ['sapphire', 'turquoise', 'aquamarine', 'fluorite', 'worthless piece of blue glass'],
            'gray': ['luckstone', 'loadstone', 'touchstone', 'flint'],
            'green': ['emerald', 'turquoise', 'aquamarine', 'fluorite', 'jade', 'worthless piece of green glass'],
            'orange': ['jacinth', 'agate', 'worthless piece of orange glass'],
            'red': ['ruby', 'garnet', 'jasper', 'worthless piece of red glass'],
            'rock': ['rock'],
            'violet': ['amethyst', 'fluorite', 'worthless piece of violet glass'],
            'white': ['dilithium crystal', 'diamond', 'opal', 'fluorite', 'worthless piece of white glass'],
            'yellow': ['citrine', 'chrysoberyl', 'worthless piece of yellow glass'],
            'yellowish brown': ['amber', 'topaz', 'worthless piece of yellowish brown glass'],
        }
        return [from_name(name, cat) for name in desc2names[desc]]

    if cat == nh.AMULET_CLASS:
        if desc == 'Amulet of Yendor':
            return [from_name('cheap plastic imitation of the Amulet of Yendor'), from_name('Amulet of Yendor')]
        return [o for o in objects if isinstance(o, Amulet) and o.name is not None]

    if cat == nh.RING_CLASS:
        return [o for o in objects if isinstance(o, Ring) and o.name is not None]

    if cat == nh.COIN_CLASS:
        ret = [o for o in objects if isinstance(o, Coin) and o.name is not None]
        assert len(ret) == 1
        return ret

    if cat == nh.POTION_CLASS:
        if desc == 'clear':
            return [from_name('water')]
        return [o for o in objects if isinstance(o, Potion) and o.name != 'water' and o.name is not None]

    if cat == nh.SCROLL_CLASS:
        ambiguous_desc_name = [('stamped', 'mail'), ('unlabeled', 'blank paper')]
        for odesc, oname in ambiguous_desc_name:
            if desc == odesc:
                return [from_name(oname, nh.SCROLL_CLASS)]

        ambiguous_descs = [odesc for odesc, _ in ambiguous_desc_name]
        return [o for o in objects if isinstance(o, Scroll) and o.name not in ambiguous_descs and o.name is not None]

    if cat == nh.SPBOOK_CLASS:
        ambiguous_desc_name = [('plain', 'blank paper'), ('paperback', 'novel'), ('papyrus', 'Book of the Dead')]
        for odesc, oname in ambiguous_desc_name:
            if desc == odesc:
                return [from_name(oname, nh.SPBOOK_CLASS)]

        ambiguous_descs = [odesc for odesc, _ in ambiguous_desc_name]
        return [o for o in objects if isinstance(o, Spell) and o.name not in ambiguous_descs and o.name is not None]

    if cat == nh.WAND_CLASS:
        return [o for o in objects if isinstance(o, Wand) and o.name is not None]

    if cat in [nh.ROCK_CLASS, nh.BALL_CLASS, 16]:
        return [o for o in objects if o is not None and (o.desc or o.name) == desc]

    if objects[obj_id] == objects[-1]:
        return [objects[-1]]

    assert 0, (obj_id, objects[obj_id], cat)


@utils.copy_result
@functools.lru_cache(len(objects))
def possible_glyphs_from_object(obj):
    return [i for i in range(nh.GLYPH_OBJ_OFF, nh.GLYPH_OBJ_OFF + nh.NUM_OBJECTS)
            if objects[i - nh.GLYPH_OBJ_OFF] is not None and obj in possibilities_from_glyph(i)]


@utils.copy_result
@functools.lru_cache(len(objects) * 20)
def desc_to_glyphs(desc, category=None):
    assert desc is not None
    ret = [i + nh.GLYPH_OBJ_OFF for i, o in enumerate(objects)
           if o is not None and ord(nh.objclass(i).oc_class) == category and o.desc == desc]
    assert ret
    return ret


@functools.lru_cache(len(objects) * 100)
def from_name(name, category=None):
    ret = []
    for i, o in enumerate(objects):
        if o is not None and o.name == name and \
                (category is None or ord(nh.objclass(i).oc_class) == category):
            ret.append(o)

    assert len(ret) == 1, (name, category, ret)
    return ret[0]


@functools.lru_cache(len(objects))
def get_category(obj):
    return ord(nh.objclass(objects.index(obj)).oc_class)
