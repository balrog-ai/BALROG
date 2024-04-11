import nle.nethack as nh

from . import monster as MON
from . import screen_symbols as SS


class WEA:
    @staticmethod
    def expected_damage(damage_str):
        if '-' in damage_str:
            raise NotImplementedError()
        ret = 0
        for word in damage_str.split('+'):
            if 'd' in word:
                sides = int(word[word.find('d') + 1:])
                mult = word[:word.find('d')]
                if not mult:
                    mult = 1
                else:
                    mult = int(mult)
            else:
                sides = 1
                mult = int(word)
            ret += mult * (1 + sides) / 2
        return ret


class SHOP:
    UNKNOWN = 0
    # names from nle/src/shknam.c
    name2id = {
        'UNKNOWN': UNKNOWN,
        "general store": 1,
        "used armor dealership": 2,
        "second-hand bookstore": 3,
        "liquor emporium": 4,
        "antique weapons outlet": 5,
        "delicatessen": 6,
        "jewelers": 7,
        "quality apparel and accessories": 8,
        "hardware store": 9,
        "rare books": 10,
        "health food store": 11,
        "lighting store": 12,
    }


class Hunger:
    SATIATED = 0
    NOT_HUNGRY = 1
    HUNGRY = 2
    WEAK = 3
    FAINTING = 4


class C:
    SIZE_X = 79
    SIZE_Y = 21


class G:  # Glyphs
    FLOOR: ['.'] = frozenset({SS.S_room, SS.S_ndoor, SS.S_darkroom, SS.S_corr, SS.S_litcorr})
    VISIBLE_FLOOR: ['.'] = frozenset({SS.S_room, SS.S_litcorr})
    STONE: [' '] = frozenset({SS.S_stone})
    WALL: ['|', '-'] = frozenset({SS.S_vwall, SS.S_hwall, SS.S_tlcorn, SS.S_trcorn, SS.S_blcorn, SS.S_brcorn,
                                  SS.S_crwall, SS.S_tuwall, SS.S_tdwall, SS.S_tlwall, SS.S_trwall})
    STAIR_UP: ['<'] = frozenset({SS.S_upstair, SS.S_upladder})
    STAIR_DOWN: ['>'] = frozenset({SS.S_dnstair, SS.S_dnladder})
    ALTAR: ['_'] = frozenset({SS.S_altar})
    FOUNTAIN = frozenset({SS.S_fountain})

    DOOR_CLOSED: ['+'] = frozenset({SS.S_vcdoor, SS.S_hcdoor})
    DOOR_OPENED: ['-', '|'] = frozenset({SS.S_vodoor, SS.S_hodoor})
    DOORS = frozenset.union(DOOR_CLOSED, DOOR_OPENED)

    BARS = frozenset({SS.S_bars})

    MONS = frozenset(MON.ALL_MONS)
    PETS = frozenset(MON.ALL_PETS)
    WARNING = frozenset({nh.GLYPH_WARNING_OFF + i for i in range(nh.WARNCOUNT)})
    INVISIBLE_MON = frozenset({nh.GLYPH_INVISIBLE, *WARNING})

    SHOPKEEPER = frozenset({MON.fn('shopkeeper')})
    ORACLE = frozenset({MON.fn('Oracle')})
    GUARD = frozenset({MON.fn('guard')})

    STATUES = frozenset({i + nh.GLYPH_STATUE_OFF for i in range(nh.NUMMONS)})

    BODIES = frozenset({nh.GLYPH_BODY_OFF + i for i in range(nh.NUMMONS)})
    OBJECTS = frozenset({nh.GLYPH_OBJ_OFF + i for i in range(nh.NUM_OBJECTS)
                         if ord(nh.objclass(i).oc_class) != nh.ROCK_CLASS})
    BOULDER = frozenset({nh.GLYPH_OBJ_OFF + i for i in range(nh.NUM_OBJECTS)
                         if ord(nh.objclass(i).oc_class) == nh.ROCK_CLASS})

    NORMAL_OBJECTS = frozenset({i for i in range(nh.MAX_GLYPH) if nh.glyph_is_normal_object(i)})
    FOOD_OBJECTS = frozenset({i for i in NORMAL_OBJECTS
                              if ord(nh.objclass(nh.glyph_to_obj(i)).oc_class) == nh.FOOD_CLASS})

    TRAPS = frozenset({SS.S_arrow_trap, SS.S_dart_trap, SS.S_falling_rock_trap, SS.S_squeaky_board, SS.S_bear_trap,
                       SS.S_land_mine, SS.S_rolling_boulder_trap, SS.S_sleeping_gas_trap, SS.S_rust_trap,
                       SS.S_fire_trap, SS.S_pit, SS.S_spiked_pit, SS.S_hole, SS.S_trap_door, SS.S_teleportation_trap,
                       SS.S_level_teleporter, SS.S_magic_portal, SS.S_web, SS.S_statue_trap, SS.S_magic_trap,
                       SS.S_anti_magic_trap, SS.S_polymorph_trap})

    SWALLOW = frozenset(range(nh.GLYPH_SWALLOW_OFF, nh.GLYPH_WARNING_OFF))

    DICT = {k: v for k, v in locals().items() if not k.startswith('_')}

    @classmethod
    def assert_map(cls, glyphs, chars):
        for glyph, char in zip(glyphs.reshape(-1), chars.reshape(-1)):
            char = bytes([char]).decode()
            for k, v in cls.__annotations__.items():
                assert glyph not in cls.DICT[k] or char in v, f'{k} {v} {glyph} {char}'


G.INV_DICT = {i: [k for k, v in G.DICT.items() if i in v]
              for i in set.union(*map(set, G.DICT.values()))}
