from collections import defaultdict

import numpy as np

from . import utils
from .glyph import C, G, SHOP, SS


class Level:
    DUNGEONS_OF_DOOM = 0
    GNOMISH_MINES = 2
    QUEST = 3
    SOKOBAN = 4

    PLANE = 1000  # TODO: fill with actual value

    dungeon_names = {v: k for k, v in locals().items() if not k.startswith('_')}

    def __init__(self, dungeon_number, level_number):
        self.dungeon_number = dungeon_number
        self.level_number = level_number

        self.walkable = np.zeros((C.SIZE_Y, C.SIZE_X), bool)
        self.seen = np.zeros((C.SIZE_Y, C.SIZE_X), bool)
        self.objects = np.zeros((C.SIZE_Y, C.SIZE_X), np.int16)
        self.objects[:] = -1
        self.was_on = np.zeros((C.SIZE_Y, C.SIZE_X), bool)

        self.shop = np.zeros((C.SIZE_Y, C.SIZE_X), bool)
        self.shop_interior = np.zeros((C.SIZE_Y, C.SIZE_X), bool)
        self.shop_type = np.zeros((C.SIZE_Y, C.SIZE_X), np.int32) + SHOP.UNKNOWN

        self.search_count = np.zeros((C.SIZE_Y, C.SIZE_X), np.int32)
        self.door_open_count = np.zeros((C.SIZE_Y, C.SIZE_X), np.int32)

        self.item_disagreement_counter = np.zeros((C.SIZE_Y, C.SIZE_X), np.int32)
        self.items = np.empty((C.SIZE_Y, C.SIZE_X), dtype=object)
        self.items.fill([])
        self.item_count = np.zeros((C.SIZE_Y, C.SIZE_X), dtype=np.int32)

        self.stair_destination = {}  # {(y, x) -> ((dungeon, level), (y, x))}
        self.altars = {}  # {(y, x) -> alignment}

        self.corpses_to_eat = defaultdict(lambda: defaultdict(lambda: -10000))  # {(y, x) -> {monster_id -> age_turn}}

        # e.g. ad aerarium -- avoid valut entrance
        self.forbidden = np.zeros((C.SIZE_Y, C.SIZE_X), bool)

    def key(self):
        return (self.dungeon_number, self.level_number)

    def get_stairs(self, down=False, up=False, portal=False, all=False):
        # TODO: add portal
        if all:
            down = up = portal = True
        assert down or up or portal
        elems = []
        if down:
            elems.append(G.STAIR_DOWN)
        if up:
            elems.append(G.STAIR_UP)
        mask = utils.isin(self.objects, *elems)
        return {(y, x): self.stair_destination.get((y, x), None) for y, x in zip(*mask.nonzero())}

    def is_light_level(self):
        return np.sum(utils.isin(self.objects, [SS.S_room, SS.S_litcorr])) > 15
