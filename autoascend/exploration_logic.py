import re

import cv2
import numpy as np
from nle import nethack as nh
from nle.nethack import actions as A

from . import utils
from .character import Character
from .exceptions import AgentPanic
from .glyph import G, C, SS
from .level import Level
from .strategy import Strategy


class ExplorationLogic:
    def __init__(self, agent):
        self.agent = agent

    # TODO: think how to handle the situation with wizard's tower
    def _level_dfs(self, start, end, path, vis):
        if start in vis:
            return

        if start == end:
            return path

        vis.add(start)
        stairs = self.agent.levels[start].get_stairs(all=True) if start in self.agent.levels else {}
        for k, t in stairs.items():
            if t is None:
                continue
            glyph = self.agent.levels[start].objects[k]
            dir = '>' if glyph in G.STAIR_DOWN else '<' if glyph in G.STAIR_UP else ''
            assert dir, glyph  # TODO: portals

            path.append((k, t, dir))
            r = self._level_dfs(t[0], end, path, vis)
            if r is not None:
                return r
            path.pop()

    def get_path_to_level(self, dungeon_number, level_number):
        return self._level_dfs(self.agent.current_level().key(), (dungeon_number, level_number), [], set())

    def get_achievable_levels(self, dungeon_number=None, level_number=None):
        assert (dungeon_number is None) == (level_number is None)
        if dungeon_number is None:
            dungeon_number, level_number = self.agent.current_level().key()

        vis = set()
        self._level_dfs((dungeon_number, level_number), (-1, -1), [], vis)
        return vis

    def levels_to_explore_to_get_to(self, dungeon_number, level_number, achievable_levels=None):
        if achievable_levels is None:
            achievable_levels = self.get_achievable_levels()

        if len(achievable_levels) == 1:
            return achievable_levels

        if (dungeon_number, level_number) in achievable_levels:
            return set()

        if any((dun == dungeon_number for dun, lev in achievable_levels)):
            closest_level_number = min((lev for dun, lev in achievable_levels if dun == dungeon_number),
                                       key=lambda lev: abs(level_number - lev))
            return {(dungeon_number, closest_level_number)}

        if dungeon_number == Level.GNOMISH_MINES:
            return set.union(*[self.levels_to_explore_to_get_to(Level.DUNGEONS_OF_DOOM, i, achievable_levels)
                               for i in range(2, 5)],
                             {(Level.DUNGEONS_OF_DOOM, i) for i in range(2, 5)
                              if (Level.DUNGEONS_OF_DOOM, i) in achievable_levels})

        if dungeon_number == Level.SOKOBAN:
            # TODO: one level below oracle
            return set.union(*[self.levels_to_explore_to_get_to(Level.DUNGEONS_OF_DOOM, i, achievable_levels)
                               for i in range(6, 11)],
                             {(Level.DUNGEONS_OF_DOOM, i) for i in range(6, 11)
                              if (Level.DUNGEONS_OF_DOOM, i) in achievable_levels})

        if all((dun == Level.GNOMISH_MINES for dun, lev in achievable_levels)):
            return self.levels_to_explore_to_get_to(Level.GNOMISH_MINES, 1).union(
                ({(Level.GNOMISH_MINES, 1)} if (Level.GNOMISH_MINES, 1) in achievable_levels else set())
            )

        # TODO: more dungeons

        assert 0, ((dungeon_number, level_number), achievable_levels)

    def get_unexplored_stairs(self, dungeon_number=None, level_number=None, **kwargs):
        assert (dungeon_number is None) == (level_number is None)
        if dungeon_number is None:
            dungeon_number, level_number = self.agent.current_level().key()
        stairs = self.agent.levels[dungeon_number, level_number].get_stairs(**kwargs)
        return [k for k, v in stairs.items() if v is None]

    @Strategy.wrap
    def explore_stairs(self, go_to_strategy, **kwargs):
        unexplored_stairs = self.get_unexplored_stairs(**kwargs)
        if len(unexplored_stairs) == 0:
            yield False
        yield True

        y, x = list(unexplored_stairs)[self.agent.rng.randint(0, len(unexplored_stairs))]
        glyph = self.agent.current_level().objects[y, x]
        dir = '>' if glyph in G.STAIR_DOWN else '<' if glyph in G.STAIR_UP else ''
        assert dir, glyph  # TODO: portals

        go_to_strategy(y, x).run()
        assert (self.agent.blstats.y, self.agent.blstats.x) == (y, x)
        while self.agent.has_pet:
            if utils.any_in(self.agent.glyphs[max(self.agent.blstats.y - 1, 0) : self.agent.blstats.y + 2,
                                              max(self.agent.blstats.x - 1, 0) : self.agent.blstats.x + 2],
                            G.PETS):
                break
            self.agent.move('.')
        self.agent.move(dir)

    @Strategy.wrap
    def follow_level_path_strategy(self, path, go_to_strategy):
        if not path:
            yield False
        yield True
        for (y, x), _, dir in path:
            go_to_strategy(y, x).run()
            assert (self.agent.blstats.y, self.agent.blstats.x) == (y, x)
            while self.agent.has_pet:
                if utils.any_in(self.agent.glyphs[max(self.agent.blstats.y - 1, 0) : self.agent.blstats.y + 2,
                                                max(self.agent.blstats.x - 1, 0) : self.agent.blstats.x + 2],
                                G.PETS):
                    break
                self.agent.move('.')
            self.agent.move(dir)

    @Strategy.wrap
    def go_to_level_strategy(self, dungeon_number, level_number, go_to_strategy, explore_strategy):
        yield True
        while 1:
            levels_to_search = self.levels_to_explore_to_get_to(dungeon_number, level_number)
            if len(levels_to_search) == 0:
                break

            @Strategy.wrap
            def go_to_least_explored_level():
                levels_to_search = self.levels_to_explore_to_get_to(dungeon_number, level_number)
                if not levels_to_search:
                    yield False

                exploration_levels = {level: self.agent.levels[level].search_count.sum() for level in levels_to_search}

                for level in sorted(levels_to_search):
                    if len(self.get_unexplored_stairs(*level, all=True)) > 0:
                        exploration_levels[level] -= 10000

                min_exploration_level = min(exploration_levels.values())
                if self.agent.current_level().key() in levels_to_search and \
                        exploration_levels[self.agent.current_level().key()] < min_exploration_level + 150:
                    yield False
                yield True

                for level in levels_to_search:
                    if exploration_levels[level] == min_exploration_level:
                        target_level = level
                        break
                else:
                    assert 0
                path = self.get_path_to_level(*target_level)
                assert path is not None
                self.follow_level_path_strategy(path, go_to_strategy).run()
                assert self.agent.current_level().key() == target_level

            if self.agent.current_level().key() not in levels_to_search:
                go_to_least_explored_level().run()
                assert self.agent.current_level().key() in levels_to_search
                continue

            explore_strategy.preempt(self.agent, [
                self.explore_stairs(go_to_strategy, all=True) \
                        .condition(lambda: self.agent.current_level().key() in levels_to_search),
                go_to_least_explored_level(),
            ], continue_after_preemption=False).run()

        path = self.get_path_to_level(dungeon_number, level_number)
        assert path is not None, \
                (self.agent.current_level().key(), (dungeon_number, level_number), self.get_achievable_levels())
        with self.agent.env.debug_log(f'going to level {Level.dungeon_names[dungeon_number]}:{level_number}'):
            self.follow_level_path_strategy(path, go_to_strategy).run()
            assert self.agent.current_level().key() == (dungeon_number, level_number)

    @Strategy.wrap
    def go_to_strategy(self, y, x, *args, **kwargs):
        if self.agent.bfs()[y, x] == -1 or (self.agent.blstats.y, self.agent.blstats.x) == (y, x):
            yield False
        yield True
        return self.agent.go_to(y, x, *args, **kwargs)

    @Strategy.wrap
    def search_neighbors_for_traps(self, offset=0):
        search_count = 0
        level = self.agent.current_level()
        for y, x in self.agent.neighbors(self.agent.blstats.y, self.agent.blstats.x, shuffle=False):
            if level.was_on[y, x] or level.objects[y, x] in G.TRAPS:
                continue
            c = level.search_count[max(y - 1, 0) : y + 2, max(x - 1, 0) : x + 2].sum()

            if (self.agent.last_observation['specials'][y, x] & nh.MG_OBJPILE) == 0:
                search_count = max(search_count, offset - c)
                continue

            c = level.search_count[max(y - 1, 0) : y + 2, max(x - 1, 0) : x + 2].sum()
            search_count = max(search_count, offset + 4 - c)

        if search_count == 0:
            yield False
        yield True

        for _ in range(search_count):
            self.agent.search()

    @Strategy.wrap
    def check_altar(self):
        level = self.agent.current_level()
        pos = (self.agent.blstats.y, self.agent.blstats.x)
        if pos in level.altars and level.altars[pos] == Character.UNKNOWN:
            yield True
            with self.agent.atom_operation():
                self.agent.step(A.Command.LOOK)
                r = re.search(r'There is an altar to [a-zA-Z- ]+ \(([a-z]+)\) here.', self.agent.message or self.agent.popup[0])
                assert r is not None, (self.agent.message, self.agent.popup)
                alignment = r.groups()[0]
                assert alignment in Character.name_to_alignment, (alignment, self.agent.message)
                alignment = Character.name_to_alignment[alignment]
                level.altars[pos] = alignment
                return

        yield False

    @utils.debug_log('patrol')
    @Strategy.wrap
    def patrol(self):
        yielded = False
        while True:
            reachable = self.agent.bfs()
            reachable[reachable < 0] = 0
            if (reachable == 0).all():
                if not yielded:
                    yield False
                return
            if not yielded:
                yield True
                yielded = True


            i = self.agent.rng.choice(range(reachable.shape[0] * reachable.shape[1]), p=(reachable.reshape(-1) / reachable.sum()))
            y, x = i // reachable.shape[1], i % reachable.shape[1]
            self.agent.go_to(y, x, fast=True)

    @utils.debug_log('explore1')
    def explore1(self, search_prio_limit=0, door_open_count=4, kick_doors=True, trap_search_offset=0, check_altar_alignment=True, fast_go_to=False):
        # TODO: refactor entire function


        @Strategy.wrap
        def open_neighbor_doors():
            # TODO: polymorphed into a handless creature, too heavy load to kick, using lockpicks

            yielded = False
            for py, px in self.agent.neighbors(self.agent.blstats.y, self.agent.blstats.x, diagonal=False):
                if (self.agent.current_level().door_open_count[py, px] < door_open_count or kick_doors) and \
                        self.agent.glyphs[py, px] in G.DOOR_CLOSED:
                    if not yielded:
                        yielded = True
                        yield True
                    with self.agent.panic_if_position_changes():
                        if not self.agent.open_door(py, px):
                            if not 'locked' in self.agent.message:
                                for _ in range(6):
                                    if self.agent.open_door(py, px):
                                        break
                                else:
                                    if kick_doors:
                                        while self.agent.glyphs[py, px] in G.DOOR_CLOSED:
                                            self.agent.kick(py, px)
                            else:
                                if kick_doors:
                                    while self.agent.glyphs[py, px] in G.DOOR_CLOSED:
                                        self.agent.kick(py, px)
                    break

            if not yielded:
                yield False

        def to_visit_func():
            level = self.agent.current_level()

            stone = ~level.seen & utils.isin(self.agent.glyphs, G.STONE)
            doors = utils.isin(self.agent.glyphs, G.DOOR_CLOSED) & (level.door_open_count < door_open_count)
            if not stone.any() and not doors.any():
                return stone

            to_visit = np.zeros((C.SIZE_Y, C.SIZE_X), dtype=bool)
            tmp = np.zeros((C.SIZE_Y, C.SIZE_X), dtype=bool)
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy != 0 or dx != 0:
                        to_visit |= utils.translate(stone, dy, dx, out=tmp)
                        if dx == 0 or dy == 0:
                            to_visit |= utils.translate(doors, dy, dx, out=tmp)
            return to_visit

        def to_search_func(prio_limit=0, return_prio=False):
            level = self.agent.current_level()
            dis = self.agent.bfs()

            prio = np.zeros((C.SIZE_Y, C.SIZE_X), np.float32)
            prio[:] = -1
            prio -= level.search_count ** 2 * 2

            counts = level.search_count[level.search_count > 0]
            search_diff = 0
            if len(counts):
                search_diff = np.max(counts) - np.quantile(counts, 0.3)
                self.agent.stats_logger.log_max_value('search_diff', search_diff)

            if search_diff > 400 and self.agent.blstats.hitpoints == self.agent.blstats.max_hitpoints\
                    and level.search_count[self.agent.blstats.y, self.agent.blstats.x] == np.max(counts):
                self.agent.stats_logger.log_event('allow_walk_traps')
                self.agent._allow_walking_through_traps_turn = self.agent._last_turn
                if search_diff > 500:
                    self.agent.stats_logger.log_event('allow_attack_all')
                    self.agent._allow_attack_all_turn = self.agent._last_turn

            # is_on_corridor = utils.isin(level.objects, G.CORRIDOR)
            is_on_door = utils.isin(level.objects, G.DOORS)

            stones = np.zeros((C.SIZE_Y, C.SIZE_X), np.int32)
            walls = np.zeros((C.SIZE_Y, C.SIZE_X), np.int32)

            tmp = np.zeros((C.SIZE_Y, C.SIZE_X), dtype=self.agent.glyphs.dtype)
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy != 0 or dx != 0:
                        stones += utils.isin(utils.translate(level.objects, dy, dx, out=tmp), G.STONE)
                        walls += utils.isin(utils.translate(level.objects, dy, dx, out=tmp), G.WALL)

            prio += (is_on_door & (stones > 3)) * 250
            prio += (np.stack([utils.translate(level.walkable, y, x, out=tmp).astype(np.int32)
                               for y, x in [(1, 0), (-1, 0), (0, 1), (0, -1)]]).sum(0) <= 1) * 250
            prio[(stones == 0) & (walls == 0)] = -np.inf

            prio[~level.walkable | (dis == -1)] = -np.inf

            if return_prio:
                return prio
            return prio >= prio_limit

        @Strategy.wrap
        def open_visit_search(search_prio_limit):
            yielded = False
            while 1:
                if check_altar_alignment and self.check_altar().check_condition():
                    if not yielded:
                        yielded = True
                        yield True
                    self.check_altar().run()
                    continue

                if open_neighbor_doors().check_condition():
                    if not yielded:
                        yielded = True
                        yield True
                    open_neighbor_doors().run()
                    continue

                to_visit = to_visit_func()
                to_search = to_search_func(search_prio_limit if search_prio_limit is not None else 0)

                if check_altar_alignment:
                    for (y, x), alignment in self.agent.current_level().altars.items():
                        if alignment == Character.UNKNOWN:
                            to_visit[y, x] = True

                # consider exploring tile only when there is a path to it
                dis = self.agent.bfs()
                to_explore = (to_visit | to_search) & (dis != -1)

                dynamic_search_fallback = False
                if not to_explore.any():
                    dynamic_search_fallback = True
                else:
                    # find all closest to_explore tiles
                    nonzero_y, nonzero_x = ((dis == dis[to_explore].min()) & to_explore).nonzero()
                    if len(nonzero_y) == 0:
                        dynamic_search_fallback = True

                if dynamic_search_fallback:
                    if search_prio_limit is not None and search_prio_limit >= 0:
                        if not yielded:
                            yield False
                        return

                    search_prio = to_search_func(return_prio=True)
                    if search_prio_limit is not None:
                        search_prio[search_prio < search_prio_limit] = -np.inf
                        search_prio -= dis * np.isfinite(search_prio) * 100
                    else:
                        search_prio -= dis * 4

                    to_search = np.isfinite(search_prio)
                    to_explore = (to_visit | to_search) & (dis != -1)
                    if not to_explore.any():
                        if not yielded:
                            yield False
                        return
                    nonzero_y, nonzero_x = ((search_prio == search_prio[to_explore].max()) & to_explore).nonzero()

                if not yielded:
                    yielded = True
                    yield True

                # select random closest to_explore tile
                i = self.agent.rng.randint(len(nonzero_y))
                target_y, target_x = nonzero_y[i], nonzero_x[i]

                with self.agent.env.debug_tiles(to_explore, color=(0, 0, 255, 64)):
                    self.agent.go_to(target_y, target_x, fast=fast_go_to, debug_tiles_args=dict(
                        color=(255 * bool(to_visit[target_y, target_x]),
                               255, 255 * bool(to_search[target_y, target_x])),
                        is_path=True))
                    if to_search[target_y, target_x] and not to_visit[target_y, target_x]:
                        self.agent.search(5)

            assert search_prio_limit is not None

        return (
            open_visit_search(search_prio_limit)
            .preempt(self.agent, [
                self.agent.inventory.gather_items(),
                self.untrap_traps().every(10),
            ])
            .preempt(self.agent, [
                self.search_neighbors_for_traps(trap_search_offset),
            ])
        )

    def worth_untrapping(self, trap_y, trap_x):
        walkable = self.agent.current_level().walkable
        last_walkable = None
        walkable_changes = 0
        for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
            y, x = trap_y - dy, trap_x - dx
            if not 0 <= y < walkable.shape[0] or not 0 <= x < walkable.shape[1]:
                w = False
            else:
                w = walkable[y, x]
            if last_walkable is not None and last_walkable != w:
                walkable_changes += 1
            last_walkable = w
        assert walkable_changes % 2 == 0
        return walkable_changes >= 4

    @utils.debug_log('untrap_traps')
    @Strategy.wrap
    def untrap_traps(self):
        if self.agent.blstats.hitpoints < 10 or (self.agent.blstats.hitpoints / self.agent.blstats.max_hitpoints) < 0.5:
            # not enough HP to risk untrapping at all
            yield False
            return

        untrappable_traps = [SS.S_web, SS.S_bear_trap, SS.S_land_mine, SS.S_dart_trap, SS.S_arrow_trap]
        # TODO: consider strategy for SS.S_pit, SS.S_spiked_pit
        level = self.agent.current_level()
        trap_mask = utils.isin(level.objects, untrappable_traps)

        if not trap_mask.any():
            yield False
            return

        trap_locations = list(zip(*trap_mask.nonzero()))
        trap_locations = [loc for loc in trap_locations if self.worth_untrapping(*loc)]

        if not trap_locations:
            yield False
            return

        trap_mask = cv2.dilate(trap_mask.astype(np.uint8), kernel=np.ones((3, 3))).astype(bool)
        bfs = self.agent.bfs()
        trap_mask[self.agent.blstats.y, self.agent.blstats.x] = 0  # don't try to untrap when standing on it
        trap_mask &= (bfs >= 0)
        if not trap_mask.any():
            yield False
            return
        trap_mask &= bfs == bfs[trap_mask].min()
        assert trap_mask.any()

        if not trap_mask.any():
            yield False
            return
        yield True

        closest_y, closest_x = trap_mask.nonzero()
        target_y, target_x = closest_y[0], closest_x[0]
        assert target_y != self.agent.blstats.y or target_x != self.agent.blstats.x

        self.agent.go_to(target_y, target_x, debug_tiles_args=dict(color=(255, 120, 0), is_path=True))
        for trap_y, trap_x in trap_locations:
            if utils.adjacent((self.agent.blstats.y, self.agent.blstats.x), (trap_y, trap_x)):
                trap_glyph = level.objects[trap_y, trap_x]
                if trap_glyph not in untrappable_traps:
                    raise AgentPanic('a trap is no longer there')
                # assert level.objects[trap_y, trap_x] == trap_glyph
                if trap_y == self.agent.blstats.y and trap_x == self.agent.blstats.x:
                    continue
                self.agent.untrap(trap_y, trap_x)
                self.agent.check_terrain(force=True)
