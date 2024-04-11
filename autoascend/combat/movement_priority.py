from ..utils import adjacent
from . import utils
from .monster_utils import WEAK_MONSTERS, ONLY_RANGED_SLOW_MONSTERS, consider_melee_only_ranged_if_hp_full, \
    imminent_death_on_melee, EXPLODING_MONSTERS, WEIRD_MONSTERS


def _draw_around(priority, y, x, value, radius=1, operation='add'):
    # TODO: optimize
    for y1 in range(y - radius, y + radius + 1):
        for x1 in range(x - radius, x + radius + 1):
            if max(abs(y1 - y), abs(x1 - x)) != radius:
                continue
            if 0 <= y1 < priority.shape[0] and 0 <= x1 < priority.shape[1]:
                if operation == 'add':
                    priority[y1, x1] += value
                elif operation == 'max':
                    priority[y1, x1] = max(priority[y1, x1], value)
                else:
                    assert 0, operation


def _draw_ranged(priority, y, x, value, walkable, radius=1, operation='add'):
    # TODO: optimize
    for direction_y in (-1, 0, 1):
        for direction_x in (-1, 0, 1):
            if direction_y != 0 or direction_x != 0:
                for i in range(1, radius + 1):
                    y1 = y + direction_y * i
                    x1 = x + direction_x * i
                    if 0 <= y1 < priority.shape[0] and 0 <= x1 < priority.shape[1]:
                        if not walkable[y1, x1]:
                            break
                        if operation == 'add':
                            priority[y1, x1] += value
                        elif operation == 'max':
                            priority[y1, x1] = max(priority[y1, x1], value)
                        else:
                            assert 0, operation


def draw_monster_priority_positive(agent, monster, priority, walkable):
    _, y, x, mon, _ = monster

    # don't move into the monster
    priority[y, x] = float('nan')

    if mon.mname in WEAK_MONSTERS:
        # weak monster - freely engage in melee
        _draw_around(priority, y, x, 2, radius=1, operation='max')
        _draw_around(priority, y, x, 1, radius=2, operation='max')
    elif 'mold' in mon.mname and mon.mname not in ONLY_RANGED_SLOW_MONSTERS:
        if agent.blstats.hitpoints >= 15 or agent.blstats.hitpoints == agent.blstats.max_hitpoints:
            # freely engage in melee
            _draw_around(priority, y, x, 2, radius=1, operation='max')
            _draw_around(priority, y, x, 1, radius=2, operation='max')
        if len(agent.inventory.get_ranged_combinations()):
            _draw_ranged(priority, y, x, 1, walkable, radius=7, operation='max')
    elif mon.mname in ONLY_RANGED_SLOW_MONSTERS:  # and agent.inventory.get_ranged_combinations():
        if consider_melee_only_ranged_if_hp_full(agent, monster):
            _draw_around(priority, y, x, 2, radius=1, operation='max')
            _draw_around(priority, y, x, 1, radius=2, operation='max')
        if len(agent.inventory.get_ranged_combinations()):
            _draw_ranged(priority, y, x, 1, walkable, radius=7, operation='max')
    elif 'unicorn' in mon.mname:
        if agent.blstats.hitpoints >= 15 or agent.blstats.hitpoints == agent.blstats.max_hitpoints:
            # freely engage in melee
            _draw_around(priority, y, x, 2, radius=1, operation='max')
            _draw_around(priority, y, x, 1, radius=2, operation='max')
    else:
        if not imminent_death_on_melee(agent, monster) and not utils.wielding_ranged_weapon(agent):
            # engage, but ensure striking first if possible
            if mon.mmove <= 12:
                _draw_around(priority, y, x, 3, radius=2, operation='max')
            else:
                _draw_around(priority, y, x, 3, radius=3, operation='max')
        if utils.wielding_ranged_weapon(agent):
            _draw_ranged(priority, y, x, 4, walkable, radius=7, operation='max')
        elif len(agent.inventory.get_ranged_combinations()):
            _draw_ranged(priority, y, x, 1, walkable, radius=7, operation='max')


def draw_monster_priority_negative(agent, monster, priority, walkable):
    _, y, x, mon, _ = monster

    if imminent_death_on_melee(agent, monster) and not mon.mname in WEAK_MONSTERS \
            and not mon.mname in ONLY_RANGED_SLOW_MONSTERS:
        if mon.mmove <= 12:
            _draw_around(priority, y, x, -10, radius=1)
        else:
            if adjacent((agent.blstats.y, agent.blstats.x), (y, x)):
                # no point in running -- monster is fast
                pass
            else:
                _draw_around(priority, y, x, -10, radius=2)
                _draw_around(priority, y, x, -5, radius=1)

        if not len(agent.inventory.get_ranged_combinations()):
            # prefer avoiding being in line of fire
            _draw_ranged(priority, y, x, -1, walkable, radius=7)

    # if agent.blstats.hitpoints <= 8 and not is_monster_faster(agent, monster) and not mon.mname in WEAK_MONSTERS \
    #         and not mon.mname in ONLY_RANGED_SLOW_MONSTERS:
    #     # stay out of melee range
    #     _draw_around(priority, y, x, -10, radius=1)
    #     if not len(agent.inventory.get_ranged_combinations()):
    #         # prefer avoiding being in line of fire
    #         _draw_ranged(priority, y, x, -1, walkable, radius=7)

    if mon.mname in EXPLODING_MONSTERS:
        _draw_around(priority, y, x, -10, radius=1)
        if mon.mname not in ONLY_RANGED_SLOW_MONSTERS:
            _draw_around(priority, y, x, -5, radius=2)
        _draw_ranged(priority, y, x, 4, walkable, radius=7)
    elif 'mold' in mon.mname and mon.mname not in ONLY_RANGED_SLOW_MONSTERS:
        # prioritize staying in ranged weapons line of fire
        if len(agent.inventory.get_ranged_combinations()):
            _draw_ranged(priority, y, x, 2, walkable, radius=7)
    elif mon.mname in WEIRD_MONSTERS:
        # stay away
        _draw_around(priority, y, x, -10, radius=1)
        # prioritize staying in ranged weapons line of fire
        if len(agent.inventory.get_ranged_combinations()):
            _draw_ranged(priority, y, x, 6, walkable, radius=7)
    elif mon.mname in ONLY_RANGED_SLOW_MONSTERS:  # and agent.inventory.get_ranged_combinations():
        # ignore
        pass
    elif 'unicorn' in mon.mname:
        pass
    else:
        if mon.mname not in WEAK_MONSTERS:
            # engage, but ensure striking first if possible
            _draw_around(priority, y, x, -9, radius=1)
            if not len(agent.inventory.get_ranged_combinations()):
                _draw_ranged(priority, y, x, -1, walkable, radius=7)

    if mon.mname == 'purple worm' and len(agent.inventory.get_ranged_combinations()):
        _draw_around(priority, y, x, -10, radius=1)
