from collections import defaultdict

import numpy as np

from . import character


class StatsLogger:
    def __init__(self):
        self._values = {
            "agent_panic": 0,
            "elbereth_write": 0,
            "container_untrap_success": 0,
            "container_untrap_fail": 0,
            "untrap_success": 0,
            "triggered_undetected_trap": 0,
            "allow_walk_traps": 0,
            "allow_attack_all": 0,
            "sokoban_dropped": 0,
            "wait_in_fight": 0,
            "melee_gas_spore": 0,
            "ad_aerarium_below_me": 0,
            "drop_gold": 0,
            **{f"cast_{n}": 0 for n in character.ALL_SPELL_NAMES},
            **{f"cast_fail_{n}": 0 for n in character.ALL_SPELL_NAMES},
        }
        self._max_values = {
            "search_diff": -float('inf'),
        }

        self._cumulative_values = {
            "max_turns_on_position": defaultdict(int),
        }

        self.gold_stats = ['mean', 'median', 'std', 'min', 'max', 'first', 'last']
        self._keys = list(self._values) + list(self._max_values) + list(self._cumulative_values) + self.gold_stats

        self.gold = []

    def log_cumulative_value(self, name, key, value):
        self._cumulative_values[name][key] += value

    def log_event(self, name):
        self._values[name] += 1

    def log_gold(self, amount):
        self.gold.append(amount)

    def log_max_value(self, name, value):
        self._max_values[name] = max(self._max_values[name], value)

    def get_stats_dict(self):
        ret = dict()
        ret.update(self._values)
        ret.update(self._max_values)
        ret.update({k: max(v.values()) for k, v in self._cumulative_values.items()})

        for stat in self.gold_stats:
            try:
                ret['gold_' + stat] = getattr(np, stat)(self.gold)
            except AttributeError:
                if stat == 'first':
                    ret['gold_' + stat] = max(self.gold[:20])
                elif stat == 'last':
                    ret['gold_' + stat] = self.gold[-1]
                else:
                    assert 0, stat
        return ret
