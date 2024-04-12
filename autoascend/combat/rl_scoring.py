import numpy as np
from . import utils

RL_CONTEXT_SIZE = 7


def fight2_action_space(agent):
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    return [
        *[('move', dy, dx) for dy, dx in directions],
        *[('melee', dy, dx) for dy, dx in directions],
        *[('ranged', dy, dx) for dy, dx in directions],
        # *[('zap', dy, dx) for dy, dx in directions],
        # ('pickup',),
    ]


def init_fight2_model(agent):
    from .. import rl_utils
    agent._fight2_model = rl_utils.RLModel((
        ('player_scalar_stats', ((5,), np.float32)),
        ('semantic_maps', ((3, RL_CONTEXT_SIZE, RL_CONTEXT_SIZE), np.float32)),
        ('heur_action_priorities', ((8 * 3,), np.float32)),
    ),
        action_space=fight2_action_space(agent),
        train=agent.rl_model_to_train == 'fight2',
        training_comm=agent.rl_model_training_comm,
    )
    with open('/workspace/muzero/rl_features_stats.json', 'r') as f:
        agent._fight2_features_stats = json.load(f)


def fight2_player_scalar_stats(agent):
    ret = [agent.blstats.hitpoints,
           agent.blstats.max_hitpoints,
           agent.blstats.hitpoints / agent.blstats.max_hitpoints,
           utils.wielding_ranged_weapon(agent),
           utils.wielding_melee_weapon(agent)]
    ret = np.array(ret, dtype=np.float32)
    assert not np.isnan(ret).any()
    return ret


def fight2_semantic_maps(agent):
    radius_y = radius_x = RL_CONTEXT_SIZE // 2
    y1, y2, x1, x2 = agent.blstats.y - radius_y, agent.blstats.y + radius_y + 1, \
                     agent.blstats.x - radius_x, agent.blstats.x + radius_x + 1
    level = agent.current_level()
    walkable = level.walkable & ~utils.isin(agent.glyphs, G.BOULDER) & \
               ~agent.monster_tracker.peaceful_monster_mask & \
               ~utils.isin(level.objects, G.TRAPS)

    mspeed = np.ones((C.SIZE_Y, C.SIZE_X), dtype=int) * np.nan
    for _, y, x, mon, _ in agent.get_visible_monsters():
        mspeed[y][x] = mon.mmove

    ret = list(map(lambda q: utils.slice_with_padding(q, y1, y2, x1, x2), (
        walkable, agent.monster_tracker.monster_mask, mspeed,
    )))
    return np.stack(ret, axis=0).astype(np.float32)


def fight2_encoded_heur_action_priorities(agent, heur_priorities):
    ret = []
    for action in agent._fight2_model.action_space:
        if action in heur_priorities:
            ret.append(heur_priorities[action])
        else:
            ret.append(np.nan)
    return np.array(ret).astype(np.float32)


def fight2_get_observation(agent, heur_priorities):
    def normalize(name, features):
        mean, std, minv = [agent._fight2_features_stats[name][k] for k in ['mean', 'std', 'min']]
        v_normalized = features.copy()
        assert len(mean) == features.shape[0], (len(mean), features.shape[0])
        for i in range(features.shape[0]):
            v_normalized[i, ...] = (features[i, ...] - mean[i]) / std[i]
        if name == 'heur_action_priorities':
            for i in range(v_normalized.shape[0]):
                if np.isnan(v_normalized[i]):
                    v_normalized[i] = minv[i]
        else:
            v_normalized[np.isnan(v_normalized)] = 0
        return v_normalized

    return {k: normalize(k, v) for k, v in
            [('player_scalar_stats', fight2_player_scalar_stats(agent)),
             ('semantic_maps', fight2_semantic_maps(agent)),
             ('heur_action_priorities', fight2_encoded_heur_action_priorities(agent, heur_priorities))]}


def rl_communicate(agent, actions):
    action_priorities_for_rl = dict()
    for pr, action in actions:
        if action[0] == 'go_to':
            continue
        if action[0] == 'pickup':
            action = (action[0],)
        if action[0] == 'zap':
            action = action[:3]
        if action[0] not in ('zap', 'pickup'):
            assert action in agent._fight2_model.action_space, action
            action_priorities_for_rl[action] = pr
    observation = agent._fight2_get_observation(action_priorities_for_rl)

    # uncomment to gather features for get_observations_stats.py
    # import pickle
    # import base64
    # encoded = base64.b64encode(pickle.dumps(observation)).decode()
    # with open('/tmp/vis/observations.txt', 'a', buffering=1) as f:
    #     f.writelines([encoded + '\n'])

    priority, best_action = max(actions, key=lambda x: x[0]) if actions else None
    rl_action = agent._fight2_model.choose_action(agent, observation, list(action_priorities_for_rl.keys()))
    # TODO: use RL
    best_action = rl_action
    return best_action
