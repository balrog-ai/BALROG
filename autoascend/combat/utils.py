def wielding_ranged_weapon(agent):
    for item in agent.inventory.items:
        if item.is_launcher() and item.equipped:
            return True
    return False


def wielding_melee_weapon(agent):
    for item in agent.inventory.items:
        if item.is_weapon() and item.equipped:
            return True
    return False


def line_dis_from(agent, y, x):
    return max(abs(agent.blstats.x - x), abs(agent.blstats.y - y))


def inside(agent, y, x):
    return 0 <= y < agent.glyphs.shape[0] and 0 <= x < agent.glyphs.shape[1]


def action_str(agent, action):
    priority, a = action
    if a[0] == 'move':
        return f'{priority}m:{a[1]},{a[2]}'
    elif a[0] == 'melee':
        return f'{priority}me:{a[1]},{a[2]}'
    elif a[0] == 'pickup':
        return f'{priority}{a[0][0]}:{len(a[1])}'
    elif a[0] == 'zap':
        wand = a[3]
        letter = agent.inventory.items.get_letter(wand)
        return f'{priority}z{letter}:{a[1]},{a[2]}'
    elif a[0] == 'elbereth':
        return f'{priority:.1f}e'
    elif a[0] == 'wait':
        return f'{priority:.1f}w'
    elif a[0] == 'go_to':
        return f'{priority}goto:{a[1]},{a[2]}'
    else:
        return f'{priority}{a[0][0]}:{a[1]},{a[2]}'
