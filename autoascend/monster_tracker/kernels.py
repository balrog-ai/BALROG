import numba as nb
import numpy as np


@nb.njit('b1[:,:](i2[:,:],i2[:,:],i4)', cache=True)
def disappearance_mask(old_mons, new_mons, max_radius):
    ret = np.zeros_like(new_mons, dtype=nb.b1)
    for y in range(new_mons.shape[0]):
        for x in range(new_mons.shape[1]):
            glyph = old_mons[y, x]
            if glyph == -1:
                continue
            ret[y, x] = (new_mons[max(0, y - max_radius): min(y + max_radius + 1, new_mons.shape[0]),
                         max(0, x - max_radius): min(x + max_radius + 1, new_mons.shape[1])] != glyph).all()
    return ret


@nb.njit('optional(b1[:,:])(i2[:,:],i2[:,:],i2[:,:],i4)', cache=True)
def figure_out_monster_movement(peaceful_mons, aggressive_mons, new_mons, max_radius):
    ret_peaceful_mons = np.zeros_like(peaceful_mons, dtype=nb.b1)
    for y in range(new_mons.shape[0]):
        for x in range(new_mons.shape[1]):
            glyph = new_mons[y, x]
            if glyph == -1:
                continue

            can_be_peaceful = False
            can_be_aggressive = False
            for py in range(max(0, y - max_radius),
                            min(y + max_radius + 1, new_mons.shape[0])):
                for px in range(max(0, x - max_radius),
                                min(x + max_radius + 1, new_mons.shape[1])):
                    if peaceful_mons[py, px] == glyph:
                        can_be_peaceful = True
                    if aggressive_mons[py, px] == glyph:
                        can_be_aggressive = True
            if can_be_peaceful == can_be_aggressive:
                return None
            if can_be_peaceful:
                ret_peaceful_mons[y, x] = True

    return ret_peaceful_mons
