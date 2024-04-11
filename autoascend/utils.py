import functools
from collections import Counter
from functools import partial, wraps
from itertools import chain

import cv2
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import seaborn as sns
import toolz

from .strategy import Strategy


@nb.njit(cache=True)
def bfs(y, x, *, walkable, walkable_diagonally, can_squeeze):
    dis = np.zeros(walkable.shape, dtype=np.int32)
    dis[:] = -1
    dis[y, x] = 0

    buf = np.zeros((walkable.shape[0] * walkable.shape[1], 2), dtype=np.uint32)
    index = 0
    buf[index] = (y, x)
    size = 1
    while index < size:
        y, x = buf[index]
        index += 1

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                py, px = y + dy, x + dx
                if 0 <= py < walkable.shape[0] and 0 <= px < walkable.shape[1] and (dy != 0 or dx != 0):
                    if (walkable[py, px] and
                            (abs(dy) + abs(dx) <= 1 or
                             (walkable_diagonally[py, px] and walkable_diagonally[y, x] and
                              (can_squeeze or walkable[py, x] or walkable[y, px])))):
                        if dis[py, px] == -1:
                            dis[py, px] = dis[y, x] + 1
                            buf[size] = (py, px)
                            size += 1

    return dis


def translate(array, y_offset, x_offset, out=None):
    if out is None:
        out = np.zeros_like(array)
    else:
        out.fill(0)

    if y_offset > 0:
        array = array[:-y_offset]
    elif y_offset < 0:
        array = array[-y_offset:]
    if x_offset > 0:
        array = array[:, :-x_offset]
    elif x_offset < 0:
        array = array[:, -x_offset:]

    sy, sx = max(y_offset, 0), max(x_offset, 0)
    out[sy: sy + array.shape[0], sx: sx + array.shape[1]] = array
    return out


@nb.njit('b1[:,:](i2[:,:],i2,i2,b1[:])', cache=True)
def _isin_kernel(array, mi, ma, mask):
    ret = np.zeros(array.shape, dtype=nb.b1)
    for y in range(array.shape[0]):
        for x in range(array.shape[1]):
            if array[y, x] < mi or array[y, x] > ma:
                continue
            ret[y, x] = mask[array[y, x] - mi]
    return ret


@functools.lru_cache(1024)
def _isin_mask(elems):
    elems = np.array(list(chain(*elems)), np.int16)
    return _isin_mask_kernel(elems)


@nb.njit('Tuple((i2,i2,b1[:]))(i2[:])', cache=True)
def _isin_mask_kernel(elems):
    mi: i2 = 32767
    ma: i2 = -32768
    for i in range(elems.shape[0]):
        if mi > elems[i]:
            mi = elems[i]
        if ma < elems[i]:
            ma = elems[i]
    ret = np.zeros(ma - mi + 1, dtype=nb.b1)
    for i in range(elems.shape[0]):
        ret[elems[i] - mi] = True
    return mi, ma, ret


def isin(array, *elems):
    assert array.dtype == np.int16

    # for memoization
    elems = tuple((
        e if isinstance(e, tuple) else
        e if isinstance(e, frozenset) else
        tuple(e) if isinstance(e, list) else
        frozenset(e) if isinstance(e, set) else
        e
        for e in elems))

    mi, ma, mask = _isin_mask(elems)
    return _isin_kernel(array, mi, ma, mask)


def any_in(array, *elems):
    # TODO: optimize
    return isin(array, *elems).any()


@toolz.curry
def debug_log(txt, fun, color=(255, 255, 255)):
    @wraps(fun)
    def wrapper(self, *args, **kwargs):
        # TODO: make it cleaner
        if type(self).__name__ != 'Agent':
            env = self.agent.env
        else:
            env = self.env

        with env.debug_log(txt=txt, color=color):
            ret = fun(self, *args, **kwargs)
            if isinstance(ret, Strategy):
                def f(strategy=ret.strategy, *a, **k):
                    it = strategy(*a, **k)
                    yield next(it)
                    with env.debug_log(txt=txt, color=color):
                        try:
                            next(it)
                            assert 0
                        except StopIteration as e:
                            return e.value

                ret.strategy = partial(f, ret.strategy)
            return ret

    return wrapper


def adjacent(p1, p2):
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1])) == 1


def calc_dps(to_hit, damage):
    return damage * min(20, max(0, (to_hit - 1))) / 20


@Strategy.wrap
def assert_strategy(error=None):
    yield True
    assert 0, error


def copy_result(func):
    @wraps(func)
    def f(*args, **kwargs):
        ret = func(*args, **kwargs)
        if isinstance(ret, list):
            return ret.copy()
        if isinstance(ret, tuple):
            return tuple((x.copy() if isinstance(x, list) else x for x in ret))
        return ret.copy()

    return f


def dilate(mask, radius=1, with_diagonal=True):
    d = radius * 2 + 1
    if with_diagonal:
        kernel = np.ones((d, d), dtype=np.uint8)
    else:
        kernel = np.zeros((d, d), dtype=np.uint8)
        kernel[radius: radius + 1, :] = 1
        kernel[:, radius: radius + 1] = 1
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def slice_with_padding(array, a1, a2, b1, b2, pad_value=0):
    ret = np.zeros_like(array, shape=(a2 - a1, b2 - b1)) + pad_value
    off_a1 = -a1 if a1 < 0 else 0
    off_b1 = -b1 if b1 < 0 else 0
    off_a2 = array.shape[0] - a2 if a2 > array.shape[0] else ret.shape[0]
    off_b2 = array.shape[1] - b2 if b2 > array.shape[1] else ret.shape[1]
    ret[off_a1: off_a2, off_b1: off_b2] = array[max(0, a1): a2, max(0, b1): b2]
    return ret


def slice_square_with_padding(array, center_y, center_x, radius, pad_value=0):
    return slice_with_padding(array, center_y - radius, center_y + radius + 1,
                              center_x - radius, center_x + radius + 1, pad_value=pad_value)


def plot_dashboard(fig, res):
    histogram_keys = ['score', 'steps', 'turns', 'level_num', 'experience_level', 'milestone']
    spec = fig.add_gridspec(len(histogram_keys) + 2, 2)
    for i, k in enumerate(histogram_keys):
        ax = fig.add_subplot(spec[i, 0])
        ax.set_title(k)
        if isinstance(res[k][0], str):
            counter = Counter(res[k])
            sns.barplot(x=[k for k, v in counter.most_common()], y=[v for k, v in counter.most_common()])
        else:
            if k in ['level_num', 'experience_level', 'milestone']:
                bins = [b + 0.5 for b in range(max(res[k]) + 1)]
            else:
                bins = np.quantile(res[k],
                                   np.linspace(0, 1, min(len(res[k]) // (20 + len(res[k]) // 50) + 2, 50)))
            sns.histplot(res[k], bins=bins, stat='density', ax=ax)
            if k == 'milestone':
                ticks = sorted(set([(int(m), str(m)) for m in res[k]]))
                plt.xticks(ticks=[t[0] for t in ticks], labels=[t[1] for t in ticks])
    ax = fig.add_subplot(spec[:len(histogram_keys) // 2, 1])
    sns.scatterplot(x='turns', y='steps', data=res, ax=ax)
    ax = fig.add_subplot(spec[len(histogram_keys) // 2: -2, 1])
    sns.scatterplot(x='turns', y='score', data=res, ax=ax)
    ax = fig.add_subplot(spec[-2:, :])
    res['role'] = [h.split('-')[0] for h in res['character']]
    res['race'] = [h.split('-')[1] for h in res['character']]
    res['gender'] = [h.split('-')[2] for h in res['character']]
    res['alignment'] = [h.split('-')[3] for h in res['character']]
    res['race-alignment'] = [f'{r}-{a}' for r, a in zip(res['race'], res['alignment'])]
    sns.violinplot(x='role', y='score', color='white', hue='gender',
                   hue_order=sorted(set(res['gender'])), split=len(set(res['gender'])) == 2,
                   order=sorted(set(res['role'])), inner='quartile',
                   data=res, ax=ax)
    palette = ['#ff7043', '#cc3311', '#ee3377', '#0077bb', '#33bbee', '#009988', '#bbbbbb']
    sns.swarmplot(x='role', y='score', hue='race-alignment', hue_order=sorted(set(res['race-alignment'])),
                  order=sorted(set(res['role'])),
                  data=res, ax=ax, palette=palette)
