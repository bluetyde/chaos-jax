"""Symmetry group of the 3D quadratic family: 6 renamings of (x, y, z) times 8 sign flips = 48 elements.

For a renaming `pi` (new variable i is old variable pi[i]) and signs `s`, the new variables are y_i = s_i * x_pi(i) and the new system is
g_i(y) = s_i * f_pi(i)(x(y)) with x_pi(i) = s_i * y_i. The result stays on the coefficient grid (only signs and positions change).
Coefficient vector layout is that of search3d: slot = 10 * equation + monomial index in MONOS.
"""
from itertools import permutations, product

import numpy as np

MONOS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]
_IDX = {m: i for i, m in enumerate(MONOS)}
GROUP = [(p, s) for p in permutations(range(3)) for s in product((1, -1), repeat=3)]


def _tables(p, s):
    """For group element (p, s): target slot and sign factor for each of the 30 source slots."""
    inv = [0] * 3
    for i, pi in enumerate(p):
        inv[pi] = i  # old variable j becomes new variable inv[j]
    dest = np.zeros(30, dtype=int)
    fac = np.zeros(30)
    for e in range(3):
        for m, a in enumerate(MONOS):
            new_e = inv[e]  # equation e (old variable e) becomes equation inv[e]
            b = [0, 0, 0]
            sign = float(s[new_e])  # derivative of y_i = s_i x_p(i)
            for j in range(3):
                b[inv[j]] = a[j]
                if a[j] % 2 == 1:
                    sign *= s[inv[j]]  # x_j = s_inv[j] * y_inv[j]
            dest[10 * e + m] = 10 * new_e + _IDX[tuple(b)]
            fac[10 * e + m] = sign
    return dest, fac


_TABLES = [_tables(p, s) for p, s in GROUP]


def images(c):
    """All 48 images of coefficient vector(s) c, shape (48, ..., 30)."""
    c = np.asarray(c, dtype=float)
    out = np.zeros((len(GROUP),) + c.shape)
    for g, (dest, fac) in enumerate(_TABLES):
        out[g][..., dest] = c * fac
    return out


def canonical(c):
    """Lexicographically smallest image of a single coefficient vector, as a tuple (rounded to 6 decimals)."""
    imgs = np.round(images(c), 6) + 0.0  # + 0.0 turns -0.0 into 0.0
    return min(tuple(v) for v in imgs)


def transform(c, g):
    """Image of coefficient vector c under group element number g."""
    dest, fac = _TABLES[g]
    out = np.zeros(30)
    out[dest] = np.asarray(c, dtype=float) * fac
    return out
