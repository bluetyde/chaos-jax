"""Enumeration of sparse 3D quadratic flows up to scaling (docs/PREDICTIONS-H2-FOURTERM.md).

Rescaling time and the three variables by positive factors maps a flow to another flow whose exponents are tau times the old ones,
so on a given support the coefficient magnitudes can be moved along the column space of the matrix M (one row per term)."""
import itertools

import numpy as np

from hyperchaos.search3d import MONOS


def slot_row(slot):
    i, m = divmod(int(slot), 10)
    r = np.zeros(4)
    r[0] = 1.0
    r[1:] = MONOS[m]
    r[1 + i] -= 1.0
    return r


def scale(c, tau, s):
    """Coefficients of the system y = x / s in time t / tau: c' = tau * c * s^e / s_i. Its exponents are tau times the original's."""
    c = np.asarray(c, dtype=float)
    s = np.asarray(s, dtype=float)
    out = np.zeros_like(c)
    for slot in np.flatnonzero(c):
        i, m = divmod(int(slot), 10)
        out[slot] = tau * c[slot] * np.prod(s ** np.asarray(MONOS[m])) / s[i]
    return out


def null_directions(slots):
    """Orthonormal basis (columns) of the scale-invariant directions of log|c| on this support, and the rank of M."""
    M = np.array([slot_row(sl) for sl in slots])
    u, sv, _ = np.linalg.svd(M)
    r = int(np.sum(sv > 1e-9))
    return u[:, r:], r


def flip_factors(slots):
    """(8, k) array: the factor each of the 8 variable sign flips applies to the coefficient of each slot (s_i times the product of s_j^e_j)."""
    out = []
    for flip in itertools.product((1.0, -1.0), repeat=3):
        row = []
        for sl in slots:
            i, m = divmod(int(sl), 10)
            row.append(flip[i] * float(np.prod(np.asarray(flip) ** np.asarray(MONOS[m]))))
        out.append(row)
    return np.array(out)


def reps(slots, t_grid, signs=True, dedupe_flips=False):
    """All coefficient vectors (length 30) for this support: invariants on t_grid (full product), all sign patterns.
    With dedupe_flips only one sign pattern per orbit of the 8 sign flips is kept (the lexicographically smallest)."""
    slots = sorted(int(x) for x in slots)
    k = len(slots)
    w, r = null_directions(slots)
    d = w.shape[1]
    ts = list(itertools.product(t_grid, repeat=d)) if d else [()]
    sg = list(itertools.product((-1.0, 1.0), repeat=k)) if signs else [(1.0,) * k]
    if dedupe_flips:
        F = flip_factors(slots)
        sg = [p for p in sg if tuple(p) == min(tuple(row) for row in F * np.asarray(p))]
    out = np.zeros((len(ts) * len(sg), 30))
    n = 0
    for t in ts:
        mag = np.exp(w @ np.asarray(t)) if d else np.ones(k)
        for sgn in sg:
            out[n, slots] = mag * np.asarray(sgn)
            n += 1
    return out


def invariants(c):
    """Scale-invariant coordinates (projection of log|c| on the null directions) of a vector."""
    slots = sorted(np.flatnonzero(c).tolist())
    w, r = null_directions(slots)
    return w.T @ np.log(np.abs(np.asarray(c)[slots])), r


def supports(k=4, all_equations=True):
    out = []
    for sup in itertools.combinations(range(30), k):
        if all_equations and len({sl // 10 for sl in sup}) < 3:
            continue
        out.append(sup)
    return out


def support_classes(k=5, all_equations=True):
    """One representative term-set per renaming class of k-slot supports (every equation nonempty)."""
    from hyperchaos import search3d as s3

    perms = list(itertools.permutations(range(3)))
    seen = {}
    for sup in itertools.combinations(range(30), k):
        if all_equations and len({sl // 10 for sl in sup}) < 3:
            continue
        seen.setdefault(s3.canonical_support(sup, perms), None)
    return sorted(seen)
