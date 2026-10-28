"""4D quadratic flows with sparse supports, enumerated up to scaling (docs/PREDICTIONS-H3-FOURTERM-4D.md).

Coefficient vector c has 60 entries: slot = 15 * equation + monomial, monomials ordered as MONOS4. Same scaling argument as `enumerate.py`."""
import itertools
from functools import lru_cache
from itertools import permutations

import jax.numpy as jnp
import numpy as np

import hyperchaos  # noqa: F401  (float64)

N = 4
MONOS4 = [tuple([0] * N)] + [tuple(1 if j == i else 0 for j in range(N)) for i in range(N)] + [
    tuple((1 if j == a else 0) + (1 if j == b else 0) for j in range(N)) for a in range(N) for b in range(a, N)]
NM = len(MONOS4)
START4 = np.array([0.05, 0.05, 0.05, 0.05])
START4B = np.array([0.1, -0.05, 0.07, 0.03])


def flow4(x, c):
    feats = jnp.stack([jnp.ones(())] + [x[i] for i in range(N)] + [x[a] * x[b] for a in range(N) for b in range(a, N)])
    return c.reshape(N, NM) @ feats


def slot_row(slot):
    i, m = divmod(int(slot), NM)
    r = np.zeros(N + 1)
    r[0] = 1.0
    r[1:] = MONOS4[m]
    r[1 + i] -= 1.0
    return r


def scale(c, tau, s):
    """c' = tau * c * s^e / s_i: the system y = x / s in time t / tau; its exponents are tau times the original's."""
    c = np.asarray(c, dtype=float)
    s = np.asarray(s, dtype=float)
    out = np.zeros_like(c)
    for slot in np.flatnonzero(c):
        i, m = divmod(int(slot), NM)
        out[slot] = tau * c[slot] * np.prod(s ** np.asarray(MONOS4[m])) / s[i]
    return out


def null_directions(slots):
    M = np.array([slot_row(sl) for sl in slots])
    u, sv, _ = np.linalg.svd(M)
    r = int(np.sum(sv > 1e-9))
    return u[:, r:], r


def reps(slots, t_grid):
    slots = sorted(int(x) for x in slots)
    k = len(slots)
    w, r = null_directions(slots)
    d = w.shape[1]
    ts = list(itertools.product(t_grid, repeat=d)) if d else [()]
    sg = list(itertools.product((-1.0, 1.0), repeat=k))
    out = np.zeros((len(ts) * len(sg), N * NM))
    n = 0
    for t in ts:
        mag = np.exp(w @ np.asarray(t)) if d else np.ones(k)
        for sgn in sg:
            out[n, slots] = mag * np.asarray(sgn)
            n += 1
    return out


_MIDX = {m: i for i, m in enumerate(MONOS4)}


def _perm_monomial(m, p):
    out = [0] * N
    for i in range(N):
        out[p[i]] = m[i]
    return tuple(out)


@lru_cache(maxsize=None)
def slot_perm(p):
    """Slot permutation induced by renaming variable i to p[i]."""
    perm = np.zeros(N * NM, dtype=int)
    for e in range(N):
        for mi, m in enumerate(MONOS4):
            perm[NM * e + mi] = NM * p[e] + _MIDX[_perm_monomial(m, p)]
    return perm


def canonical_support(slots):
    return min(tuple(sorted(int(slot_perm(p)[s]) for s in slots)) for p in permutations(range(N)))


def supports4(k=4):
    """Supports of k slots with every equation nonempty, one canonical representative per renaming class."""
    seen = {}
    for sup in itertools.combinations(range(N * NM), k):
        if len({sl // NM for sl in sup}) < N:
            continue
        cs = canonical_support(sup)
        seen.setdefault(cs, None)
    return sorted(seen)
