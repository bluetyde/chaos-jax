"""H2: random search over 3D quadratic flows (see docs/PREDICTIONS-H2.md for every definition).

Coefficient vector `c` has 30 entries: slot = 10 * equation + monomial, monomials ordered as MONOS.
"""
from functools import lru_cache
from itertools import permutations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import hyperchaos  # noqa: F401  (float64)
from hyperchaos.classify import classify
from hyperchaos.spectrum import spectrum_p

MONOS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]
NAMES = ["1", "x", "y", "z", "xx", "xy", "xz", "yy", "yz", "zz"]
import os

# Registered grid: magnitudes 0.1 .. 5.0 (50 values) with both signs = 100 nonzero values (docs/PREDICTIONS-H2.md).
# H2_GRID_MAX=10 reproduces the wide-grid run of 2026-10-27 (200 nonzero values), kept in results/h2_wide_grid/.
N_MAG = int(round(float(os.environ.get("H2_GRID_MAX", "5.0")) * 10))
GRID = np.round(np.arange(1, N_MAG + 1) * 0.1, 1)
START = np.array([0.05, 0.05, 0.05])
START2 = np.array([0.1, -0.05, 0.07])


def flow(x, c):
    feats = jnp.stack([1.0, x[0], x[1], x[2], x[0] * x[0], x[0] * x[1], x[0] * x[2], x[1] * x[1], x[1] * x[2], x[2] * x[2]])
    return c.reshape(3, 10) @ feats


# ---------- published cases (LITERATURE.md section C) ----------
_CASES = {
    "A": ["y", "-x+yz", "1-yy"],
    "B": ["yz", "x-y", "1-xy"],
    "C": ["yz", "x-y", "1-xx"],
    "D": ["-y", "x+z", "xz+3yy"],
    "E": ["yz", "xx-y", "1-4x"],
    "F": ["y+z", "-x+0.5y", "xx-z"],
    "G": ["0.4x+z", "xz-y", "-x+y"],
    "H": ["-y+zz", "x+0.5y", "x-z"],
    "I": ["-0.2y", "x+z", "x+yy-z"],
    "J": ["2z", "-2y+z", "-x+y+yy"],
    "K": ["xy-z", "x-y", "x+0.3z"],
    "L": ["y+3.9z", "0.9xx-y", "1-x"],
    "M": ["-z", "-xx-y", "1.7+1.7x+y"],
    "N": ["-2y", "x+zz", "1+y-2z"],
    "O": ["y", "x-z", "x+xz+2.7y"],
    "P": ["2.7y+z", "-x+yy", "x+y"],
    "Q": ["-z", "x-y", "3.1x+yy+0.5z"],
    "R": ["0.9-y", "0.4+z", "xy-z"],
    "S": ["-x-4y", "x+zz", "1+x"],
}
PUBLISHED_EXPONENTS = {
    "A": (0.014, 0, -0.014), "B": (0.210, 0, -1.210), "C": (0.163, 0, -1.163), "D": (0.103, 0, -1.320),
    "E": (0.078, 0, -1.078), "F": (0.117, 0, -0.617), "G": (0.034, 0, -0.634), "H": (0.117, 0, -0.617),
    "I": (0.012, 0, -1.012), "J": (0.076, 0, -2.076), "K": (0.038, 0, -0.890), "L": (0.061, 0, -1.061),
    "M": (0.044, 0, -1.044), "N": (0.076, 0, -2.076), "O": (0.049, 0, -0.319), "P": (0.087, 0, -0.481),
    "Q": (0.109, 0, -0.609), "R": (0.062, 0, -1.062), "S": (0.188, 0, -1.188),
}


def parse_equation(text):
    """'-x+0.5yz' -> {monomial index: coefficient}. A monomial name is a run of x, y, z or the constant."""
    out = {}
    i = 0
    s = text.replace(" ", "")
    while i < len(s):
        sign = 1.0
        if s[i] in "+-":
            sign = -1.0 if s[i] == "-" else 1.0
            i += 1
        j = i
        while j < len(s) and (s[j].isdigit() or s[j] == "."):
            j += 1
        num = float(s[i:j]) if j > i else None
        k = j
        while k < len(s) and s[k] in "xyz":
            k += 1
        mono = s[j:k]
        if mono == "":
            mono = "1"
            assert num is not None, text
        coef = sign * (1.0 if num is None else num)
        idx = NAMES.index("".join(sorted(mono)) if mono != "1" else "1")
        assert idx not in out, (text, mono)
        out[idx] = coef
        i = k
    return out


def case_coefficients(case):
    c = np.zeros(30)
    for e, text in enumerate(_CASES[case]):
        for m, v in parse_equation(text).items():
            c[10 * e + m] = v
    return c


# ---------- support classes ----------
def _perm_monomial(m, p):
    out = [0, 0, 0]
    for i in range(3):
        out[p[i]] = m[i]
    return tuple(out)


@lru_cache(maxsize=None)
def _slot_perm(p):
    """Slot permutation induced by renaming variable i to p[i]."""
    idx = {m: i for i, m in enumerate(MONOS)}
    perm = np.zeros(30, dtype=int)
    for e in range(3):
        for mi, m in enumerate(MONOS):
            perm[10 * e + mi] = 10 * p[e] + idx[_perm_monomial(m, p)]
    return perm


def support_of(c):
    return frozenset(np.flatnonzero(np.asarray(c) != 0).tolist())


def canonical_support(slots, perms=None):
    perms = list(permutations(range(3))) if perms is None else perms
    return min(tuple(sorted(int(_slot_perm(p)[s]) for s in slots)) for p in perms)


def support_class(slots, perms=None):
    perms = list(permutations(range(3))) if perms is None else perms
    return {tuple(sorted(int(_slot_perm(p)[s]) for s in slots)) for p in perms}


# ---------- sampling ----------
def sample(rng, n, ks=(5, 6)):
    """n coefficient vectors from the distribution D of docs/PREDICTIONS-H2.md."""
    k = rng.choice(np.asarray(ks), size=n)
    rank = np.argsort(rng.random((n, 30)), axis=1).argsort(axis=1)
    mask = rank < k[:, None]
    mag = GRID[rng.integers(0, N_MAG, size=(n, 30))]
    sign = np.where(rng.random((n, 30)) < 0.5, -1.0, 1.0)
    return np.where(mask, mag * sign, 0.0)


def sample_on_support(rng, slots, n):
    slots = sorted(slots)
    c = np.zeros((n, 30))
    mag = GRID[rng.integers(0, N_MAG, size=(n, len(slots)))]
    sign = np.where(rng.random((n, len(slots))) < 0.5, -1.0, 1.0)
    c[:, slots] = mag * sign
    return c


# ---------- stage 1 (cheap screen) ----------
def _rk4(f, x, dt):
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


@lru_cache(maxsize=None)
def _stage1a(dt=0.02, steps_a=1600, steps_b=400):
    def one(c):
        f = lambda x: flow(x, c)  # noqa: E731

        def step_a(carry, _):
            x, m = carry
            x = _rk4(f, x, dt)
            return (x, jnp.maximum(m, jnp.max(jnp.abs(x)))), None

        def step_b(carry, _):
            x, m, lo, hi = carry
            x = _rk4(f, x, dt)
            return (x, jnp.maximum(m, jnp.max(jnp.abs(x))), jnp.minimum(lo, x), jnp.maximum(hi, x)), None

        (x, m), _ = lax.scan(step_a, (jnp.asarray(START), 0.0), None, length=steps_a)
        (x, m, lo, hi), _ = lax.scan(step_b, (x, m, x, x), None, length=steps_b)
        moving = jnp.max(hi - lo) > 1e-3
        return (m <= 1e4) & moving

    return jax.jit(jax.vmap(one))


@lru_cache(maxsize=None)
def _stage1b(dt=0.02, inner=25, outer=1200, spin=200):
    def one(c):
        f = lambda x: flow(x, c)  # noqa: E731

        def aug(x, w):
            fx, fw = jax.jvp(f, (x,), (w,))
            return fx, fw

        def step(carry, _):
            x, w = carry
            k1x, k1w = aug(x, w)
            k2x, k2w = aug(x + 0.5 * dt * k1x, w + 0.5 * dt * k1w)
            k3x, k3w = aug(x + 0.5 * dt * k2x, w + 0.5 * dt * k2w)
            k4x, k4w = aug(x + dt * k3x, w + dt * k3w)
            x = x + dt / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
            w = w + dt / 6.0 * (k1w + 2 * k2w + 2 * k3w + k4w)
            return (x, w), None

        def chunk(carry, i):
            x, w, acc, m = carry
            (x, w), _ = lax.scan(step, (x, w), None, length=inner)
            nrm = jnp.linalg.norm(w)
            acc = acc + jnp.where(i >= spin, jnp.log(nrm), 0.0)
            return (x, w / nrm, acc, jnp.maximum(m, jnp.max(jnp.abs(x)))), None

        w0 = jnp.ones(3) / jnp.sqrt(3.0)
        (x, w, acc, m), _ = lax.scan(chunk, (jnp.asarray(START), w0, 0.0, 0.0), jnp.arange(outer))
        t_meas = (outer - spin) * inner * dt
        return acc / t_meas, m <= 1e4

    return jax.jit(jax.vmap(one))


def stage1a(cs):
    return np.asarray(_stage1a()(jnp.asarray(cs)))


def stage1b(cs, batch=256):
    """Largest-exponent estimate; padded to fixed batches so the compile is reused."""
    cs = np.asarray(cs)
    lam = np.empty(len(cs))
    ok = np.empty(len(cs), dtype=bool)
    for i in range(0, len(cs), batch):
        part = cs[i : i + batch]
        pad = np.zeros((batch, 30))
        pad[: len(part)] = part
        l, o = _stage1b()(jnp.asarray(pad))
        lam[i : i + len(part)] = np.asarray(l)[: len(part)]
        ok[i : i + len(part)] = np.asarray(o)[: len(part)]
    return lam, ok


LAMBDA_KEEP = 0.01


def screen(cs, batch_a=2048):
    """Stage 1a then 1b. Returns indices (into cs) that survive and their largest-exponent estimates."""
    cs = np.asarray(cs)
    keep_a = np.zeros(len(cs), dtype=bool)
    for i in range(0, len(cs), batch_a):
        part = cs[i : i + batch_a]
        pad = np.zeros((batch_a, 30))
        pad[: len(part)] = part
        keep_a[i : i + len(part)] = stage1a(pad)[: len(part)]
    idx = np.flatnonzero(keep_a)
    if len(idx) == 0:
        return idx, np.empty(0), keep_a
    lam, ok = stage1b(cs[idx])
    good = ok & (lam > LAMBDA_KEEP)
    return idx[good], lam[good], keep_a


# ---------- stages 2 and confirmation (H1 evaluator) ----------
def evaluate(c, x0=START, T=5000.0, dt=0.01):
    r = spectrum_p(flow, np.asarray(c), x0, dt=dt, T=T)
    return r, classify(r, rule="v2")


def is_chaotic(c, T=5000.0, x0=START):
    r, k = evaluate(c, x0, T)
    return k.label == "chaotic" and not k.conservative, r


def confirm(c, T=20000.0):
    ok1, r1 = is_chaotic(c, T, START)
    if not ok1:
        return False, r1
    ok2, _ = is_chaotic(c, T, START2)
    return ok2, r1


# ---------- middle filter (H2B, docs/PREDICTIONS-H2B.md) ----------
MID_TIERS = ((300.0, 100.0), (1000.0, 200.0))  # (T, spin) of each tier
MID_EPS = 0.005


def mid_drop(res):
    """True if a short run clearly fails: diverges, fixed point, or the largest exponent is not clearly above zero."""
    if classify(res, rule="v2").label in ("diverges", "fixed point"):
        return True
    return bool(res.exponents[0] + 3.0 * res.se[0] <= MID_EPS)


def mid_filter(c, tiers=MID_TIERS):
    """True if the system survives every tier; a failing system stops at the first tier that drops it."""
    for T, spin in tiers:
        r = spectrum_p(flow, np.asarray(c), START, dt=0.01, T=T, T_spin=spin)
        if mid_drop(r):
            return False
    return True
