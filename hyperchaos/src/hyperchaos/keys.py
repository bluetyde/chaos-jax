"""Structure and parameter keys of 3D quadratic flows (docs/PREDICTIONS-H2-ENUM5.md).

structure key: the term-set and signs, minimised over the 48 renamings and sign flips.
parameter key: the scale-invariant coordinates (enumerate.invariants) of the representative that attains the minimum.
Two systems that differ only by renaming, sign flips and positive rescaling of variables and time have the same structure key and the same parameter key."""
import numpy as np

from hyperchaos import enumerate as en
from hyperchaos.orbits import images


def _structure_and_best(c):
    imgs = images(np.asarray(c, dtype=float))
    signs = [tuple(int(v) for v in np.sign(img)) for img in imgs]
    key = min(signs)
    best = [imgs[g] for g in range(len(imgs)) if signs[g] == key]
    return key, best


def structure_key(c):
    return _structure_and_best(c)[0]


def param_key(c):
    key, best = _structure_and_best(c)
    return min(tuple(np.round(en.invariants(img)[0], 3) + 0.0) for img in best)


def full_key(c):
    return structure_key(c), param_key(c)


def support_key(c):
    """Renaming class of the term-set alone (the 'canonical support' used for counting families)."""
    from hyperchaos import search3d as s

    return tuple(s.canonical_support(s.support_of(c)))
