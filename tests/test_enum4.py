import numpy as np

from hyperchaos import enum4 as e4
from hyperchaos import systems
from hyperchaos.spectrum import spectrum_p


def _c_from(terms):
    c = np.zeros(e4.N * e4.NM)
    for eq, mono, v in terms:
        c[e4.NM * eq + e4._MIDX[mono]] = v
    return c


def _li_sprott(a=2.6, b=0.44):
    one = lambda i: tuple(1 if j == i else 0 for j in range(4))
    two = lambda i, j: tuple((1 if k == i else 0) + (1 if k == j else 0) for k in range(4))
    return _c_from([(0, one(1), 1.0), (0, one(0), -1.0), (1, two(0, 2), -1.0), (1, one(3), 1.0), (2, two(0, 1), 1.0), (2, (0, 0, 0, 0), -a), (3, one(1), -b)])


def test_g4_enum4_flow_matches_li_sprott():
    c = _li_sprott()
    ref = systems.li_sprott()
    x = np.array([0.3, -0.7, 1.1, 0.4])
    assert np.allclose(np.asarray(e4.flow4(x, c)), np.asarray(ref(x)))


def test_g4_enum4_scaling_multiplies_exponents_by_tau():
    c = _li_sprott()
    tau, sc = 1.25, [0.9, 1.2, 0.8, 1.1]
    c2 = e4.scale(c, tau, sc)
    assert np.array_equal(c2 != 0, c != 0)
    r1 = spectrum_p(e4.flow4, c, np.array([1.0, 1.0, 1.0, 1.0]), T=2000.0)
    r2 = spectrum_p(e4.flow4, c2, np.array([1.0, 1.0, 1.0, 1.0]) / np.array(sc), T=2000.0)
    assert np.allclose(r2.exponents, tau * r1.exponents, atol=0.06)


def test_g4_enum4_canonical_support_is_renaming_invariant():
    from itertools import permutations

    rng = np.random.default_rng(3)
    for _ in range(5):
        sup = tuple(sorted(rng.choice(e4.N * e4.NM, size=4, replace=False).tolist()))
        base = e4.canonical_support(sup)
        for p in permutations(range(4)):
            assert e4.canonical_support(tuple(int(e4.slot_perm(p)[s]) for s in sup)) == base


def test_g4_enum4_rank_counts():
    sups = [tuple(m + e4.NM * eq for eq, m in enumerate(ms)) for ms in __import__("itertools").product(range(15), repeat=4)]
    ranks = {}
    for sup in sups:
        r = e4.null_directions(sup)[1]
        ranks[r] = ranks.get(r, 0) + 1
    assert ranks == {4: 41273, 3: 8869, 2: 478, 1: 5}
