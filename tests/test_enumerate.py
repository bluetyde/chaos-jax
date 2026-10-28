import numpy as np

from hyperchaos import enumerate as en
from hyperchaos import search3d as s
from hyperchaos.spectrum import spectrum_p


def test_g4_enumerate_scaling_multiplies_exponents_by_tau():
    c = s.case_coefficients("B")
    tau, sc = 1.3, [0.8, 1.25, 0.9]
    c2 = en.scale(c, tau, sc)
    assert np.array_equal(c2 != 0, c != 0)
    r1 = spectrum_p(s.flow, c, s.START, T=2000.0)
    r2 = spectrum_p(s.flow, c2, s.START, T=2000.0)
    assert np.allclose(r2.exponents, tau * r1.exponents, atol=0.05)


def test_g4_enumerate_reps_hold_invariants_and_cover_signs():
    slots = (3, 6, 12, 25)
    w, r = en.null_directions(slots)
    assert w.shape[1] == 4 - r
    cs = en.reps(slots, [-1.0, 0.0, 1.0])
    assert cs.shape[0] == (3 ** (4 - r)) * 16
    assert all(set(np.flatnonzero(c)) == set(slots) for c in cs)
    # every representative of a rank-4 support has all magnitudes 1
    full = [sup for sup in en.supports(4) if en.null_directions(sup)[0].shape[1] == 0][0]
    assert np.allclose(np.abs(en.reps(full, [0.0])[:, list(full)]), 1.0)


def test_g4_enumerate_invariants_are_scale_invariant():
    c = s.case_coefficients("A")
    c2 = en.scale(c, 1.7, [0.6, 1.4, 1.1])
    a, _ = en.invariants(c)
    b, _ = en.invariants(c2)
    assert np.allclose(a, b)


def test_g4_enumerate_support_counts():
    sups = en.supports(4)
    ranks = {}
    for sup in sups:
        ranks[en.null_directions(sup)[1]] = ranks.get(en.null_directions(sup)[1], 0) + 1
    assert len(sups) == 13500 and ranks == {4: 8346, 3: 4881, 2: 273}
