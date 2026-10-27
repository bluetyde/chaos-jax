"""H2 unit tests: grid, sampling, support classes, decode, screen. Added after the first main run used magnitudes up to 10
instead of the registered 5 (docs/PREDICTIONS-H2.md, Amendment 2). Names start with g5 so the planted-bug script selects them."""
from itertools import permutations

import numpy as np

from hyperchaos import search3d as s


def test_g5_grid_is_the_registered_100_values():
    assert len(s.GRID) == 50
    assert abs(s.GRID.max() - 5.0) < 1e-12 and abs(s.GRID.min() - 0.1) < 1e-12
    cs = s.sample(np.random.default_rng(1), 5000)
    vals = np.unique(np.abs(cs[cs != 0]))
    assert len(vals) == 50 and vals.max() <= 5.0 + 1e-12


def test_g5_sample_term_counts_and_support_distribution():
    cs = s.sample(np.random.default_rng(2), 20000)
    k = (cs != 0).sum(1)
    assert set(np.unique(k)) == {5, 6}
    assert abs((k == 5).mean() - 0.5) < 0.02
    # every one of the 30 slots is used about equally
    used = (cs != 0).mean(0)
    assert used.min() > 0.15 and used.max() < 0.22


def test_g5_published_cases_have_the_stated_term_counts():
    for case in "ABCDE":
        c = s.case_coefficients(case)
        assert (c != 0).sum() == 5 and sum(1 for i in np.flatnonzero(c) if i % 10 >= 4) == 2
    for case in "FGHIJKLMNOPQRS":
        c = s.case_coefficients(case)
        assert (c != 0).sum() == 6 and sum(1 for i in np.flatnonzero(c) if i % 10 >= 4) == 1


def test_g5_canonical_support_is_invariant_under_renaming_variables():
    for case in "BFS":
        c = s.case_coefficients(case)
        base = s.canonical_support(s.support_of(c))
        for p in permutations(range(3)):
            sup = frozenset(int(s._slot_perm(p)[i]) for i in s.support_of(c))
            assert s.canonical_support(sup) == base
        assert len(s.support_class(s.support_of(c))) == 6


def test_g5_flow_matches_the_hand_written_sprott_cases():
    x = np.array([0.7, -1.3, 0.4])
    X, Y, Z = x
    hand = {
        "B": [Y * Z, X - Y, 1 - X * Y],
        "F": [Y + Z, -X + 0.5 * Y, X * X - Z],
        "S": [-X - 4 * Y, X + Z * Z, 1 + X],
        "O": [Y, X - Z, X + X * Z + 2.7 * Y],
    }
    for case, want in hand.items():
        got = np.asarray(s.flow(x, s.case_coefficients(case)))
        assert np.allclose(got, want, atol=1e-12), (case, got, want)


def test_g5_stage1_keeps_chaos_and_drops_divergence():
    chaotic = np.array([s.case_coefficients("B"), s.case_coefficients("S")])
    div = np.zeros((1, 30))
    div[0, 0] = 1.0  # x' = 1: drifts to infinity linearly, but slowly; use a faster one below
    div[0, 4] = 1.0  # x' = 1 + x^2: blows up in finite time
    idx, lam, _ = s.screen(np.vstack([chaotic, div]))
    assert sorted(idx.tolist()) == [0, 1]


def _lin(rows):
    c = np.zeros(30)
    for e, m, v in rows:
        c[10 * e + s.NAMES.index(m)] = v
    return c


def test_g5_mid_filter_keeps_chaos():
    for case in "BS":
        assert s.mid_filter(s.case_coefficients(case)), case


def test_g5_mid_filter_drops_sink_divergence_and_periodic_orbit():
    sink = _lin([(0, "x", -1), (1, "y", -1), (2, "z", -1)])
    div = _lin([(0, "1", 1), (0, "xx", 1), (1, "y", -1), (2, "z", -1)])
    centre = _lin([(0, "y", 1), (1, "x", -1), (2, "z", -1)])
    for name, c in (("sink", sink), ("divergent", div), ("centre", centre)):
        assert not s.mid_filter(c), name


def test_g5_bound_keep_keeps_chaos():
    cs = np.array([s.case_coefficients("B"), s.case_coefficients("S")])
    assert s.bound_keep(cs).tolist() == [True, True]


def test_g5_bound_keep_drops_escape_blowup_and_sink():
    grow = _lin([(0, "x", 1), (1, "y", -1), (2, "z", -1)])  # x' = x: exponential escape, slower than finite-time blow-up
    blow = _lin([(0, "1", 1), (0, "xx", 1), (1, "y", -1), (2, "z", -1)])
    sink = _lin([(0, "x", -1), (1, "y", -1), (2, "z", -1)])
    assert s.bound_keep(np.array([grow, blow, sink])).tolist() == [False, False, False]


STIFF = [0.0] * 5 + [1.6, 0.0, 0.0, 0.0, 3.4, -4.5, 0.0, 0.0, 0.0, 2.5, 1.1, 0.0, 0.0, 4.5] + [0.0] * 11  # float64 label "inconsistent": exponent sum far from the divergence
NOZERO = [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1, 0.0, 0.0, -0.2, 0.0, 0.0, 3.9, 0.0, 0.0, 0.0]  # lambda_1 0.37, no exponent near zero


def test_g5_f32_keep_keeps_chaos():
    cs = np.array([s.case_coefficients("B"), s.case_coefficients("S")])
    assert s.f32_keep(cs).tolist() == [True, True]


def test_g5_f32_keep_drops_sink_centre_and_sum_mismatch():
    sink = _lin([(0, "x", -1), (1, "y", -1), (2, "z", -1)])
    centre = _lin([(0, "y", 1), (1, "x", -1), (2, "z", -1)])
    assert s.f32_keep(np.array([sink, centre])).tolist() == [False, False]


def test_g5_f32_sum_check_drops_stiff_system():
    assert s.f32_keep(np.array([STIFF])).tolist() == [False]


def test_g5_f32_zero_exponent_rule_drops_system_without_zero_exponent():
    assert s.f32_keep(np.array([NOZERO])).tolist() == [False]
