"""Orbit group tests (names start with g5 so the planted-bug script selects them). See docs/PREDICTIONS-H2F.md, GF1."""
import numpy as np

from hyperchaos import orbits
from hyperchaos import search3d as s


def _flow_np(x, c):
    feats = np.array([1.0, x[0], x[1], x[2], x[0] * x[0], x[0] * x[1], x[0] * x[2], x[1] * x[1], x[1] * x[2], x[2] * x[2]])
    return c.reshape(3, 10) @ feats


def test_g5_group_has_48_elements():
    assert len(orbits.GROUP) == 48


def test_g5_transformed_flow_equals_the_change_of_variables():
    rng = np.random.default_rng(0)
    for _ in range(5):
        c = s.sample(rng, 1)[0]
        for g, (p, sg) in enumerate(orbits.GROUP):
            cg = orbits.transform(c, g)
            for _ in range(10):
                y = rng.normal(size=3)
                x = np.zeros(3)
                for i in range(3):
                    x[p[i]] = sg[i] * y[i]  # x_p(i) = s_i y_i
                want = np.array([sg[i] * _flow_np(x, c)[p[i]] for i in range(3)])
                assert np.allclose(_flow_np(y, cg), want, atol=1e-12), (g, p, sg)


def test_g5_canonical_form_is_the_same_for_all_48_images():
    rng = np.random.default_rng(1)
    for c in s.sample(rng, 20):
        base = orbits.canonical(c)
        for img in orbits.images(c):
            assert orbits.canonical(img) == base


def test_g5_generic_vector_has_48_distinct_images_and_a_changed_vector_a_different_orbit():
    rng = np.random.default_rng(3)
    c = rng.normal(size=30)  # all 30 slots filled with distinct values: no system symmetry, so the orbit has 48 members
    assert len({tuple(np.round(v, 6)) for v in orbits.images(c)}) == 48
    d = c.copy()
    d[np.flatnonzero(d)[0]] += 0.1
    assert orbits.canonical(c) != orbits.canonical(d)


def test_g5_weighted_sampler_with_zero_theta_is_the_uniform_sampler():
    from hyperchaos import weighted

    wts = {5: weighted.Weights(np.zeros(30), 1.0, 0.1, 5), 6: weighted.Weights(np.zeros(30), 1.0, 0.1, 6)}
    cs = weighted.sample_weighted(np.random.default_rng(5), 20000, wts)
    k = (cs != 0).sum(1)
    assert set(np.unique(k)) == {5, 6} and abs((k == 5).mean() - 0.5) < 0.02
    used = (cs != 0).mean(0)
    assert used.min() > 0.15 and used.max() < 0.22
    assert len(np.unique(np.abs(cs[cs != 0]))) == 50


def test_g5_weighted_sampler_prefers_slots_with_positive_theta_and_keeps_the_term_mix():
    from hyperchaos import weighted

    th = np.zeros(30)
    th[3] = 2.0
    th[17] = -2.0
    wts = {5: weighted.Weights(th, 8.0, 0.1, 5), 6: weighted.Weights(th, 8.0, 0.1, 6)}
    cs = weighted.sample_weighted(np.random.default_rng(6), 20000, wts)
    used = (cs != 0).mean(0)
    base = np.median(used)
    assert used[3] > 1.5 * base and used[17] < 0.7 * base
    assert abs(((cs != 0).sum(1) == 5).mean() - 0.5) < 0.02
