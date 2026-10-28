import numpy as np

from hyperchaos import enumerate as en
from hyperchaos import keys
from hyperchaos import orbits
from hyperchaos import search3d as s


def test_g4_keys_invariant_under_rescaling_renaming_and_flips():
    rng = np.random.default_rng(5)
    for case in "ABCDE":
        c = s.case_coefficients(case)
        for _ in range(4):
            g = int(rng.integers(0, 48))
            c2 = en.scale(orbits.transform(c, g), float(rng.uniform(0.5, 2)), rng.uniform(0.6, 1.6, size=3))
            assert keys.structure_key(c2) == keys.structure_key(c)
            assert keys.param_key(c2) == keys.param_key(c)


def test_g4_keys_distinguish_sprott_cases():
    ks = {case: keys.full_key(s.case_coefficients(case)) for case in "ABCDE"}
    assert len(set(ks.values())) == 5
    assert len({v[0] for v in ks.values()}) == 5


def test_g4_keys_flip_invariance_needs_all_48():
    # a sign flip alone must not change the structure key
    c = s.case_coefficients("D")
    flipped = [orbits.transform(c, g) for g in range(48)]
    assert len({keys.structure_key(f) for f in flipped}) == 1
