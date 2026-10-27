"""Unit tests of the label rules on synthetic spectra (added after planted bug 3 was not caught)."""
import numpy as np

from hyperchaos.classify import label_from_exponents


def lab(lam, se, rule="v2"):
    return label_from_exponents(np.array(lam), np.array(se), rule=rule)


def test_g3_classifier_one_positive_is_chaotic():
    assert lab([0.9, 0.001, 0.0, -14.0], [0.003, 0.001, 0.001, 0.003]).label == "chaotic"


def test_g3_classifier_small_second_exponent_is_not_positive():
    # 0.004 is below the absolute floor 0.005
    assert lab([0.1, 0.004, 0.0, -1.0], [0.001] * 4).label == "chaotic"


def test_g3_classifier_noisy_second_exponent_is_not_positive():
    # 3 * SE = 0.03 is above 0.02
    assert lab([0.1, 0.02, 0.0, -1.0], [0.001, 0.01, 0.001, 0.01]).label == "chaotic"


def test_g3_classifier_clear_second_exponent_is_hyperchaotic():
    assert lab([0.1, 0.02, 0.0, -1.0], [0.001] * 4).label == "hyperchaotic"


def test_g3_classifier_two_zero_exponents_is_quasi_periodic():
    assert lab([0.0003, 0.0001, -1.0, -2.0], [0.0005] * 4).label == "quasi-periodic"


def test_g3_classifier_one_zero_exponent_is_periodic():
    assert lab([-0.0002, -1.0], [0.0005, 0.001]).label == "periodic"


def test_g3_classifier_no_zero_exponent_is_inconsistent():
    assert lab([0.3, 0.2, -0.1, -1.0], [0.001] * 4).label == "inconsistent"


def test_g3_classifier_v2_ignores_large_se_of_a_stiff_exponent_v1_does_not():
    lam = [0.11, 0.02, 0.0, -25.0]
    se = [0.003, 0.002, 0.001, 0.8]
    assert lab(lam, se, "v2").label == "hyperchaotic"
    assert lab(lam, se, "v1").label == "quasi-periodic"


def _res(exps, mean_div, max_abs=10.0):
    from types import SimpleNamespace

    return SimpleNamespace(
        finite=True, max_abs=max_abs, late_range=np.array([1.0] * 4),
        exponents=np.array(exps), se=np.array([0.001] * len(exps)), mean_div=mean_div,
    )


def test_g3_classify_sum_mismatch_is_inconsistent_not_chaotic():
    from hyperchaos.classify import classify

    # exponents of a drifting orbit: first exponent large, sum far from the mean divergence
    assert classify(_res([162.0, 0.0, -0.0001, -97.98], -0.62, 1e5)).label == "inconsistent"
    # the same spectrum with a matching mean divergence is labelled by the usual rules
    assert classify(_res([0.147, 0.0, 0.0, -0.767], -0.62)).label == "chaotic"


def test_g3_classify_kopp_origin_start_is_not_chaotic():
    import hyperchaos  # noqa: F401
    from hyperchaos.classify import classify
    from hyperchaos.spectrum import spectrum
    from hyperchaos.systems import kopp25

    r = spectrum(kopp25(), np.zeros(4), dt=0.01, T=2000.0)
    assert classify(r).label != "chaotic"
