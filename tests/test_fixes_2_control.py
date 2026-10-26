"""Phase 4 control: every method must match the closed-form derivative to 1e-4. Runs before any Lorenz test."""
import numpy as np
import pytest

import chaos_jax  # noqa: F401
from chaos_jax import controls

TOL = 1e-4
EXACT = controls.exact_derivative()
X0S = controls.start_points()


def _rel(v):
    return abs(v - EXACT) / abs(EXACT)


def test_exact_value_is_the_phase_2_number():
    np.testing.assert_allclose(EXACT, 0.37098692, rtol=1e-7)


def test_control_a_naive():
    errs = [_rel(controls.naive_or_truncated(x)) for x in X0S]
    print("naive control errors:", errs)
    assert max(errs) < TOL


def test_control_b_truncated_with_whole_window_equals_naive():
    x = X0S[0]
    assert controls.naive_or_truncated(x, seg_time=1e9) == controls.naive_or_truncated(x)


def test_control_c_ensemble_mean_of_naive():
    est = np.mean([controls.naive_or_truncated(x) for x in X0S])
    assert _rel(est) < TOL


@pytest.mark.xfail(strict=True, reason=(
    "Failed gate, reported not adjusted: LSS on a finite window has an O(1/T) bias. Measured 4.5e-2 at 10 periods. "
    "See test_control_d_lss_error_falls_like_one_over_T and HANDOFF.md."))
def test_control_d_lss():
    errs = [_rel(controls.lss(x)) for x in X0S]
    print("lss control errors:", errs)
    assert max(errs) < TOL


def test_control_d_lss_error_falls_like_one_over_T():
    """Evidence that the 4.5% control error is a finite-window effect and not a coding error."""
    e10 = controls.lss_scaling_error(10, 200)
    e40 = controls.lss_scaling_error(40, 200)
    e160 = controls.lss_scaling_error(160, 200)
    print("lss control error at 10, 40, 160 periods:", e10, e40, e160)
    assert 0.15 < e40 / e10 < 0.35
    assert 0.15 < e160 / e40 < 0.35
    assert e160 < 5e-3
