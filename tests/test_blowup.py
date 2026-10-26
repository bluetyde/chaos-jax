import jax
import jax.numpy as jnp
import numpy as np

import chaos_jax  # noqa: F401
from chaos_jax.blowup import log_jacobian_norms, lorenz_time_average_z, naive_gradients
from chaos_jax.integrate import integrate_final
from chaos_jax.systems import damped_oscillator, damped_oscillator_params, lorenz, lorenz_params
from chaos_jax.tangent import state_jacobian

X0 = jnp.array([1.0, 2.0, 25.0])
DT = 0.01


def test_control_damped_oscillator_log_norm_slope_is_minus_half_c():
    p = damped_oscillator_params(0.4, 4.0)
    logs = log_jacobian_norms(damped_oscillator, jnp.array([1.0, 0.5]), p, 0.01, 40, 50)
    t = 0.5 * np.arange(1, 41)
    sel = t >= 2.0
    slope = np.polyfit(t[sel], np.asarray(logs)[sel], 1)[0]
    assert abs(slope - (-0.2)) < 0.02, slope


def test_log_norm_history_matches_jacfwd_through_the_scan():
    p = lorenz_params()
    logs = log_jacobian_norms(lorenz, X0, p, DT, 8, 50)  # t = 0.5 .. 4
    J = state_jacobian(lorenz, X0, p, DT, 200)  # t = 2
    np.testing.assert_allclose(float(logs[3]), float(jnp.log(jnp.linalg.norm(J, 2))), rtol=1e-9)


def test_naive_gradient_matches_forward_mode_in_rho():
    n = 200
    rev = naive_gradients(X0[None], 28.0, DT, n)[0]
    fwd = jax.jvp(lambda r: lorenz_time_average_z(r, X0, DT, n), (28.0,), (1.0,))[1]
    np.testing.assert_allclose(float(rev), float(fwd), rtol=1e-9)


def test_naive_gradient_matches_central_difference_at_T1():
    n, h = 100, 1e-5
    rev = float(naive_gradients(X0[None], 28.0, DT, n)[0])
    f = lambda r: float(lorenz_time_average_z(r, X0, DT, n))
    fd = (f(28.0 + h) - f(28.0 - h)) / (2 * h)
    assert abs(rev - fd) / abs(fd) < 1e-6
