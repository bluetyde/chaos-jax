"""G1.1 (RK4 order) and G1.2 (autodiff against finite difference). Controls come first."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

import chaos_jax  # noqa: F401
from chaos_jax import cases
from chaos_jax.integrate import integrate, integrate_final
from chaos_jax.systems import (
    damped_oscillator, damped_oscillator_params, double_pendulum, lorenz, lorenz_params,
    pendulum_params,
)
from chaos_jax.tangent import state_jacobian


def _logistic(x, p):
    return p[0] * x * (1.0 - x)


def _logistic_exact(x0, r, t):
    return x0 * np.exp(r * t) / (1.0 - x0 + x0 * np.exp(r * t))


def _osc_matrix(p):
    c, k = float(p[0]), float(p[1])
    return np.array([[0.0, 1.0], [-k, -c]])


def _orders(errors):
    return [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]


# ---- G1.1: RK4 convergence order is in [3.8, 4.2] on smooth non-chaotic ODEs ----

DT_LADDER = [0.1, 0.05, 0.025, 0.0125]


def test_g1_1_order_damped_oscillator():
    p = damped_oscillator_params()
    x0 = jnp.array(cases.OSCILLATOR_START)
    exact = expm(_osc_matrix(p) * 1.0) @ np.array(cases.OSCILLATOR_START)
    errs = []
    for dt in DT_LADDER:
        x = integrate_final(damped_oscillator, x0, p, dt, int(round(1.0 / dt)))
        errs.append(np.linalg.norm(np.array(x) - exact))
    orders = _orders(errs)
    assert all(3.8 <= o <= 4.2 for o in orders), (orders, errs)


def test_g1_1_order_logistic():
    p = jnp.array([1.5])
    x0 = jnp.array([0.1])
    exact = _logistic_exact(0.1, 1.5, 3.0)
    errs = []
    for dt in DT_LADDER:
        x = integrate_final(_logistic, x0, p, dt, int(round(3.0 / dt)))
        errs.append(abs(float(x[0]) - exact))
    orders = _orders(errs)
    assert all(3.8 <= o <= 4.2 for o in orders), (orders, errs)


def test_integrate_matches_integrate_final_and_includes_start():
    p = lorenz_params()
    x0 = jnp.array([1.0, 2.0, 3.0])
    xs = integrate(lorenz, x0, p, 0.01, 200)
    assert xs.shape == (201, 3)
    np.testing.assert_array_equal(xs[0], x0)
    np.testing.assert_allclose(xs[-1], integrate_final(lorenz, x0, p, 0.01, 200), rtol=1e-14)


# ---- G1.2: autodiff vs central finite difference, T = 1, float64 ----

H = 1e-5


def _fd_jacobian(f, x0, p, dt, n):
    cols = []
    for i in range(x0.shape[0]):
        e = jnp.zeros_like(x0).at[i].set(H)
        cols.append((integrate_final(f, x0 + e, p, dt, n) - integrate_final(f, x0 - e, p, dt, n)) / (2 * H))
    return jnp.stack(cols, axis=1)


def _rel_err(a, b):
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


def test_g1_2_oscillator_jacobian_control():
    p = damped_oscillator_params()
    x0 = jnp.array(cases.OSCILLATOR_START)
    dt, n = 0.01, 100
    ad = state_jacobian(damped_oscillator, x0, p, dt, n)
    fd = _fd_jacobian(damped_oscillator, x0, p, dt, n)
    assert _rel_err(ad, fd) < 1e-6
    # The exact answer is the matrix exponential. The gap is the RK4 error, far below 1e-6 at dt = 0.01.
    assert _rel_err(ad, expm(_osc_matrix(p) * 1.0)) < 1e-6


def test_g1_2_lorenz_jacobian():
    p = lorenz_params()
    x0 = jnp.array([1.0, 2.0, 3.0])
    dt, n = cases.LORENZ_DT, 100
    ad = state_jacobian(lorenz, x0, p, dt, n)
    fd = _fd_jacobian(lorenz, x0, p, dt, n)
    assert _rel_err(ad, fd) < 1e-6


def test_g1_2_lorenz_parameter_gradient():
    x0 = jnp.array([1.0, 2.0, 3.0])
    n = 100

    def zT(rho):
        p = lorenz_params(rho=rho)
        return integrate_final(lorenz, x0, p, cases.LORENZ_DT, n)[2]

    ad = jax.grad(zT)(28.0)
    fd = (zT(28.0 + H) - zT(28.0 - H)) / (2 * H)
    assert abs(float(ad - fd)) / abs(float(fd)) < 1e-6


def test_g1_2_pendulum_jacobian_extra():
    p = pendulum_params()
    x0 = jnp.array(cases.PENDULUM_HIGH)
    n = int(round(1.0 / cases.PENDULUM_DT))
    ad = state_jacobian(double_pendulum, x0, p, cases.PENDULUM_DT, n)
    fd = _fd_jacobian(double_pendulum, x0, p, cases.PENDULUM_DT, n)
    assert _rel_err(ad, fd) < 1e-6
