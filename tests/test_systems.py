import jax.numpy as jnp
import numpy as np

import chaos_jax  # noqa: F401
from chaos_jax.systems import (
    LORENZ_BETA, LORENZ_RHO, LORENZ_SIGMA, damped_oscillator, damped_oscillator_params,
    double_pendulum, double_pendulum_energy, lorenz, lorenz_params, pendulum_params,
)
import jax


def test_lorenz_parameters_are_the_plan_values():
    assert (LORENZ_SIGMA, LORENZ_RHO, LORENZ_BETA) == (10.0, 28.0, 8.0 / 3.0)


def test_lorenz_known_values():
    p = lorenz_params()
    # dx = 10 (2-1) = 10; dy = 1 (28-3) - 2 = 23; dz = 1*2 - (8/3) 3 = -6
    np.testing.assert_allclose(lorenz(jnp.array([1.0, 2.0, 3.0]), p), [10.0, 23.0, -6.0], rtol=1e-14)


def test_lorenz_fixed_points():
    p = lorenz_params()
    c = np.sqrt(LORENZ_BETA * (LORENZ_RHO - 1.0))
    for s in (1.0, -1.0):
        x = jnp.array([s * c, s * c, LORENZ_RHO - 1.0])
        np.testing.assert_allclose(lorenz(x, p), 0.0, atol=1e-12)
    np.testing.assert_allclose(lorenz(jnp.zeros(3), p), 0.0, atol=1e-15)


def test_lorenz_divergence_is_minus_sigma_plus_one_plus_beta():
    p = lorenz_params()
    for x in ([1.0, 2.0, 3.0], [-4.0, 5.0, 20.0]):
        jac = jax.jacfwd(lorenz)(jnp.array(x), p)
        np.testing.assert_allclose(jnp.trace(jac), -(LORENZ_SIGMA + 1.0 + LORENZ_BETA), rtol=1e-13)


def test_damped_oscillator_field():
    p = damped_oscillator_params(0.4, 4.0)
    np.testing.assert_allclose(damped_oscillator(jnp.array([2.0, 3.0]), p), [3.0, -0.4 * 3.0 - 8.0], rtol=1e-14)


def test_pendulum_rest_points_and_symmetry():
    p = pendulum_params()
    np.testing.assert_allclose(double_pendulum(jnp.zeros(4), p), 0.0, atol=1e-14)
    # Hanging straight down and straight up are both equilibria.
    np.testing.assert_allclose(double_pendulum(jnp.array([jnp.pi, jnp.pi, 0.0, 0.0]), p), 0.0, atol=1e-13)


def test_pendulum_small_angle_matches_linear_normal_mode():
    # Linearisation for l = m = 1: a1 = -2 g th1 + g th2, a2 = 2 g th1 - 2 g th2.
    p = pendulum_params()
    eps = 1e-6
    a = double_pendulum(jnp.array([eps, 0.0, 0.0, 0.0]), p)[2:]
    np.testing.assert_allclose(a / eps, [-2.0 * 9.81, 2.0 * 9.81], rtol=1e-5)
    a = double_pendulum(jnp.array([0.0, eps, 0.0, 0.0]), p)[2:]
    np.testing.assert_allclose(a / eps, [9.81, -2.0 * 9.81], rtol=1e-5)


def test_pendulum_energy_is_conserved_by_the_vector_field():
    # dE/dt = grad E . f must be zero at any state.
    p = pendulum_params(9.81, 1.0, 1.3, 1.0, 0.7)
    x = jnp.array([0.7, -1.1, 0.4, 1.9])
    dE = jnp.dot(jax.grad(double_pendulum_energy)(x, p), double_pendulum(x, p))
    assert abs(float(dE)) < 1e-12
