"""Chaotic cases: G1.3 (Lorenz) and G1.5 (double pendulum). Run after the controls."""
from functools import lru_cache

import jax.numpy as jnp
import numpy as np

import chaos_jax  # noqa: F401
from chaos_jax import cases
from chaos_jax.integrate import integrate, integrate_final
from chaos_jax.lyapunov import lyapunov_spectrum, lyapunov_two_trajectory
from chaos_jax.systems import (
    LORENZ_BETA, LORENZ_SIGMA, double_pendulum, double_pendulum_energy, lorenz, lorenz_params,
    pendulum_params,
)


@lru_cache(maxsize=None)
def _lorenz_lambdas(start_index):
    p = lorenz_params()
    dt = cases.LORENZ_DT
    x0 = integrate_final(lorenz, cases.vec(cases.LORENZ_STARTS[start_index]), p, dt,
                         int(round(cases.LORENZ_SPINUP_T / dt)))
    n_blocks = int(round(cases.LORENZ_LAMBDA_T / (cases.STEPS_PER_BLOCK * dt)))
    exps, _, _ = lyapunov_spectrum(lorenz, x0, p, dt, n_blocks, cases.STEPS_PER_BLOCK, k=3,
                                   discard_blocks=cases.DISCARD_BLOCKS)
    two = lyapunov_two_trajectory(lorenz, x0, p, dt, n_blocks, cases.STEPS_PER_BLOCK,
                                  discard_blocks=cases.DISCARD_BLOCKS)
    return np.array(exps), float(two)


def test_g1_3_lorenz_lambda1_two_starts_agree_and_in_band():
    (a, _), (b, _) = _lorenz_lambdas(0), _lorenz_lambdas(1)
    print("lorenz lambda1 (start 0, start 1):", a[0], b[0])
    assert abs(a[0] - b[0]) < 0.02, (a[0], b[0])
    assert 0.85 <= a[0] <= 0.95 and 0.85 <= b[0] <= 0.95, (a[0], b[0])


def test_lorenz_two_trajectory_method_agrees_with_tangent_method():
    for i in (0, 1):
        exps, two = _lorenz_lambdas(i)
        assert abs(two - exps[0]) < 0.02, (two, exps[0])


def test_lorenz_spectrum_sum_equals_divergence():
    # Sum of exponents = trace of the Jacobian = -(sigma + 1 + beta). Independent of the 0.906 figure.
    exps, _ = _lorenz_lambdas(0)
    np.testing.assert_allclose(exps.sum(), -(LORENZ_SIGMA + 1.0 + LORENZ_BETA), atol=1e-3)
    assert abs(exps[1]) < 0.02  # the neutral direction along the flow


def test_g1_5_pendulum_high_energy_lambda_and_energy_drift():
    p = pendulum_params()
    x0 = jnp.array(cases.PENDULUM_HIGH)
    dt = cases.PENDULUM_DT
    n_blocks = int(round(cases.PENDULUM_LAMBDA_T / (cases.STEPS_PER_BLOCK * dt)))
    exps, _, _ = lyapunov_spectrum(double_pendulum, x0, p, dt, n_blocks, cases.STEPS_PER_BLOCK, k=1,
                                   discard_blocks=cases.DISCARD_BLOCKS)
    two = lyapunov_two_trajectory(double_pendulum, x0, p, dt, n_blocks, cases.STEPS_PER_BLOCK,
                                  discard_blocks=cases.DISCARD_BLOCKS)
    print("pendulum high-energy lambda (tangent, two-trajectory):", float(exps[0]), float(two))
    assert float(exps[0]) > 0.1, float(exps[0])
    assert float(two) > 0.1, float(two)

    xs = integrate(double_pendulum, x0, p, dt, int(round(cases.PENDULUM_ENERGY_T / dt)))
    e = double_pendulum_energy(xs, p)
    drift = float(jnp.max(jnp.abs(e - e[0])) / jnp.abs(e[0]))
    print("pendulum high-energy relative energy drift T=10, dt:", drift, dt)
    assert drift < 1e-6, drift
