"""Controls: run before the chaotic cases. Part of G1.4."""
import jax.numpy as jnp
import numpy as np

import chaos_jax  # noqa: F401
from chaos_jax import cases
from chaos_jax.lyapunov import lyapunov_spectrum, lyapunov_two_trajectory
from chaos_jax.systems import damped_oscillator, damped_oscillator_params, double_pendulum, pendulum_params


def test_g1_4_damped_oscillator_lambda_is_negative():
    c = 0.4
    p = damped_oscillator_params(c, 4.0)
    x0 = jnp.array(cases.OSCILLATOR_START)
    exps, _, _ = lyapunov_spectrum(damped_oscillator, x0, p, 0.01, n_blocks=2000, k=2)
    assert float(exps[0]) < 0.0
    # Both exponents of the linear system equal -c/2 (complex pair, underdamped).
    np.testing.assert_allclose(exps, [-c / 2, -c / 2], atol=2e-3)


def test_g1_4_low_energy_pendulum_lambda_is_near_zero():
    p = pendulum_params()
    x0 = jnp.array(cases.PENDULUM_LOW)
    dt = cases.PENDULUM_DT
    n_blocks = int(round(cases.PENDULUM_LAMBDA_T / (cases.STEPS_PER_BLOCK * dt)))
    exps, _, _ = lyapunov_spectrum(double_pendulum, x0, p, dt, n_blocks, cases.STEPS_PER_BLOCK, k=1,
                                   discard_blocks=cases.DISCARD_BLOCKS)
    two = lyapunov_two_trajectory(double_pendulum, x0, p, dt, n_blocks, cases.STEPS_PER_BLOCK,
                                  discard_blocks=cases.DISCARD_BLOCKS)
    assert abs(float(exps[0])) < 0.02, float(exps[0])
    assert abs(float(two)) < 0.02, float(two)
