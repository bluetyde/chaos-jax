"""Lyapunov exponent estimators.

lyapunov_spectrum: Benettin method with tangent vectors and QR re-orthonormalisation.
lyapunov_two_trajectory: Benettin's original method with two nearby trajectories.
The two share no code beyond the integrator, so they are independent checks.
"""
from functools import partial

import jax
import jax.numpy as jnp

from chaos_jax.integrate import rk4_step
from chaos_jax.tangent import step_and_tangent


@partial(jax.jit, static_argnames=("f", "n_blocks", "steps_per_block", "k"))
def _spectrum_scan(f, x0, Q0, p, dt, n_blocks, steps_per_block, k):
    def block(carry, _):
        x, Q = carry

        def inner(c, _):
            xx, QQ = c
            return step_and_tangent(f, xx, QQ, dt, p), None

        (x, Q), _ = jax.lax.scan(inner, (x, Q), None, length=steps_per_block)
        Q, R = jnp.linalg.qr(Q)
        return (x, Q), jnp.log(jnp.abs(jnp.diag(R)))

    (x, _), logs = jax.lax.scan(block, (x0, Q0), None, length=n_blocks)
    return x, logs


def lyapunov_spectrum(f, x0, p, dt, n_blocks, steps_per_block=10, k=1, discard_blocks=0, key=None):
    """Top-k Lyapunov exponents. Returns (exponents, per-block log growth, final state).

    x0 should already lie on the attractor. The first discard_blocks blocks are
    left out of the average so the tangent vectors can align first.
    """
    n = x0.shape[0]
    key = jax.random.PRNGKey(0) if key is None else key
    Q0, _ = jnp.linalg.qr(jax.random.normal(key, (n, k)))
    x_final, logs = _spectrum_scan(f, x0, Q0, p, dt, n_blocks, steps_per_block, k)
    kept = logs[discard_blocks:]
    exps = kept.sum(axis=0) / (kept.shape[0] * steps_per_block * dt)
    return exps, logs, x_final


@partial(jax.jit, static_argnames=("f", "n_blocks", "steps_per_block"))
def _two_traj_scan(f, x0, y0, p, dt, d0, n_blocks, steps_per_block):
    def block(carry, _):
        x, y = carry

        def inner(c, _):
            xx, yy = c
            return (rk4_step(f, xx, dt, p), rk4_step(f, yy, dt, p)), None

        (x, y), _ = jax.lax.scan(inner, (x, y), None, length=steps_per_block)
        sep = y - x
        d = jnp.linalg.norm(sep)
        return (x, x + sep * (d0 / d)), jnp.log(d / d0)

    _, logs = jax.lax.scan(block, (x0, y0), None, length=n_blocks)
    return logs


def lyapunov_two_trajectory(f, x0, p, dt, n_blocks, steps_per_block=10, d0=1e-8, discard_blocks=0):
    """Largest exponent from the growth of a separation d0 between two trajectories."""
    direction = jnp.ones_like(x0) / jnp.sqrt(x0.shape[0])
    logs = _two_traj_scan(f, x0, x0 + d0 * direction, p, dt, d0, n_blocks, steps_per_block)
    kept = logs[discard_blocks:]
    return kept.sum() / (kept.shape[0] * steps_per_block * dt)
