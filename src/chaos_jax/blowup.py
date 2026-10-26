"""Measurements for Phase 3: growth of the state Jacobian, and the naive gradient of a time average."""
from functools import partial

import jax
import jax.numpy as jnp

from chaos_jax.integrate import rk4_step
from chaos_jax.systems import lorenz, lorenz_params
from chaos_jax.tangent import step_and_tangent


@partial(jax.jit, static_argnames=("f", "n_blocks", "steps_per_block"))
def log_jacobian_norms(f, x0, p, dt, n_blocks, steps_per_block):
    """log of the largest singular value of dx(t)/dx0, at t = k * steps_per_block * dt, k = 1..n_blocks."""
    n = x0.shape[0]

    def block(carry, _):
        def inner(c, _):
            return step_and_tangent(f, c[0], c[1], dt, p), None

        carry, _ = jax.lax.scan(inner, carry, None, length=steps_per_block)
        return carry, jnp.log(jnp.linalg.norm(carry[1], 2))

    _, logs = jax.lax.scan(block, (x0, jnp.eye(n, dtype=x0.dtype)), None, length=n_blocks)
    return logs


def lorenz_time_average_z(rho, x0, dt, n_steps):
    """(1/n) sum of z over the n post-step states of the RK4 trajectory, at parameter rho."""
    p = lorenz_params(rho=rho)

    def body(carry, _):
        x, total = carry
        x = rk4_step(lorenz, x, dt, p)
        return (x, total + x[2]), None

    (_, total), _ = jax.lax.scan(body, (x0, jnp.zeros((), x0.dtype)), None, length=n_steps)
    return total / n_steps


@partial(jax.jit, static_argnames=("n_steps",))
def naive_gradients(x0s, rho, dt, n_steps):
    """Reverse-mode d/drho of the time average of z, for each start point in x0s."""
    g = jax.grad(lambda r, x0: lorenz_time_average_z(r, x0, dt, n_steps))
    return jax.vmap(g, in_axes=(None, 0))(rho, x0s)
