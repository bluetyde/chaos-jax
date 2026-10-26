"""Fixed-step RK4 integrator built on jax.lax.scan."""
from functools import partial

import jax
import jax.numpy as jnp


def rk4_step(f, x, dt, p):
    k1 = f(x, p)
    k2 = f(x + 0.5 * dt * k1, p)
    k3 = f(x + 0.5 * dt * k2, p)
    k4 = f(x + dt * k3, p)
    return x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@partial(jax.jit, static_argnames=("f", "n_steps"))
def integrate(f, x0, p, dt, n_steps):
    """Return the trajectory, shape (n_steps + 1, n), including x0."""
    def body(x, _):
        x_new = rk4_step(f, x, dt, p)
        return x_new, x_new

    _, xs = jax.lax.scan(body, x0, None, length=n_steps)
    return jnp.concatenate([x0[None], xs], axis=0)


@partial(jax.jit, static_argnames=("f", "n_steps"))
def integrate_final(f, x0, p, dt, n_steps):
    """Return only the state at step n_steps. Stores no trajectory."""
    def body(x, _):
        return rk4_step(f, x, dt, p), None

    x, _ = jax.lax.scan(body, x0, None, length=n_steps)
    return x
