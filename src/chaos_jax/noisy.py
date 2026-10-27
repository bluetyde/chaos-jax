"""Time averages of a flow with additive white noise: RK4 drift step, then eps * sqrt(dt) * normal noise."""
from functools import partial

import jax
import jax.numpy as jnp

from chaos_jax.integrate import rk4_step


@partial(jax.jit, static_argnames=("f", "obs", "n_spin", "n_avg"))
def noisy_time_average(f, obs, x0, key, p, dt, n_spin, n_avg, eps):
    """Spin up n_spin steps, then average obs over n_avg steps. The noise at step k depends on (key, k) only."""
    n = x0.shape[0]
    amp = eps * jnp.sqrt(dt)

    def body(carry, k):
        x, total = carry
        x = rk4_step(f, x, dt, p) + amp * jax.random.normal(jax.random.fold_in(key, k), (n,), dtype=x.dtype)
        return (x, total + jnp.where(k >= n_spin, obs(x), 0.0)), None

    (_, total), _ = jax.lax.scan(body, (x0, jnp.zeros((), x0.dtype)), jnp.arange(n_spin + n_avg))
    return total / n_avg


def ensemble_noisy_time_average(f, obs, x0s, keys, p, dt, n_spin, n_avg, eps):
    """noisy_time_average for every start point (rows of x0s) with its own key (rows of keys)."""
    return jax.vmap(lambda x0, k: noisy_time_average(f, obs, x0, k, p, dt, n_spin, n_avg, eps))(x0s, keys)
