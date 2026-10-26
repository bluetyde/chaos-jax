"""Ensemble finite-difference estimate of the sensitivity of a long-time average to a parameter."""
from functools import partial

import jax
import jax.numpy as jnp

from chaos_jax.integrate import integrate_final, rk4_step


@partial(jax.jit, static_argnames=("f", "obs", "n_spin", "n_avg"))
def time_average(f, obs, x0, p, dt, n_spin, n_avg):
    """Run n_spin steps to leave the transient, then average obs(x) over the next n_avg steps."""
    x = integrate_final(f, x0, p, dt, n_spin)

    def body(carry, _):
        x, total = carry
        x = rk4_step(f, x, dt, p)
        return (x, total + obs(x)), None

    (_, total), _ = jax.lax.scan(body, (x, jnp.zeros(())), None, length=n_avg)
    return total / n_avg


def ensemble_time_average(f, obs, x0s, p, dt, n_spin, n_avg):
    """time_average for every start point in x0s (shape (N, n)). Returns shape (N,)."""
    return jax.vmap(lambda x0: time_average(f, obs, x0, p, dt, n_spin, n_avg))(x0s)


def bootstrap_se(stat, data, key, n_boot=2000):
    """Bootstrap standard error of stat(mean over rows of data).

    data has shape (N, m). Each resample draws N rows with replacement, so columns stay paired.
    stat maps a length-m vector of means to a scalar.
    """
    n = data.shape[0]
    idx = jax.random.randint(key, (n_boot, n), 0, n)
    means = data[idx].mean(axis=1)
    values = jax.vmap(stat)(means)
    return jnp.std(values, ddof=1)


def central_slope(means, h):
    """means = [value at -h, value at +h]; returns the central difference."""
    return (means[1] - means[0]) / (2.0 * h)
