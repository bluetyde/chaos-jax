"""A3b controls: the noisy integrator against an Ornstein-Uhlenbeck process with known statistics."""
import jax
import jax.numpy as jnp
import numpy as np

import chaos_jax  # noqa: F401
from chaos_jax.noisy import ensemble_noisy_time_average, noisy_time_average
from chaos_jax.response import time_average
from chaos_jax.systems import lorenz, lorenz_params

DT = 0.01


def ou(x, p):
    return -p[0] * x


def _x2(x):
    return x[0] ** 2


def _keys(key, n):
    return jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(n))


def test_eps_zero_is_the_deterministic_average():
    p, x0 = lorenz_params(), jnp.array([1.0, 2.0, 25.0])
    a = noisy_time_average(lorenz, lambda x: x[2], x0, jax.random.PRNGKey(0), p, DT, 300, 500, 0.0)
    b = time_average(lorenz, lambda x: x[2], x0, p, DT, 300, 500)
    np.testing.assert_allclose(float(a), float(b), rtol=1e-13)


def test_same_key_gives_same_noise_and_other_key_does_not():
    p, x0 = lorenz_params(), jnp.array([1.0, 2.0, 25.0])
    f = lambda key: float(noisy_time_average(lorenz, lambda x: x[2], x0, key, p, DT, 100, 100, 1.0))
    assert f(jax.random.PRNGKey(1)) == f(jax.random.PRNGKey(1))
    assert f(jax.random.PRNGKey(1)) != f(jax.random.PRNGKey(2))


def test_c_n1_ou_slope_in_k_within_3_percent():
    """d<x^2>/dk = -eps^2 / (2 k^2) = -0.5 for k = 1, eps = 1."""
    eps, h, n = 1.0, 0.1, 4000
    x0s = jax.random.uniform(jax.random.PRNGKey(3), (n, 1), minval=-3.0, maxval=3.0)
    keys = _keys(jax.random.PRNGKey(4), n)
    m = [float(jnp.mean(ensemble_noisy_time_average(ou, _x2, x0s, keys, jnp.array([1.0 + s * h]), DT, 5000, 10000, eps)))
         for s in (-1.0, 1.0)]
    slope = (m[1] - m[0]) / (2 * h)
    rel = abs(slope - (-0.5)) / 0.5
    print("C-N1 OU slope", slope, "relative error", rel)
    assert rel < 0.03
