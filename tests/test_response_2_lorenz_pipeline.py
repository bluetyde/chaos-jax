import jax.numpy as jnp
import numpy as np

import chaos_jax  # noqa: F401
from chaos_jax.integrate import integrate
from chaos_jax.response import ensemble_time_average
from chaos_jax.systems import lorenz, lorenz_params


def test_time_average_equals_mean_of_trajectory_after_spinup():
    p = lorenz_params()
    x0s = jnp.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 20.0]])
    n_spin, n_avg, dt = 300, 500, 0.01
    got = ensemble_time_average(lorenz, lambda x: x[2], x0s, p, dt, n_spin, n_avg)
    for i in range(2):
        xs = integrate(lorenz, x0s[i], p, dt, n_spin + n_avg)
        np.testing.assert_allclose(got[i], xs[n_spin + 1:, 2].mean(), rtol=1e-12)
