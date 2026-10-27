"""A4 control C-A4.2: fixed points of the Lorenz-96 and Roessler vector fields."""
import jax.numpy as jnp
import numpy as np

import chaos_jax  # noqa: F401
from chaos_jax.systems import lorenz96, lorenz96_params, rossler, rossler_params


def test_lorenz96_fixed_point_is_F_everywhere():
    for n in (8, 16):
        for F in (6.0, 8.0, 10.0):
            np.testing.assert_allclose(lorenz96(jnp.full(n, F), lorenz96_params(F)), 0.0, atol=1e-13)


def test_lorenz96_known_value():
    # u = (1, 2, 3, 4): du_0 = (u_1 - u_2) u_3 - u_0 + F = (2 - 3) 4 - 1 + F with periodic indices (u_{-1} = u_3, u_{-2} = u_2).
    out = lorenz96(jnp.array([1.0, 2.0, 3.0, 4.0]), lorenz96_params(8.0))
    np.testing.assert_allclose(float(out[0]), (2.0 - 3.0) * 4.0 - 1.0 + 8.0, rtol=1e-14)


def test_rossler_fixed_point():
    for c in (5.0, 7.0, 9.0):
        a, b = 0.2, 0.2
        x = (c - np.sqrt(c ** 2 - 4 * a * b)) / 2
        fp = jnp.array([x, -x / a, x / a])
        np.testing.assert_allclose(rossler(fp, rossler_params(a, b, c)), 0.0, atol=1e-12)
