import jax
import jax.numpy as jnp
import numpy as np

import chaos_jax  # noqa: F401
from chaos_jax.fixes import avg_obs, lss_derivative, param_gradient
from chaos_jax.integrate import integrate, integrate_final
from chaos_jax.systems import lorenz, lorenz_params

ORACLE = 1.0029
DT = 0.01
RHO_IDX = 1


def _z(x):
    return x[2]


def _attractor_points(n, key=7):
    k = jax.random.split(jax.random.PRNGKey(key), 3)
    s = jnp.stack([jax.random.uniform(k[0], (n,), minval=-15.0, maxval=15.0),
                   jax.random.uniform(k[1], (n,), minval=-15.0, maxval=15.0),
                   jax.random.uniform(k[2], (n,), minval=5.0, maxval=45.0)], axis=1)
    p = lorenz_params()
    return jax.vmap(lambda x: integrate_final(lorenz, x, p, DT, 5000))(s)


def test_truncated_with_long_segment_equals_naive():
    p, x0 = lorenz_params(), jnp.array([1.0, 2.0, 25.0])
    a = param_gradient(lorenz, _z, x0, p, RHO_IDX, DT, 300)
    b = param_gradient(lorenz, _z, x0, p, RHO_IDX, DT, 300, 10 ** 6)
    np.testing.assert_allclose(float(a), float(b), rtol=1e-13)


def test_truncated_equals_independent_segments():
    """Two segments: the second starts from a constant state, so rho only acts inside it."""
    p, x0 = lorenz_params(), jnp.array([1.0, 2.0, 25.0])
    n, half = 400, 200
    trunc = float(param_gradient(lorenz, _z, x0, p, RHO_IDX, DT, n, half))
    xm = integrate_final(lorenz, x0, p, DT, half)
    g1 = param_gradient(lorenz, _z, x0, p, RHO_IDX, DT, half)
    g2 = param_gradient(lorenz, _z, xm, p, RHO_IDX, DT, half)
    np.testing.assert_allclose(trunc, 0.5 * (float(g1) + float(g2)), rtol=1e-11)


def test_lss_on_lorenz_is_near_the_oracle_at_T40():
    p = lorenz_params()
    x0s = _attractor_points(8)
    ests = []
    for x0 in x0s:
        xs = integrate(lorenz, x0, p, DT, 4000)[1:]
        ests.append(float(lss_derivative(lorenz, _z, xs, p, RHO_IDX, DT, 10.0)[0]))
    print("lss T=40 estimates:", ests)
    assert abs(np.median(ests) - ORACLE) < 0.15 * ORACLE
    assert min(ests) > 0.5 and max(ests) < 1.5
