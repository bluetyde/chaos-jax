"""LSS solver checks: block Thomas against dense, and the full KKT system against the Thomas pipeline."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chaos_jax  # noqa: F401
from chaos_jax.fixes import _block_thomas, lss_derivative
from chaos_jax.integrate import integrate
from chaos_jax.systems import lorenz, lorenz_params


def _zobs(x):
    return x[2]


def test_block_thomas_matches_dense_solve():
    rng = np.random.default_rng(0)
    k, n = 12, 3
    B = rng.normal(size=(k * n + 5, k * n))
    M = B.T @ B + np.eye(k * n)
    # keep only the block-tridiagonal part of a general SPD matrix by building from tridiagonal blocks instead
    D = np.stack([np.eye(n) * 6 + 0.1 * rng.normal(size=(n, n)) for _ in range(k)])
    D = (D + np.swapaxes(D, 1, 2)) / 2
    U = 0.5 * rng.normal(size=(k - 1, n, n))
    full = np.zeros((k * n, k * n))
    for i in range(k):
        full[i * n:(i + 1) * n, i * n:(i + 1) * n] = D[i]
    for i in range(k - 1):
        full[i * n:(i + 1) * n, (i + 1) * n:(i + 2) * n] = U[i]
        full[(i + 1) * n:(i + 2) * n, i * n:(i + 1) * n] = U[i].T
    r = rng.normal(size=(k, n))
    got = np.asarray(_block_thomas(jnp.asarray(D), jnp.asarray(U), jnp.asarray(r)))
    np.testing.assert_allclose(got.ravel(), np.linalg.solve(full, r.ravel()), rtol=1e-10)


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_lss_matches_dense_kkt_system(idx):
    """Build the constrained least-squares problem from its definition and solve the KKT system densely."""
    p = lorenz_params()
    dt, m, alpha = 0.01, 40, 10.0
    xs = np.asarray(integrate(lorenz, jnp.array([1.0, 2.0, 25.0]), p, dt, m)[1:])
    n = 3
    A = np.asarray(jax.vmap(lambda x: jax.jacfwd(lorenz)(x, p))(jnp.asarray(xs)))
    S = np.zeros((m, n))  # analytic d f / d parameter: sigma (index 0) (y - x, 0, 0), rho (1) (0, x, 0), beta (2) (0, 0, -z)
    if idx == 0:
        S[:, 0] = xs[:, 1] - xs[:, 0]
    elif idx == 1:
        S[:, 1] = xs[:, 0]
    else:
        S[:, 2] = -xs[:, 2]
    nv, ne = m * n, m - 1
    C = np.zeros((m - 1) * n, dtype=float).reshape(-1, 1) @ np.zeros((1, nv + ne))
    rhs = np.zeros((m - 1) * n)
    for i in range(1, m):
        E = -np.eye(n) / dt - 0.5 * A[i - 1]
        G = np.eye(n) / dt - 0.5 * A[i]
        f_i = (xs[i] - xs[i - 1]) / dt
        rows = slice((i - 1) * n, i * n)
        C[rows, (i - 1) * n:i * n] = E
        C[rows, i * n:(i + 1) * n] = G
        C[rows, nv + i - 1] = -f_i
        rhs[rows] = 0.5 * (S[i - 1] + S[i])
    H = np.diag(np.concatenate([np.ones(nv), alpha ** 2 * np.ones(ne)]))
    K = np.block([[H, C.T], [C, np.zeros((C.shape[0], C.shape[0]))]])
    sol = np.linalg.solve(K, np.concatenate([np.zeros(nv + ne), rhs]))
    v_ref, eta_ref = sol[:nv].reshape(m, n), sol[nv:nv + ne]
    deriv, v, eta = lss_derivative(lorenz, _zobs, jnp.asarray(xs), p, idx, dt, alpha)
    np.testing.assert_allclose(np.asarray(v), v_ref, rtol=1e-7, atol=1e-9 * np.abs(v_ref).max())
    np.testing.assert_allclose(np.asarray(eta), eta_ref, rtol=1e-7, atol=1e-9 * np.abs(eta_ref).max())
    # Eq. 22 and 23 assembled independently in numpy. The observable is z, so dJ/du = (0, 0, 1).
    J = xs[:, 2]
    Jt = 0.5 * (J[:-1] + J[1:]) - J.mean()
    deriv_ref = v_ref[:, 2].sum() / m + (eta_ref * Jt).sum() / (m - 1)
    np.testing.assert_allclose(float(deriv), deriv_ref, rtol=1e-8)
    assert abs((eta_ref * Jt).sum() / (m - 1)) > 1e-3 * abs(deriv_ref)  # the eta term is not negligible, so a shift would show
