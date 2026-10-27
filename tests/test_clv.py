"""A5b controls C-B1 to C-B5 for the covariant Lyapunov vector code."""
import jax
import jax.numpy as jnp
import numpy as np

import chaos_jax  # noqa: F401
from chaos_jax.clv import covariant_vectors, line_angle
from chaos_jax.integrate import integrate_final
from chaos_jax.systems import lorenz, lorenz_params
from chaos_jax.tangent import step_and_tangent

P0 = lorenz_params()


def _lorenz_run(n_blocks=3000, spb=20, dt=0.005, seed=0, key_back=None):
    x0 = integrate_final(lorenz, jnp.array([1.0, 2.0, 25.0]), P0, dt, 4000)
    return covariant_vectors(lorenz, x0, P0, dt, n_blocks, spb, jax.random.PRNGKey(seed), key_back=key_back)


def test_c_b1_linear_flow_clvs_are_the_eigenvectors():
    A = jnp.array([[1.0, 2.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, -3.0]])
    lin = lambda x, p: A @ x
    # 200 time units: about 10 time units of forward history and of backward convergence are needed at these exponent gaps
    xs, V, growth, Rs = covariant_vectors(lin, jnp.array([1.0, 1.0, 1.0]), jnp.zeros(1), 0.002, 4000, 25, jax.random.PRNGKey(0))
    w, E = np.linalg.eig(np.asarray(A))
    order = np.argsort(-w.real)
    E = E[:, order].real
    for j in range(3):
        ang = line_angle(V[400:3600, :, j], jnp.broadcast_to(jnp.asarray(E[:, j]), (3200, 3)))
        assert float(ang.max()) < 1e-8, (j, float(ang.max()))


def test_c_b2_neutral_vector_is_the_flow():
    xs, V, growth, Rs = _lorenz_run()
    f = jax.vmap(lambda x: lorenz(x, P0))(xs)
    ang = line_angle(V[500:2500, :, 1], f[500:2500])
    assert float(ang.max()) < 1e-6, float(ang.max())


def test_c_b3_exponents_match_the_benettin_values():
    xs, V, growth, Rs = _lorenz_run(n_blocks=20000)
    lam = np.asarray(growth[500:19000].mean(axis=0)) / 0.1  # block length 0.1
    print("C-B3 CLV exponents:", lam)
    assert abs(lam[0] - 0.904) < 0.02 and abs(lam[1]) < 0.02 and abs(lam[2] + 14.57) < 0.1


def test_c_b4_backward_start_does_not_matter():
    _, Va, _, _ = _lorenz_run(key_back=jax.random.PRNGKey(11))
    _, Vb, _, _ = _lorenz_run(key_back=jax.random.PRNGKey(22))
    for j in range(3):
        ang = line_angle(Va[300:2500, :, j], Vb[300:2500, :, j])
        assert float(ang.max()) < 1e-8, (j, float(ang.max()))


def test_c_b5_clvs_are_covariant_under_the_true_tangent_flow():
    xs, V, growth, Rs = _lorenz_run()
    for n in (800, 1200, 1700):
        x, M = xs[n], V[n]
        for _ in range(20):
            x, M = step_and_tangent(lorenz, x, M, 0.005, P0)
        for j in range(3):
            assert float(line_angle(M[:, j], V[n + 1][:, j])) < 1e-8, (n, j)
