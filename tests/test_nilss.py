"""A5 controls C-A5.1 to C-A5.4 for the NILSS written from the papers."""
import jax
import jax.numpy as jnp
import numpy as np

import chaos_jax  # noqa: F401
from chaos_jax.integrate import integrate_final
from chaos_jax.nilss import integrate_tangents, nilss, solve_nilss_problem
from chaos_jax.systems import lorenz, lorenz_params

P0 = lorenz_params()
DT = 0.005


def _z(x):
    return x[2]


def _on_attractor(x=(1.0, 2.0, 25.0)):
    return integrate_final(lorenz, jnp.array(x), P0, 0.01, 3000)


def test_c_a5_1_schur_solve_equals_dense_kkt():
    rng = np.random.default_rng(0)
    K, M, P = 5, 2, 3
    A = rng.normal(size=(K, M, M))
    Cs = A @ np.swapaxes(A, 1, 2) + np.eye(M)
    ds, Rs, bs = rng.normal(size=(K, M, P)), rng.normal(size=(K - 1, M, M)), rng.normal(size=(K - 1, M, P))
    a = np.asarray(solve_nilss_problem(*(jnp.asarray(x) for x in (Cs, ds, Rs, bs))))
    # dense KKT, one right-hand side at a time
    n = K * M
    Cb = np.zeros((n, n))
    B = np.zeros(((K - 1) * M, n))
    for k in range(K):
        Cb[k * M:(k + 1) * M, k * M:(k + 1) * M] = Cs[k]
    for i in range(1, K):
        B[(i - 1) * M:i * M, (i - 1) * M:i * M] = -Rs[i - 1]
        B[(i - 1) * M:i * M, i * M:(i + 1) * M] = np.eye(M)
    KKT = np.block([[Cb, B.T], [B, np.zeros((B.shape[0], B.shape[0]))]])
    for p in range(P):
        rhs = np.concatenate([-ds[:, :, p].ravel(), bs[:, :, p].ravel()])
        sol = np.linalg.solve(KKT, rhs)[:n].reshape(K, M)
        np.testing.assert_allclose(a[:, :, p], sol, rtol=1e-9, atol=1e-11)


def test_c_a5_2_tangents_match_finite_differences():
    u0 = _on_attractor()
    n_steps, h = 200, 1e-6
    W0 = jnp.eye(3)
    V0 = jnp.zeros((3, 3))
    u1, W1, V1 = integrate_tangents(lorenz, u0, W0, V0, P0, (0, 1, 2), DT, n_steps)
    run = lambda x, p: integrate_final(lorenz, x, p, DT, n_steps)
    for j in range(3):
        e = jnp.zeros(3).at[j].set(h)
        fd = (run(u0 + e, P0) - run(u0 - e, P0)) / (2 * h)
        np.testing.assert_allclose(np.asarray(W1[:, j]), np.asarray(fd), rtol=1e-5, atol=1e-7 * float(jnp.abs(fd).max()))
        fdp = (run(u0, P0.at[j].add(h)) - run(u0, P0.at[j].add(-h))) / (2 * h)
        np.testing.assert_allclose(np.asarray(V1[:, j]), np.asarray(fdp), rtol=1e-5, atol=1e-7 * float(jnp.abs(fdp).max()))


def _short(n_seg=4, steps=100, M=1):
    return nilss(lorenz, _z, _on_attractor(), P0, (0, 1, 2), DT, steps, n_seg, M, jax.random.PRNGKey(0))


def test_c_a5_3_v_perp_continuous_across_interfaces():
    res = _short(n_seg=6, steps=100, M=2)
    rec, a = res["rec"], res["a"]
    for i in range(5):
        left = rec["Vp_end"][i] + rec["Wp_end"][i] @ a[i]                          # v_perp_i(t_{i+1})
        right = rec["Vp_end"][i] - rec["Q"][i] @ rec["b"][i] + rec["Q"][i] @ a[i + 1]  # v_perp_{i+1}(t_{i+1})
        scale = float(jnp.linalg.norm(left))
        assert float(jnp.linalg.norm(left - right)) < 1e-8 * max(scale, 1.0), (i, scale)


def test_c_a5_4_formula_41_equals_formula_15_by_direct_quadrature():
    """Rebuild v(t) on every step (D.4), then evaluate (15) directly. Compare with (41)."""
    n_seg, steps, M = 4, 100, 1
    u0 = _on_attractor()
    res = nilss(lorenz, _z, u0, P0, (1,), DT, steps, n_seg, M, jax.random.PRNGKey(0))
    a, xi = np.asarray(res["a"])[:, :, 0], np.asarray(res["xi"])[:, 0]
    rec = res["rec"]
    u, W, V = u0, res["W0"], jnp.zeros((3, 1))
    f = lambda x: lorenz(x, P0)
    gradz = np.array([0.0, 0.0, 1.0])
    us, vs, Js = [u0], [np.asarray(res["W0"] @ jnp.asarray(a[0]))], [float(u0[2])]   # v(0) = v*(0) + W(0) a_0 = W(0) a_0
    xi_star = 0.0
    for i in range(n_seg):
        for _ in range(steps):
            u, W, V = integrate_tangents(lorenz, u, W, V, P0, (1,), DT, 1)
            us.append(u); Js.append(float(u[2]))
            vs.append(np.asarray(V[:, 0] + W @ jnp.asarray(a[i])) + xi_star * np.asarray(f(u)))
        # end of segment: reset as in eq. 31 and 34, accumulate xi (D.2)
        fu = f(u)
        xi_star += float(xi[i])
        Vp, Wp = V - jnp.outer(fu, fu @ V) / (fu @ fu), W - jnp.outer(fu, fu @ W) / (fu @ fu)
        Q = rec["Q"][i]
        W, V = Q, Vp - Q @ (Q.T @ Vp)
    Jarr, varr = np.array(Js), np.array(vs)
    T = n_seg * steps * DT
    Jbar = np.trapezoid(Jarr, dx=DT) / T
    integral = np.trapezoid(varr @ gradz, dx=DT)
    xiT = xi_star
    direct = (integral + xiT * Jbar - xiT * Jarr[-1]) / T       # eq. 15 with xi(0) = 0
    print("C-A5.4 relative difference (15) against (41):", abs(direct - float(res["dJds"][0])) / abs(float(res["dJds"][0])))
    np.testing.assert_allclose(direct, float(res["dJds"][0]), rtol=1e-3)


def test_lorenz_sensitivities_agree_with_the_independent_shadowing_values():
    """Regression against the values that nilss-jax and our LSS gave (A2, A3): rho 1.017, sigma 0.134, beta -1.659."""
    p = lorenz_params()
    k = jax.random.split(jax.random.PRNGKey(5), 3)
    starts = jnp.stack([jax.random.uniform(k[0], (6,), minval=-15.0, maxval=15.0), jax.random.uniform(k[1], (6,), minval=-15.0, maxval=15.0),
                        jax.random.uniform(k[2], (6,), minval=5.0, maxval=45.0)], axis=1)
    vals = []
    for i in range(6):
        u0 = integrate_final(lorenz, starts[i], p, DT, 4000)
        r = nilss(lorenz, _z, u0, p, (1, 0, 2), DT, 100, 500, 1, jax.random.PRNGKey(100 + i))   # T = 250
        vals.append(np.asarray(r["dJds"]))
    m = np.mean(vals, axis=0)
    print("T=250 mean (rho, sigma, beta):", m)
    assert abs(m[0] - 1.017) < 0.01 and abs(m[1] - 0.134) < 0.006 and abs(m[2] - (-1.659)) < 0.03
