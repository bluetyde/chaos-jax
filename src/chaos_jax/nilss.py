"""NILSS (non-intrusive least squares shadowing), written from arXiv 1611.00880 and 1711.06633.

Equation numbers below are those of arXiv 1611.00880 unless noted. See docs/PREDICTIONS-A5.md.
"""
from functools import partial

import jax
import jax.numpy as jnp

from chaos_jax.fixes import _block_thomas


def _rk4_tree(rhs, y, dt):
    k1 = rhs(y)
    k2 = rhs(jax.tree_util.tree_map(lambda a, b: a + 0.5 * dt * b, y, k1))
    k3 = rhs(jax.tree_util.tree_map(lambda a, b: a + 0.5 * dt * b, y, k2))
    k4 = rhs(jax.tree_util.tree_map(lambda a, b: a + dt * b, y, k3))
    return jax.tree_util.tree_map(lambda a, b1, b2, b3, b4: a + dt / 6.0 * (b1 + 2.0 * b2 + 2.0 * b3 + b4), y, k1, k2, k3, k4)


def _make_fields(f, p, idxs):
    ix = jnp.array(idxs)

    def fp(u):  # df/ds for each differentiated parameter, shape (n, P)
        return jax.jacfwd(lambda q: f(u, p.at[ix].set(q)))(p[ix])

    def jac_times(u, X):  # (df/du) X for the columns of X
        return jax.vmap(lambda x: jax.jvp(lambda uu: f(uu, p), (u,), (x,))[1], in_axes=1, out_axes=1)(X)

    def rhs(y):
        u, W, V = y
        return f(u, p), jac_times(u, W), jac_times(u, V) + fp(u)

    return rhs


def _project(fu, X):
    """Orthogonal projection of the columns of X onto the plane perpendicular to fu (eq. 8)."""
    return X - jnp.outer(fu, fu @ X) / (fu @ fu)


@partial(jax.jit, static_argnames=("f", "idxs", "n_steps"))
def integrate_tangents(f, u0, W0, V0, p, idxs, dt, n_steps):
    """RK4 integration of (u, W, V): du = f, dW = (df/du) W, dV = (df/du) V + df/ds. Returns the end state."""
    rhs = _make_fields(f, p, idxs)
    y, _ = jax.lax.scan(lambda y, _: (_rk4_tree(rhs, y, dt), None), (u0, W0, V0), None, length=n_steps)
    return y


def solve_nilss_problem(Cs, ds, Rs, bs):
    """Minimise sum_i 2 d_i^T a_i + a_i^T C_i a_i subject to a_i = R_i a_{i-1} + b_i (eq. 37).

    Cs (K, M, M), ds (K, M, P), Rs (K - 1, M, M) with Rs[j] the R_i of the constraint i = j + 1, bs (K - 1, M, P).
    Schur complement of the KKT system (arXiv 1711.06633, eq. 47 to 51), block-tridiagonal solve. Returns a (K, M, P).
    """
    K, M, P = ds.shape
    Cinv = jnp.linalg.inv(Cs)
    if K == 1:
        return -jnp.einsum("kab,kbp->kap", Cinv, ds)
    # S = B C^-1 B^T on the constraint rows i = 1 .. K - 1 (index j = i - 1).
    D = jnp.einsum("jab,jbc,jdc->jad", Rs, Cinv[:-1], Rs) + Cinv[1:]
    U = -jnp.einsum("jab,jcb->jac", Cinv[1:-1], Rs[1:])  # S_{i,i+1} = -C_i^-1 R_{i+1}^T
    # right-hand side B C^-1 d + b, row i: C_i^-1 d_i - R_i C_{i-1}^-1 d_{i-1} + b_i
    Cd = jnp.einsum("kab,kbp->kap", Cinv, ds)
    rvec = Cd[1:] - jnp.einsum("jab,jbp->jap", Rs, Cd[:-1]) + bs
    lam = jax.vmap(lambda r: _block_thomas(D, U, r), in_axes=2, out_axes=2)(-rvec)  # S lam = -(B C^-1 d + b)
    # a = -C^-1 (B^T lam + d); (B^T lam)_k = lam_k - R_{k+1}^T lam_{k+1}
    zero = jnp.zeros((1, M, P), dtype=ds.dtype)
    lam_k = jnp.concatenate([zero, lam], axis=0)                      # lam at k = 0 .. K - 1, lam_0 = 0
    RTlam = jnp.einsum("jba,jbp->jap", Rs, lam)                       # R_{k+1}^T lam_{k+1} for k = 0 .. K - 2
    Btlam = lam_k - jnp.concatenate([RTlam, zero], axis=0)
    return -jnp.einsum("kab,kbp->kap", Cinv, Btlam + ds)


@partial(jax.jit, static_argnames=("f", "obs", "idxs", "steps_per_seg", "n_seg", "M"))
def nilss(f, obs, u0, p, idxs, dt, steps_per_seg, n_seg, M, key):
    """NILSS sensitivity of the long-time average of obs to the parameters p[idxs], from one orbit.

    u0 must lie on the attractor. Returns a dict with the sensitivities `dJds` (P,), the exponents `lyap` (M,),
    the mean `J`, and the arrays needed to check the construction.
    """
    n = u0.shape[0]
    idx_arr = jnp.array(idxs)
    P = len(idxs)
    rhs = _make_fields(f, p, idxs)
    grad_obs = jax.grad(obs)

    def integrands(y):
        u, W, V = y
        fu = f(u, p)
        Wp, Vp = _project(fu, W), _project(fu, V)
        gJ = grad_obs(u)
        return Wp.T @ Wp, Wp.T @ Vp, gJ @ W, gJ @ V, obs(u)

    def segment(carry, _):
        u, W, V = carry
        g0 = integrands((u, W, V))

        def step(c, _):
            y, g_old, acc = c
            y = _rk4_tree(rhs, y, dt)
            g_new = integrands(y)
            acc = jax.tree_util.tree_map(lambda a, go, gn: a + 0.5 * dt * (go + gn), acc, g_old, g_new)
            return (y, g_new, acc), None

        zero_acc = jax.tree_util.tree_map(jnp.zeros_like, g0)
        (y, _, acc), _ = jax.lax.scan(step, ((u, W, V), g0, zero_acc), None, length=steps_per_seg)
        u1, W1, V1 = y
        C, d, JW, JV, Jint = acc
        fu = f(u1, p)
        ff = fu @ fu
        Wp1, Vp1 = _project(fu, W1), _project(fu, V1)
        Q, R = jnp.linalg.qr(Wp1)                       # eq. 30
        b = Q.T @ Vp1                                   # eq. 35
        out = {"C": C, "d": d, "R": R, "b": b, "JW": JW, "JV": JV, "Jint": Jint, "J_end": obs(u1),
               "xi_star": (fu @ V1) / ff, "xi_W": (fu @ W1) / ff, "log_diag_R": jnp.log(jnp.abs(jnp.diag(R))),
               "Vp_end": Vp1, "Wp_end": Wp1, "Q": Q}
        return (u1, Q, Vp1 - Q @ b), out               # eq. 31 and 34

    fu0 = f(u0, p)
    W0, _ = jnp.linalg.qr(_project(fu0, jax.random.normal(key, (n, M), dtype=u0.dtype)))
    V0 = jnp.zeros((n, P), dtype=u0.dtype)
    _, rec = jax.lax.scan(segment, (u0, W0, V0), None, length=n_seg)

    a = solve_nilss_problem(rec["C"], rec["d"], rec["R"][:-1], rec["b"][:-1])             # eq. 37
    T = n_seg * steps_per_seg * dt
    Jbar = rec["Jint"].sum() / T
    xi = rec["xi_star"] + jnp.einsum("km,kmp->kp", rec["xi_W"], a)                         # eq. 40
    seg_term = rec["JV"] + jnp.einsum("km,kmp->kp", rec["JW"], a)
    dJds = (seg_term + xi * (Jbar - rec["J_end"])[:, None]).sum(axis=0) / T                # eq. 41
    return {"dJds": dJds, "lyap": rec["log_diag_R"].sum(axis=0) / T, "J": Jbar, "a": a, "xi": xi, "T": T, "rec": rec,
            "W0": W0}
