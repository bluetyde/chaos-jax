"""Gradient estimators for the sensitivity of a time average to a parameter.

naive / truncated: reverse-mode autodiff through the unrolled RK4 scan.
lss: least squares shadowing, discretised from Eq. 13 and 15 of arXiv 1204.0159 (see docs/PREDICTIONS-4.md
for where this differs from the printed Eq. 16 to 21), block-tridiagonal solve.
"""
from functools import partial

import jax
import jax.numpy as jnp

from chaos_jax.integrate import rk4_step


def avg_obs(f, obs, x0, p, dt, n_steps, seg_steps=None, n_skip=0):
    """Mean of obs over the post-step states after the first n_skip of n_steps.

    With seg_steps, stop the gradient at the start of every segment of that many steps.
    """
    def body(carry, k):
        x, total = carry
        if seg_steps is not None:
            x = jnp.where(k % seg_steps == 0, jax.lax.stop_gradient(x), x)
        x = rk4_step(f, x, dt, p)
        return (x, total + jnp.where(k >= n_skip, obs(x), 0.0)), None

    (_, total), _ = jax.lax.scan(body, (x0, jnp.zeros((), x0.dtype)), jnp.arange(n_steps))
    return total / (n_steps - n_skip)


@partial(jax.jit, static_argnames=("f", "obs", "idx", "n_steps", "seg_steps"))
def param_gradient(f, obs, x0, p, idx, dt, n_steps, seg_steps=None):
    """d/dp[idx] of avg_obs for one start point, reverse mode."""
    return jax.grad(lambda r: avg_obs(f, obs, x0, p.at[idx].set(r), dt, n_steps, seg_steps))(p[idx])


def batched_param_gradient(f, obs, x0s, p, idx, dt, n_steps, seg_steps=None, chunk=250):
    """param_gradient for every row of x0s, in chunks to limit memory."""
    g = jax.jit(jax.vmap(lambda x0: param_gradient(f, obs, x0, p, idx, dt, n_steps, seg_steps)))
    out = [g(x0s[i:i + chunk]) for i in range(0, x0s.shape[0], chunk)]
    return jnp.concatenate(out)


def _block_thomas(D, U, r):
    """Solve a symmetric block-tridiagonal system. D: (k,n,n) diagonal blocks, U: (k-1,n,n) upper blocks."""
    def fwd(carry, inp):
        C_prev, y_prev = carry
        Dj, Ujm1, rj = inp
        X = jnp.linalg.solve(C_prev, jnp.concatenate([Ujm1, y_prev[:, None]], axis=1))
        C = Dj - Ujm1.T @ X[:, :-1]
        y = rj - Ujm1.T @ X[:, -1]
        return (C, y), (C, y)

    _, (Cs, ys) = jax.lax.scan(fwd, (D[0], r[0]), (D[1:], U, r[1:]))
    Cs = jnp.concatenate([D[:1], Cs])
    ys = jnp.concatenate([r[:1], ys])
    w_last = jnp.linalg.solve(Cs[-1], ys[-1])

    def bwd(w_next, inp):
        Cj, yj, Uj = inp
        w = jnp.linalg.solve(Cj, yj - Uj @ w_next)
        return w, w

    _, ws = jax.lax.scan(bwd, w_last, (Cs[:-1], ys[:-1], U), reverse=True)
    return jnp.concatenate([ws, w_last[None]])


@partial(jax.jit, static_argnames=("f", "obs", "idx"))
def lss_derivative(f, obs, xs, p, idx, dt, alpha):
    """LSS estimate of d<obs>/dp[idx] from trajectory samples xs, shape (m, n), spaced dt apart.

    Returns (derivative, v, eta). Uses m - 1 constraints:
    E_i v_{i-1} + G_i v_i - f_i eta_i = b_i, i = 1..m-1,
    minimising sum |v|^2 / 2 + alpha^2 sum eta^2 / 2.
    """
    m, n = xs.shape
    I = jnp.eye(n, dtype=xs.dtype)
    A = jax.vmap(lambda x: jax.jacfwd(f, argnums=0)(x, p))(xs)
    S = jax.vmap(lambda x: jax.jacfwd(lambda q: f(x, p.at[idx].set(q)))(p[idx]))(xs)
    E = -I / dt - 0.5 * A[:-1]
    G = I / dt - 0.5 * A[1:]
    fi = (xs[1:] - xs[:-1]) / dt
    b = 0.5 * (S[:-1] + S[1:])

    D = E @ jnp.swapaxes(E, 1, 2) + G @ jnp.swapaxes(G, 1, 2) + fi[:, :, None] * fi[:, None, :] / alpha ** 2
    U = G[:-1] @ jnp.swapaxes(E[1:], 1, 2)
    w = _block_thomas(D, U, -b)

    Gt_w = jnp.einsum("jab,ja->jb", G, w)   # G_i^T w_i, enters v_i (sample i)
    Et_w = jnp.einsum("jab,ja->jb", E, w)   # E_i^T w_i, enters v_{i-1}
    zero = jnp.zeros((1, n), dtype=xs.dtype)
    v = -(jnp.concatenate([zero, Gt_w]) + jnp.concatenate([Et_w, zero]))
    eta = jnp.einsum("ja,ja->j", fi, w) / alpha ** 2

    J = jax.vmap(obs)(xs)
    dJ = jax.vmap(jax.grad(obs))(xs)
    Jt = 0.5 * (J[:-1] + J[1:]) - J.mean()
    deriv = jnp.sum(dJ * v) / m + jnp.sum(eta * Jt) / (m - 1)
    return deriv, v, eta
