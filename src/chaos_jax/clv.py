"""Covariant Lyapunov vectors by the dynamical algorithm of Ginelli et al. (arXiv 1212.3961, Sec. 4).

Forward: evolve an orthonormal tangent basis G with QR every block, M_n G_n = G_{n+1} R_{n+1} (eq. 23).
Backward: C_n = R_{n+1}^-1 C_{n+1} with unit columns (eq. 30 and 32). The CLVs are V_n = G_n C_n (eq. 27).
"""
from functools import partial

import jax
import jax.numpy as jnp

from chaos_jax.tangent import step_and_tangent


@partial(jax.jit, static_argnames=("f", "n_blocks", "steps_per_block"))
def forward_gram_schmidt(f, x0, G0, p, dt, n_blocks, steps_per_block):
    """Per block: the state and basis at its start, and the R of the QR at its end. QR gauge: positive diagonal."""
    def block(carry, _):
        x, G = carry

        def inner(c, _):
            return step_and_tangent(f, c[0], c[1], dt, p), None

        (x1, V), _ = jax.lax.scan(inner, (x, G), None, length=steps_per_block)
        Q, R = jnp.linalg.qr(V)
        s = jnp.sign(jnp.diag(R))
        s = jnp.where(s == 0, 1.0, s)
        Q, R = Q * s[None, :], R * s[:, None]
        return (x1, Q), (x, G, R)

    (xe, Ge), (xs, Gs, Rs) = jax.lax.scan(block, (x0, G0), None, length=n_blocks)
    return xs, Gs, Rs


@jax.jit
def backward_coefficients(Rs, C_end):
    """C_n for n = N-1 .. 0, from the R of each block (Rs[n] = R_{n+1}) and a starting upper-triangular C_N."""
    def step(C, R):
        C = jax.scipy.linalg.solve_triangular(R, C, lower=False)
        C = C / jnp.linalg.norm(C, axis=0, keepdims=True)
        return C, C

    _, Cs = jax.lax.scan(step, C_end, Rs, reverse=True)
    return Cs


def covariant_vectors(f, x0, p, dt, n_blocks, steps_per_block, key, n_forward_transient=0, key_back=None):
    """Return (states, CLVs V of shape (N, n, n), log growth per block of each CLV, R) for the N = n_blocks samples.

    The first n_forward_transient samples and the last backward-transient samples are not valid. The caller discards them.
    """
    n = x0.shape[0]
    G0, _ = jnp.linalg.qr(jax.random.normal(key, (n, n)))
    xs, Gs, Rs = forward_gram_schmidt(f, x0, G0, p, dt, n_blocks, steps_per_block)
    kb = jax.random.fold_in(key, 1) if key_back is None else key_back
    C_end = jnp.triu(jax.random.normal(kb, (n, n)))
    C_end = C_end / jnp.linalg.norm(C_end, axis=0, keepdims=True)
    Cs = backward_coefficients(Rs, C_end)
    V = jnp.einsum("nab,nbc->nac", Gs, Cs)
    growth = jnp.log(jnp.linalg.norm(jnp.einsum("nab,nbc->nac", Rs, Cs), axis=1))  # |R_{n+1} c_n| per CLV
    return xs, V, growth, Rs


def line_angle(a, b):
    """Angle in [0, pi/2] between the lines spanned by a and b (vectors along the last axis).

    atan2 of the part of a perpendicular to b and the part along b. Accurate for tiny angles,
    where arccos of a cosine near 1 has a floor of about 1.5e-8.
    """
    bh = b / jnp.linalg.norm(b, axis=-1, keepdims=True)
    par = jnp.sum(a * bh, axis=-1)
    perp = jnp.linalg.norm(a - par[..., None] * bh, axis=-1)
    return jnp.arctan2(perp, jnp.abs(par))


def vector_plane_angle(w, u1, u2):
    """Angle in [0, pi/2] between the line of w and the plane spanned by u1 and u2 (3D)."""
    n = jnp.cross(u1, u2)
    s = jnp.abs(jnp.sum(w * n, axis=-1)) / (jnp.linalg.norm(w, axis=-1) * jnp.linalg.norm(n, axis=-1))
    return jnp.arcsin(jnp.clip(s, 0.0, 1.0))
