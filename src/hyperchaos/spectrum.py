"""Lyapunov spectrum of an autonomous flow by the Benettin QR method (float64, fixed-step RK4).

The state and an orthonormal tangent frame are integrated together. Every `qr_every` steps the
frame is re-orthonormalised with a QR factorisation and log|diag R| is added up. The time
average of the divergence tr J is accumulated along the same path (the sum of the exponents must
equal it, which is checked in the tests).
"""
from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


@dataclass
class SpectrumResult:
    exponents: np.ndarray  # sorted, largest first
    se: np.ndarray  # standard error of each exponent, from block means
    blocks: np.ndarray  # (n_blocks, n) block exponents in the same order as `exponents`
    mean_div: float  # time average of tr J along the trajectory
    max_abs: float  # largest |x| seen at the chunk ends
    finite: bool  # every output is finite
    late_range: np.ndarray  # range of each coordinate over the last block
    x_end: np.ndarray
    dt: float
    T: float
    n_blocks: int


def _build(F, n, dt, qr_every, chunks_per_block, n_blocks, spin_steps):
    """Jitted run(x0, c) for a flow F(x, c). `c` is a runtime parameter array (not part of the cache key)."""
    jac = jax.jacfwd(F, argnums=0)

    def run(x0, c):
        f = lambda x: F(x, c)  # noqa: E731
        jacc = lambda x: jac(x, c)  # noqa: E731

        def aug(x, q):
            j = jacc(x)
            return f(x), j @ q, jnp.trace(j)

        def rk4_aug(carry, _):
            x, q, div = carry
            k1x, k1q, t1 = aug(x, q)
            k2x, k2q, t2 = aug(x + 0.5 * dt * k1x, q + 0.5 * dt * k1q)
            k3x, k3q, t3 = aug(x + 0.5 * dt * k2x, q + 0.5 * dt * k2q)
            k4x, k4q, t4 = aug(x + dt * k3x, q + dt * k3q)
            x = x + dt / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
            q = q + dt / 6.0 * (k1q + 2 * k2q + 2 * k3q + k4q)
            div = div + dt / 6.0 * (t1 + 2 * t2 + 2 * t3 + t4)
            return (x, q, div), None

        def rk4_state(x, _):
            k1 = f(x)
            k2 = f(x + 0.5 * dt * k1)
            k3 = f(x + 0.5 * dt * k2)
            k4 = f(x + dt * k3)
            return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), None

        def chunk(carry, _):
            x, q, ssum, div = carry
            (x, q, d), _ = lax.scan(rk4_aug, (x, q, 0.0), None, length=qr_every)
            q, r = jnp.linalg.qr(q)
            ssum = ssum + jnp.log(jnp.abs(jnp.diag(r)))
            return (x, q, ssum, div + d), x

        def block(carry, _):
            x, q = carry
            (x, q, ssum, div), xs = lax.scan(
                chunk, (x, q, jnp.zeros(n), 0.0), None, length=chunks_per_block
            )
            return (x, q), (ssum, div, xs)

        x, _ = lax.scan(rk4_state, x0, None, length=spin_steps)
        (x, _), (ss, dv, xs) = lax.scan(block, (x, jnp.eye(n)), None, length=n_blocks)
        return ss, dv, xs

    return jax.jit(run)


@lru_cache(maxsize=None)
def _runner(f, n, dt, qr_every, chunks_per_block, n_blocks, spin_steps):
    return _build(lambda x, c: f(x), n, dt, qr_every, chunks_per_block, n_blocks, spin_steps)


@lru_cache(maxsize=None)
def _runner_p(F, n, dt, qr_every, chunks_per_block, n_blocks, spin_steps):
    return _build(F, n, dt, qr_every, chunks_per_block, n_blocks, spin_steps)


def spectrum(f, x0, dt=0.01, T=5000.0, T_spin=200.0, n_blocks=10, qr_time=0.1):
    """Spectrum of x' = f(x) from x0. `f` must be a JAX function of one array argument."""
    return _spectrum(None, f, None, x0, dt, T, T_spin, n_blocks, qr_time)


def spectrum_p(F, c, x0, dt=0.01, T=5000.0, T_spin=200.0, n_blocks=10, qr_time=0.1):
    """Spectrum of x' = F(x, c) where `c` is a parameter array. One compile serves every `c`."""
    return _spectrum(F, None, c, x0, dt, T, T_spin, n_blocks, qr_time)


def _spectrum(F, f, c, x0, dt, T, T_spin, n_blocks, qr_time):
    x0 = jnp.asarray(x0, dtype=jnp.float64)
    n = int(x0.shape[0])
    qr_every = int(round(qr_time / dt))
    chunks_per_block = int(round(T / n_blocks / (qr_every * dt)))
    if abs(chunks_per_block * n_blocks * qr_every * dt - T) > 1e-9:
        raise ValueError("T must be n_blocks * a whole number of QR intervals")
    spin_steps = int(round(T_spin / dt))
    args = (n, float(dt), qr_every, chunks_per_block, int(n_blocks), spin_steps)
    if F is None:
        ss, dv, xs = (np.asarray(a) for a in _runner(f, *args)(x0, None))
    else:
        ss, dv, xs = (np.asarray(a) for a in _runner_p(F, *args)(x0, jnp.asarray(c, dtype=jnp.float64)))
    block_time = chunks_per_block * qr_every * dt
    blocks = ss / block_time
    exps = blocks.mean(axis=0)
    order = np.argsort(-exps, kind="stable")
    blocks = blocks[:, order]
    exps = exps[order]
    se = blocks.std(axis=0, ddof=1) / np.sqrt(n_blocks)
    finite = bool(np.all(np.isfinite(ss)) and np.all(np.isfinite(xs)) and np.all(np.isfinite(dv)))
    with np.errstate(invalid="ignore"):
        max_abs = float(np.max(np.abs(xs))) if finite else float("inf")
    return SpectrumResult(
        exponents=exps,
        se=se,
        blocks=blocks,
        mean_div=float(dv.sum() / (block_time * n_blocks)),
        max_abs=max_abs,
        finite=finite,
        late_range=np.ptp(xs[-1], axis=0),
        x_end=xs[-1, -1],
        dt=float(dt),
        T=float(T),
        n_blocks=int(n_blocks),
    )
