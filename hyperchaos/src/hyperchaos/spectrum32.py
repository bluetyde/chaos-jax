"""Batched float32 Lyapunov spectrum (coarse screen only; docs/PREDICTIONS-H2D.md).

Same method as spectrum.py (Benettin QR, fixed-step RK4) but every array is float32 and a whole batch of
parameter vectors is integrated at once with vmap. It never decides a hit: survivors go to the float64 evaluator.
"""
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

F32 = jnp.float32


@lru_cache(maxsize=None)
def _runner32(F, dt, qr_every, chunks, spin_steps):
    jac = jax.jacfwd(F, argnums=0)
    dt32 = F32(dt)

    def one(x0, c):
        f = lambda x: F(x, c)  # noqa: E731

        def aug(x, q):
            j = jac(x, c)
            return f(x), j @ q, jnp.trace(j)

        def rk4_aug(carry, _):
            x, q, div = carry
            k1x, k1q, t1 = aug(x, q)
            k2x, k2q, t2 = aug(x + F32(0.5) * dt32 * k1x, q + F32(0.5) * dt32 * k1q)
            k3x, k3q, t3 = aug(x + F32(0.5) * dt32 * k2x, q + F32(0.5) * dt32 * k2q)
            k4x, k4q, t4 = aug(x + dt32 * k3x, q + dt32 * k3q)
            x = x + dt32 / F32(6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
            q = q + dt32 / F32(6.0) * (k1q + 2 * k2q + 2 * k3q + k4q)
            div = div + dt32 / F32(6.0) * (t1 + 2 * t2 + 2 * t3 + t4)
            return (x, q, div), None

        def rk4_state(x, _):
            k1 = f(x)
            k2 = f(x + F32(0.5) * dt32 * k1)
            k3 = f(x + F32(0.5) * dt32 * k2)
            k4 = f(x + dt32 * k3)
            return x + dt32 / F32(6.0) * (k1 + 2 * k2 + 2 * k3 + k4), None

        def chunk(carry, _):
            x, q, ssum, div, mx = carry
            (x, q, d), _ = lax.scan(rk4_aug, (x, q, jnp.zeros((), F32)), None, length=qr_every)
            q, r = jnp.linalg.qr(q)
            ssum = ssum + jnp.log(jnp.abs(jnp.diag(r)))
            return (x, q, ssum, div + d, jnp.maximum(mx, jnp.max(jnp.abs(x)))), None

        x, _ = lax.scan(rk4_state, x0, None, length=spin_steps)
        init = (x, jnp.eye(3, dtype=F32), jnp.zeros(3, F32), jnp.zeros((), F32), jnp.zeros((), F32))
        (x, _, ssum, div, mx), _ = lax.scan(chunk, init, None, length=chunks)
        return ssum, div, mx

    return jax.jit(jax.vmap(one, in_axes=(None, 0)))


def spectrum32_batch(F, cs, x0, dt=0.01, T=5000.0, T_spin=200.0, qr_time=0.1, batch=64):
    """Returns per system (exponents sorted largest first, mean divergence, largest |x| at chunk ends); no standard error."""
    cs = np.asarray(cs)
    qr_every = int(round(qr_time / dt))
    chunks = int(round(T / (qr_every * dt)))
    run = _runner32(F, float(dt), qr_every, chunks, int(round(T_spin / dt)))
    x0 = jnp.asarray(x0, dtype=F32)
    exps = np.empty((len(cs), 3))
    div = np.empty(len(cs))
    mx = np.empty(len(cs))
    for i in range(0, len(cs), batch):
        part = cs[i : i + batch]
        pad = np.zeros((batch, cs.shape[1]), dtype=np.float32)
        pad[: len(part)] = part
        ss, dv, m = (np.asarray(a, dtype=np.float64) for a in run(x0, jnp.asarray(pad)))
        k = len(part)
        exps[i : i + k] = -np.sort(-(ss[:k] / (chunks * qr_every * dt)), axis=1)
        div[i : i + k] = dv[:k] / (chunks * qr_every * dt)
        mx[i : i + k] = m[:k]
    return exps, div, mx
