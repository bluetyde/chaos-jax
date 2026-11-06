"""Float32 copy of the stage-1 screen (stage 1a and 1b of search3d), same steps and thresholds (H2F W2a, docs/PREDICTIONS-H2F.md).
Used to find out whether a float32 screen keeps the systems the float64 screen keeps, before any GPU kernel is written."""
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from hyperchaos import search3d as s

F32 = jnp.float32


def _rk4(f, x, dt):
    k1 = f(x)
    k2 = f(x + F32(0.5 * dt) * k1)
    k3 = f(x + F32(0.5 * dt) * k2)
    k4 = f(x + F32(dt) * k3)
    return x + F32(dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@lru_cache(maxsize=None)
def _stage1a32(dt=0.02, steps_a=1600, steps_b=400):
    def one(c):
        f = lambda x: s.flow(x, c)  # noqa: E731

        def step_a(carry, _):
            x, m = carry
            x = _rk4(f, x, dt)
            return (x, jnp.maximum(m, jnp.max(jnp.abs(x)))), None

        def step_b(carry, _):
            x, m, lo, hi = carry
            x = _rk4(f, x, dt)
            return (x, jnp.maximum(m, jnp.max(jnp.abs(x))), jnp.minimum(lo, x), jnp.maximum(hi, x)), None

        x0 = jnp.asarray(s.START, dtype=F32)
        (x, m), _ = lax.scan(step_a, (x0, jnp.zeros((), F32)), None, length=steps_a)
        (x, m, lo, hi), _ = lax.scan(step_b, (x, m, x, x), None, length=steps_b)
        moving = jnp.max(hi - lo) > 1e-3
        return (m <= 1e4) & moving

    return jax.jit(jax.vmap(one))


@lru_cache(maxsize=None)
def _stage1b32(dt=0.02, inner=25, outer=1200, spin=200):
    def one(c):
        f = lambda x: s.flow(x, c)  # noqa: E731

        def aug(x, w):
            return jax.jvp(f, (x,), (w,))

        def step(carry, _):
            x, w = carry
            k1x, k1w = aug(x, w)
            k2x, k2w = aug(x + F32(0.5 * dt) * k1x, w + F32(0.5 * dt) * k1w)
            k3x, k3w = aug(x + F32(0.5 * dt) * k2x, w + F32(0.5 * dt) * k2w)
            k4x, k4w = aug(x + F32(dt) * k3x, w + F32(dt) * k3w)
            x = x + F32(dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
            w = w + F32(dt / 6.0) * (k1w + 2 * k2w + 2 * k3w + k4w)
            return (x, w), None

        def chunk(carry, i):
            x, w, acc, m = carry
            (x, w), _ = lax.scan(step, (x, w), None, length=inner)
            nrm = jnp.linalg.norm(w)
            acc = acc + jnp.where(i >= spin, jnp.log(nrm), jnp.zeros((), F32))
            return (x, w / nrm, acc, jnp.maximum(m, jnp.max(jnp.abs(x)))), None

        w0 = jnp.ones(3, F32) / jnp.sqrt(F32(3.0))
        x0 = jnp.asarray(s.START, dtype=F32)
        (x, w, acc, m), _ = lax.scan(chunk, (x0, w0, jnp.zeros((), F32), jnp.zeros((), F32)), jnp.arange(outer))
        t_meas = (outer - spin) * inner * dt
        return acc / F32(t_meas), m <= 1e4

    return jax.jit(jax.vmap(one))


def _pad_run(fn, cs, batch):
    cs = np.asarray(cs, dtype=np.float32)
    outs = None
    for i in range(0, len(cs), batch):
        part = cs[i : i + batch]
        pad = np.zeros((batch, 30), dtype=np.float32)
        pad[: len(part)] = part
        r = fn(jnp.asarray(pad))
        r = r if isinstance(r, tuple) else (r,)
        r = [np.asarray(a)[: len(part)] for a in r]
        outs = r if outs is None else [np.concatenate([o, a]) for o, a in zip(outs, r)]
    return outs


def screen32(cs, batch_a=2048, batch_b=256):
    """Stage 1a then 1b in float32. Same return as search3d.screen: (indices that survive, their exponent estimates, stage-1a mask)."""
    cs = np.asarray(cs)
    keep_a = _pad_run(_stage1a32(), cs, batch_a)[0].astype(bool)
    idx = np.flatnonzero(keep_a)
    if len(idx) == 0:
        return idx, np.empty(0), keep_a
    lam, ok = _pad_run(_stage1b32(), cs[idx], batch_b)
    good = ok.astype(bool) & (lam > s.LAMBDA_KEEP)
    return idx[good], lam[good], keep_a
