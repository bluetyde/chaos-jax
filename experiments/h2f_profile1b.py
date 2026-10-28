"""H2F W3a: where does stage 1b spend its steps? Instrumented copy of search3d._stage1b (same arithmetic) on the stage-1a survivors of seeds 2000-2005.
Per system: the log of the tangent norm after each of the 1200 chunks, and the first chunk with a non-finite state or |x| > 1e4.
Usage: python experiments/h2f_profile1b.py <workers>   -> results/h2f_profile1b.npz and a summary printed."""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
DT, INNER, OUTER, SPIN = 0.02, 25, 1200, 200


def build():
    import jax
    import jax.numpy as jnp
    from jax import lax

    from hyperchaos import search3d as s

    def one(c):
        f = lambda x: s.flow(x, c)  # noqa: E731

        def aug(x, w):
            return jax.jvp(f, (x,), (w,))

        def step(carry, _):
            x, w = carry
            k1x, k1w = aug(x, w)
            k2x, k2w = aug(x + 0.5 * DT * k1x, w + 0.5 * DT * k1w)
            k3x, k3w = aug(x + 0.5 * DT * k2x, w + 0.5 * DT * k2w)
            k4x, k4w = aug(x + DT * k3x, w + DT * k3w)
            x = x + DT / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
            w = w + DT / 6.0 * (k1w + 2 * k2w + 2 * k3w + k4w)
            return (x, w), None

        def chunk(carry, i):
            x, w, m, tf = carry
            (x, w), _ = lax.scan(step, (x, w), None, length=INNER)
            nrm = jnp.linalg.norm(w)
            bad = ~jnp.all(jnp.isfinite(x)) | (jnp.max(jnp.abs(x)) > 1e4)
            tf = jnp.where((tf < 0) & bad, i + 1, tf)
            return (x, w / nrm, jnp.maximum(m, jnp.max(jnp.abs(x))), tf), jnp.log(nrm)

        w0 = jnp.ones(3) / jnp.sqrt(3.0)
        (x, w, m, tf), logn = lax.scan(chunk, (jnp.asarray(s.START), w0, 0.0, -1), jnp.arange(OUTER))
        return logn, m, tf

    return jax.jit(jax.vmap(one))


def w(seed):
    import jax.numpy as jnp

    from hyperchaos import search3d as s

    cs = s.sample(np.random.default_rng(seed), 20000)
    ka = np.zeros(len(cs), bool)
    for i in range(0, len(cs), 2048):
        part = cs[i : i + 2048]
        pad = np.zeros((2048, 30))
        pad[: len(part)] = part
        ka[i : i + len(part)] = s.stage1a(pad)[: len(part)]
    idx = np.flatnonzero(ka)
    run = build()
    t0 = time.time()
    logn = np.empty((len(idx), OUTER))
    m = np.empty(len(idx))
    tf = np.empty(len(idx), dtype=int)
    for i in range(0, len(idx), 256):
        part = cs[idx[i : i + 256]]
        pad = np.zeros((256, 30))
        pad[: len(part)] = part
        a, b, c = run(jnp.asarray(pad))
        logn[i : i + len(part)] = np.asarray(a)[: len(part)]
        m[i : i + len(part)] = np.asarray(b)[: len(part)]
        tf[i : i + len(part)] = np.asarray(c)[: len(part)]
    t_meas = (OUTER - SPIN) * INNER * DT
    lam = np.nansum(logn[:, SPIN:], axis=1) / t_meas
    keep = (m <= 1e4) & (lam > s.LAMBDA_KEEP)
    ref_idx, _, _ = s.screen(cs)
    same = set(idx[keep].tolist()) == set(ref_idx.tolist())
    return dict(seed=seed, idx=idx, logn=logn, m=m, tf=tf, keep=keep, same=same, secs=time.time() - t0)


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=int(sys.argv[1])) as ex:
        out = list(ex.map(w, range(2000, 2006)))
    for o in out:
        print("seed", o["seed"], "stage-1a survivors", len(o["idx"]), "keepers", int(o["keep"].sum()), "matches search3d.screen:", o["same"], "%.0f s" % o["secs"])
    np.savez(os.path.join(RES, "h2f_profile1b.npz"), logn=np.concatenate([o["logn"] for o in out]), m=np.concatenate([o["m"] for o in out]),
             tf=np.concatenate([o["tf"] for o in out]), keep=np.concatenate([o["keep"] for o in out]),
             seed=np.concatenate([[o["seed"]] * len(o["idx"]) for o in out]))
