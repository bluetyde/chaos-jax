"""H2E pooled pipeline: phase 1 screen + boundedness per chunk, phase 2 float32 pass on the pooled survivors, phase 3 float64 stage 2 + confirmation.
Usage: python experiments/h2_pooled.py <tag> <workers> <n_seeds> <seed0> [batch]   -> results/h2_<tag>.json"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def w_phase1(seed):
    from hyperchaos import search3d as s

    cs = s.sample(np.random.default_rng(seed), 20000)
    idx, lam, ka = s.screen(cs)
    n_b = len(idx)
    if len(idx):
        idx = idx[s.bound_keep(cs[idx])]
    return dict(seed=seed, keep_a=int(ka.sum()), keep_b=n_b, bound=int(len(idx)), cs=cs[idx])


def w_phase2(args):
    cs, batch = args
    from hyperchaos import search3d as s

    return s.f32_keep(cs, batch=batch) if len(cs) else np.zeros(0, bool)


def w_phase3(c):
    from hyperchaos import search3d as s

    ok, r = s.is_chaotic(c, 5000.0)
    if not ok:
        return None
    ok3, r3 = s.confirm(c, 20000.0)
    if not ok3:
        return dict(stage2=1, hit=None)
    sup = s.support_of(c)
    return dict(stage2=1, hit=dict(c=c.tolist(), k=len(sup), canon=list(s.canonical_support(sup)), exps=r3.exponents.tolist(), se=r3.se.tolist()))


if __name__ == "__main__":
    tag, workers, n_seeds, seed0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    batch = int(sys.argv[5]) if len(sys.argv) > 5 else 128
    t_all = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        t0 = time.time()
        p1 = list(ex.map(w_phase1, range(seed0, seed0 + n_seeds)))
        t1 = time.time() - t0
        pool = np.concatenate([p["cs"] for p in p1])
        slices = np.array_split(pool, workers)
        t0 = time.time()
        keeps = list(ex.map(w_phase2, [(sl, batch) for sl in slices]))
        t2 = time.time() - t0
        keep = np.concatenate(keeps)
        kept = pool[keep]
        t0 = time.time()
        p3 = [r for r in ex.map(w_phase3, list(kept)) if r is not None]
        t3 = time.time() - t0
    hits = [r["hit"] for r in p3 if r["hit"] is not None]
    out = dict(tag=tag, workers=workers, n_seeds=n_seeds, seed0=seed0, batch=batch, draws=20000 * n_seeds,
               keep_a=sum(p["keep_a"] for p in p1), keep_b=sum(p["keep_b"] for p in p1), bound=int(len(pool)), f32=int(keep.sum()),
               stage2=len(p3), confirmed=len(hits), t_phase1=t1, t_phase2=t2, t_phase3=t3, wall=time.time() - t_all, hits=hits)
    json.dump(out, open(os.path.join(RES, "h2_%s.json" % tag), "w"))
    print({k: v for k, v in out.items() if k != "hits"})
