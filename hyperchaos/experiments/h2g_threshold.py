"""H2G: store float32 stage-1 exponent estimates and flags, then sweep the exponent threshold. Usage: python experiments/h2g_threshold.py <workers> <n_seeds> <seed0>
Writes results/h2g_threshold.json (per-seed arrays and the sweep)."""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
THETAS = (0.01, 0.005, 0.0, -0.01, -0.02, -0.05, -0.1)


def stage1_f32(cs):
    """Float32 stage 1 split into its parts: stage-1a mask, then exponent estimate and boundedness flag for the stage-1a survivors."""
    from hyperchaos import screen32 as t

    ka = t._pad_run(t._stage1a32(), cs, 2048)[0].astype(bool)
    idx = np.flatnonzero(ka)
    if len(idx) == 0:
        return ka, idx, np.empty(0), np.empty(0, bool)
    lam, ok = t._pad_run(t._stage1b32(), cs[idx], 256)
    return ka, idx, lam, ok.astype(bool)


def w(seed):
    from hyperchaos import search3d as s

    cs = s.sample(np.random.default_rng(seed), 20000)
    i64, _, _ = s.screen(cs)
    ka, idx, lam, ok = stage1_f32(cs)
    # float32 keep sets for each threshold, as indices into cs
    sets = {str(th): idx[ok & (lam > th)].tolist() for th in THETAS}
    # for the float64 keepers: the float32 stage-1a flag, boundedness flag and exponent (nan when stage 1a dropped them)
    pos = {int(i): j for j, i in enumerate(idx)}
    rows = []
    for i in i64.tolist():
        j = pos.get(i)
        rows.append(dict(i=i, ka32=bool(ka[i]), ok32=None if j is None else bool(ok[j]), lam32=None if j is None else float(lam[j])))
    return dict(seed=seed, keep64=i64.tolist(), sets=sets, rows=rows)


def w_hits(_):
    from hyperchaos import search3d as s  # noqa: F401

    hits = [h for line in open(os.path.join(RES, "h2_main.jsonl")) for h in json.loads(line)["hits"]]
    cs = np.array([h["c"] for h in hits])
    ka, idx, lam, ok = stage1_f32(cs)
    pos = {int(i): j for j, i in enumerate(idx)}
    rows = []
    for i, h in enumerate(hits):
        j = pos.get(i)
        rows.append(dict(i=i, k=h["k"], lam64=h["exps"][0], ka32=bool(ka[i]), ok32=None if j is None else bool(ok[j]), lam32=None if j is None else float(lam[j])))
    return rows


if __name__ == "__main__":
    workers, n, seed0 = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fh = ex.submit(w_hits, 0)
        rows = list(ex.map(w, range(seed0, seed0 + n)))
        hrows = fh.result()
    n64 = sum(len(r["keep64"]) for r in rows)
    sweep = {}
    for th in THETAS:
        kept64 = 0
        n32 = 0
        for r in rows:
            s32 = set(r["sets"][str(th)])
            kept64 += len(s32 & set(r["keep64"]))
            n32 += len(s32)
        hk = sum(1 for h in hrows if h["ka32"] and h["ok32"] and h["lam32"] > th)
        sweep[str(th)] = dict(keepers_kept=kept64, keepers_kept_share=kept64 / n64, f32_keepers=n32, f32_over_f64=n32 / n64, hits_kept=hk)
        print("theta %6.3f  float64 keepers kept %4d of %d (%.3f)  float32 keepers %5d (x%.2f)  hits kept %d of %d" % (th, kept64, n64, kept64 / n64, n32, n32 / n64, hk, len(hrows)))
    print("lost hits (float32 stage-1a flag, boundedness flag, float32 exponent, float64 exponent):")
    for h in hrows:
        if not (h["ka32"] and h["ok32"] and h["lam32"] > 0.01):
            print("  ", h)
    json.dump(dict(rows=rows, hits=hrows, sweep=sweep, n64=n64), open(os.path.join(RES, "h2g_threshold.json"), "w"))
    print("wall %.0f s" % (time.time() - t0))
