"""H2F W2a: float32 stage 1 against float64 stage 1. Usage: python experiments/h2f_screen32.py <workers> <n_seeds> <seed0>
Also runs the 472 known hits through the float32 screen alone. Writes results/h2f_screen32.json."""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def w(seed):
    from hyperchaos import search3d as s
    from hyperchaos.screen32 import screen32

    cs = s.sample(np.random.default_rng(seed), 20000)
    t0 = time.time()
    i64, l64, ka64 = s.screen(cs)
    t64 = time.time() - t0
    t0 = time.time()
    i32, l32, ka32 = screen32(cs)
    t32 = time.time() - t0
    a, b = set(i64.tolist()), set(i32.tolist())
    return dict(seed=seed, keep64=len(a), keep32=len(b), lost=sorted(a - b), extra=len(b - a), ka64=int(ka64.sum()), ka32=int(ka32.sum()),
                ka_lost=int((ka64 & ~ka32).sum()), ka_extra=int((~ka64 & ka32).sum()), t64=t64, t32=t32,
                lost_lam64=[float(l64[list(i64).index(i)]) for i in sorted(a - b)])


def w_hits(_):
    from hyperchaos import search3d as s
    from hyperchaos.screen32 import screen32

    hits = [h for line in open(os.path.join(RES, "h2_main.jsonl")) for h in json.loads(line)["hits"]]
    cs = np.array([h["c"] for h in hits])
    i32, l32, ka32 = screen32(cs)
    kept = set(i32.tolist())
    return dict(n=len(hits), kept=len(kept), lost=[dict(i=i, k=hits[i]["k"], exps=hits[i]["exps"], stage1a=bool(ka32[i])) for i in range(len(hits)) if i not in kept])


if __name__ == "__main__":
    workers, n, seed0 = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fh = ex.submit(w_hits, 0)
        rows = list(ex.map(w, range(seed0, seed0 + n)))
        hits = fh.result()
    tot = {k: sum(r[k] for r in rows) for k in ("keep64", "keep32", "extra", "ka64", "ka32", "ka_lost", "ka_extra", "t64", "t32")}
    lost = sum(len(r["lost"]) for r in rows)
    print("seeds", n, tot, "float64 keepers lost by float32:", lost, "wall %.0f s" % (time.time() - t0))
    print("known hits through float32 stage 1: kept %d of %d" % (hits["kept"], hits["n"]))
    for l in hits["lost"]:
        print("  lost hit", l)
    json.dump(dict(rows=rows, hits=hits, totals=tot, lost_total=lost), open(os.path.join(RES, "h2f_screen32.json"), "w"))
