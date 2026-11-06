"""H2D calibration (seeds 2000-2005, not used for any gate): float32 batched spectrum on the boundedness-filter survivors,
next to the float64 stage-2 labels already in results/h2c_explore.json. Usage: python experiments/h2d_calib.py <workers>"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def w(seed):
    from hyperchaos import search3d as s
    from hyperchaos.spectrum32 import spectrum32_batch

    rng = np.random.default_rng(seed)
    cs = s.sample(rng, 20000)
    idx, lam, ka = s.screen(cs)
    rows64 = {r["i"]: r for o in json.load(open(os.path.join(RES, "h2c_explore.json"))) if o["seed"] == seed for r in o["rows"]}
    idx = np.array([i for i in idx if rows64[int(i)]["tfirst"] < 0 and rows64[int(i)]["late_range"] >= s.BOUND_STUCK])
    t0 = time.time()
    ex, dv, mx = spectrum32_batch(s.flow, cs[idx], s.START)
    t32 = time.time() - t0
    rows = [dict(seed=seed, i=int(i), label=rows64[int(i)]["label"], conservative=rows64[int(i)]["conservative"], lam1_64=rows64[int(i)]["lam1"],
                 exps32=ex[j].tolist(), div32=float(dv[j]), max32=float(mx[j])) for j, i in enumerate(idx)]
    return dict(seed=seed, n=len(idx), t32=t32, rows=rows)


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=int(sys.argv[1])) as ex:
        out = list(ex.map(w, range(2000, 2006)))
    json.dump(out, open(os.path.join(RES, "h2d_calib.json"), "w"))
    for o in out:
        print(o["seed"], "n", o["n"], "float32 batched %.1f s" % o["t32"])
