"""H2C exploration (calibration only, seeds 2000-2005, not used for any gate): for every stage-1b survivor, the stage-2 label
(H1 evaluator, T = 5000) and the state-only boundedness statistics at T = 5000. Usage: python experiments/h2c_explore.py <workers>"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def w(seed):
    from hyperchaos import search3d as s
    from hyperchaos.classify import classify
    from hyperchaos.spectrum import spectrum_p

    rng = np.random.default_rng(seed)
    cs = s.sample(rng, 20000)
    t0 = time.time()
    idx, lam, ka = s.screen(cs)
    t_screen = time.time() - t0
    t0 = time.time()
    tf, rg = s.bound_stats(cs[idx], 5000.0, 0.02)
    t_bound = time.time() - t0
    rows = []
    t0 = time.time()
    for j, i in enumerate(idx):
        r = spectrum_p(s.flow, cs[i], s.START, dt=0.01, T=5000.0)
        k = classify(r, rule="v2")
        rows.append(dict(seed=seed, i=int(i), label=k.label, conservative=bool(k.conservative), tfirst=float(tf[j]), late_range=float(rg[j]),
                         lam1=float(r.exponents[0]), max_abs=float(r.max_abs)))
    return dict(seed=seed, t_screen=t_screen, t_bound=t_bound, t_stage2=time.time() - t0, rows=rows)


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=int(sys.argv[1])) as ex:
        out = list(ex.map(w, range(2000, 2006)))
    json.dump(out, open(os.path.join(RES, "h2c_explore.json"), "w"))
    for o in out:
        print(o["seed"], "survivors", len(o["rows"]), "screen %.1f bound %.1f stage2 %.1f" % (o["t_screen"], o["t_bound"], o["t_stage2"]))
