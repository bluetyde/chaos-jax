"""Two checks after the four-term search: (1) each conservative candidate has a quadratic first integral a^2 - b^2 (derivative zero to rounding),
(2) the coarse invariant grid (step 0.5) still finds Sprott cases B-E. Usage: python experiments/h2_fourterm_checks.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hyperchaos import search3d as s

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
cons = json.load(open(os.path.join(RES, "h2_fourterm_search.json")))["cons"]
rng = np.random.default_rng(0)
res1 = []
for i, h in enumerate(cons):
    c = np.array(h["c"])
    best = None
    for a, b in ((0, 1), (0, 2), (1, 2)):
        worst = 0.0
        for _ in range(200):
            x = rng.normal(size=3) * 2
            f = np.asarray(s.flow(x, c))
            worst = max(worst, abs(2 * x[a] * f[a] - 2 * x[b] * f[b]))
        if best is None or worst < best[0]:
            best = (worst, (a, b))
    res1.append(dict(i=i, max_abs_derivative=best[0], pair="xyz"[best[1][0]] + "^2-" + "xyz"[best[1][1]] + "^2"))
    print("candidate", i, "d/dt(%s) max over 200 random points = %.2e" % (res1[-1]["pair"], best[0]), flush=True)
from h2_fourterm import COARSE, w_chunk
from hyperchaos import enumerate as en
res2 = {}
for k in "BCDE":
    sup = tuple(sorted(s.support_of(s.case_coefficients(k))))
    r = w_chunk(([sup], COARSE))
    res2[k] = dict(n=r["n"], stage2=r["stage2"], hits=len(r["hits"]), cons=len(r["cons"]))
    print("coarse grid, case", k, res2[k], flush=True)
json.dump(dict(first_integrals=res1, coarse_control=res2), open(os.path.join(RES, "h2_fourterm_checks.json"), "w"), indent=1)
