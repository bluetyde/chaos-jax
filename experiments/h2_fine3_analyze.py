"""Analyse the fine-pass conservative candidates: quadratic first integral a^2 - b^2 test (and k a^2 - b^2 forms are not tried), and overlap with the coarse-pass candidates.
Usage: python experiments/h2_fine3_analyze.py"""
import json, os
import numpy as np
from hyperchaos import search3d as s

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
fine = json.load(open(os.path.join(RES, "h2_fine3_summary.json")))["cons"]
coarse = json.load(open(os.path.join(RES, "h2_fourterm_search.json")))["cons"]
rng = np.random.default_rng(1)
rows = []
for i, h in enumerate(fine):
    c = np.array(h["c"])
    best = (1e9, None)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        worst = 0.0
        for _ in range(200):
            x = rng.normal(size=3) * 2
            f = np.asarray(s.flow(x, c))
            worst = max(worst, abs(2 * x[a] * f[a] - 2 * x[b] * f[b]))
        if worst < best[0]:
            best = (worst, "xyz"[a] + "^2-" + "xyz"[b] + "^2")
    rows.append(dict(i=i, support=h["support"], rank=h["rank"], l1=h["exps"][0], max_abs_derivative=best[0], integral=best[1], has_integral=bool(best[0] < 1e-9)))
print("fine-pass conservative candidates:", len(fine), " with a^2-b^2 first integral:", sum(r["has_integral"] for r in rows))
for r in rows:
    if not r["has_integral"]:
        print("NO integral of that form:", r)
cs = {tuple(np.round(h["c"], 9)) for h in fine}
r3 = [h for h in coarse if h["rank"] == 3]
print("coarse candidates of rank 3:", len(r3), " re-found in the fine pass:", sum(tuple(np.round(h["c"], 9)) in cs for h in r3))
print("distinct supports among fine candidates:", len({tuple(h["support"]) for h in fine}))
json.dump(rows, open(os.path.join(RES, "h2_fine3_analysis.json"), "w"), indent=1)
