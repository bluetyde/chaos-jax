"""Check the 4D four-term conservative candidates for a quadratic first integral a^2 - b^2 (any pair of variables) and summarise them.
Usage: python experiments/h3_cons_check.py"""
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h3_fourterm4d import RES  # noqa: E402

from hyperchaos import enum4 as e4  # noqa: E402

cons = []
for f in sorted(glob.glob(os.path.join(RES, "h3_4d", "chunk_*.json"))):
    cons += json.load(open(f))["cons"]
rng = np.random.default_rng(2)
rows = []
for i, h in enumerate(cons):
    c = np.array(h["c"])
    best = (1e9, None)
    for a in range(4):
        for b in range(a + 1, 4):
            worst = 0.0
            for _ in range(100):
                x = rng.normal(size=3 + 1) * 2
                f = np.asarray(e4.flow4(x, c))
                worst = max(worst, abs(2 * x[a] * f[a] - 2 * x[b] * f[b]))
            if worst < best[0]:
                best = (worst, "xyzw"[a] + "^2-" + "xyzw"[b] + "^2")
    rows.append(dict(i=i, rank=h["rank"], label=h["label"], l1=h["exps"][0], max_abs_derivative=best[0], integral=best[1], has_integral=bool(best[0] < 1e-9), support=h["support"]))
print("conservative candidates:", len(rows), " with an a^2-b^2 first integral:", sum(r["has_integral"] for r in rows))
print("labels:", dict(Counter(r["label"] for r in rows)), " distinct supports:", len({tuple(r["support"]) for r in rows}))
for r in rows:
    if not r["has_integral"]:
        print("no a^2-b^2 integral:", r["i"], r["label"], r["rank"], np.round(r["l1"], 4), r["support"])
json.dump(rows, open(os.path.join(RES, "h3_cons_check.json"), "w"), indent=1)
