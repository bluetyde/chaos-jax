"""Gate GA of H2C: the 472 confirmed hits of results/h2_main.jsonl through bound_keep alone. Usage: python experiments/h2c_recall.py"""
import json
import os

import numpy as np

from hyperchaos import search3d as s

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
hits = []
for line in open(os.path.join(RES, "h2_main.jsonl")):
    hits.extend(json.loads(line)["hits"])
cs = np.array([h["c"] for h in hits])
tf, rg = s.bound_stats(cs)
keep = (tf < 0) & (rg >= s.BOUND_STUCK)
print("hits", len(hits), "kept", int(keep.sum()), "recall %.4f" % (keep.mean()))
lost = [dict(i=int(i), k=hits[i]["k"], exps=hits[i]["exps"], tfirst=float(tf[i]), late_range=float(rg[i])) for i in np.flatnonzero(~keep)]
for l in lost:
    print(l)
json.dump(dict(n=len(hits), kept=int(keep.sum()), lost=lost), open(os.path.join(RES, "h2c_recall.json"), "w"), indent=1)
