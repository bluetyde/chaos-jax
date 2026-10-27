"""Gate GA of H2D: the 472 confirmed hits of results/h2_main.jsonl through f32_keep alone. Usage: python experiments/h2d_recall.py"""
import json
import os
import time

import numpy as np

from hyperchaos import search3d as s
from hyperchaos.spectrum32 import spectrum32_batch

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
hits = []
for line in open(os.path.join(RES, "h2_main.jsonl")):
    hits.extend(json.loads(line)["hits"])
cs = np.array([h["c"] for h in hits])
t0 = time.time()
e, dv, mx = spectrum32_batch(s.flow, cs, s.START, batch=32)
print("float32 on %d hits: %.0f s" % (len(hits), time.time() - t0))
with np.errstate(invalid="ignore"):
    ok = np.isfinite(e).all(1) & np.isfinite(dv) & np.isfinite(mx) & (mx <= s.F32_LIMIT)
    r_sum = np.abs(e.sum(1) - dv) <= s.F32_SUM_TOL * np.maximum(1.0, np.abs(dv))
    r_l1 = e[:, 0] > s.F32_EPS_L1
    r_zero = np.abs(e).min(1) <= s.F32_ZERO
keep = ok & r_sum & r_l1 & r_zero
print("hits", len(hits), "kept", int(keep.sum()), "recall %.4f" % keep.mean())
lost = [dict(i=int(i), k=hits[i]["k"], exps64=hits[i]["exps"], exps32=e[i].tolist(), sumdiff=float(e[i].sum() - dv[i]), div=float(dv[i]),
             rules=dict(finite=bool(ok[i]), sum=bool(r_sum[i]), l1=bool(r_l1[i]), zero=bool(r_zero[i]))) for i in np.flatnonzero(~keep)]
for l in lost:
    print(l)
json.dump(dict(n=len(hits), kept=int(keep.sum()), lost=lost, seconds=time.time() - t0), open(os.path.join(RES, "h2d_recall.json"), "w"), indent=1)
