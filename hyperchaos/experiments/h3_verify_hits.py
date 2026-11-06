"""Amendment A3 verification of the 4D four-term search hits (docs/PREDICTIONS-H3-FOURTERM-4D.md): re-run every hit from two generic (non-symmetric) starts
at dt 0.01 and 0.005, require the same chaotic or hyperchaotic label, a bounded orbit that does not settle (late range above 1e-3), and a negative exponent sum.
Usage: python experiments/h3_verify_hits.py"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h3_fourterm4d import RES, evaluate4  # noqa: E402

GENERIC = [np.array([0.31, -0.72, 0.46, 0.83]), np.array([-0.4, 0.2, 0.9, -0.6])]

hits = []
for f in sorted(glob.glob(os.path.join(RES, "h3_4d", "chunk_*.json"))):
    hits += json.load(open(f))["hits"]
print("hits to verify:", len(hits), flush=True)
rows = []
for i, h in enumerate(hits):
    c = np.array(h["c"])
    row = dict(i=i, rank=h["rank"], label=h["label"], support=h["support"], orig_start=h["start"], runs={})
    ok = True
    for si, x0 in enumerate(GENERIC):
        for dt in (0.01, 0.005):
            r, k = evaluate4(c, x0, 20000.0, dt=dt)
            settled = bool(np.max(r.late_range) < 1e-3)
            good = (k.label in ("chaotic", "hyperchaotic")) and (not k.conservative) and (not settled) and float(np.sum(r.exponents)) < -0.005
            ok &= good
            row["runs"]["G%d_%g" % (si, dt)] = dict(label=k.label, conservative=bool(k.conservative), settled=settled, exps=np.round(r.exponents, 4).tolist(), max_abs=float(r.max_abs), late_range=float(np.max(r.late_range)))
    row["verified"] = bool(ok)
    rows.append(row)
    print(i, h["label"], "VERIFIED" if ok else "not verified", {k: (v["label"], "settled" if v["settled"] else "") for k, v in row["runs"].items()}, flush=True)
print("verified", sum(r["verified"] for r in rows), "of", len(rows))
json.dump(rows, open(os.path.join(RES, "h3_verify_hits.json"), "w"), indent=1)
