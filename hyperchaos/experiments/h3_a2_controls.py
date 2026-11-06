"""Gate GA2-1 of docs/PREDICTIONS-H3-FOURTERM-4D.md (Amendment A2): the new screen keeps Li-Sprott and the 4D embeddings of Sprott cases B-E (3D flow plus w' = -w).
Usage: python experiments/h3_a2_controls.py"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h3_fourterm4d import RES, c_from_system, confirm4, screen4  # noqa: E402

from hyperchaos import enum4 as e4  # noqa: E402
from hyperchaos import search3d as s3  # noqa: E402


def embed(case):
    c3 = s3.case_coefficients(case)
    c = np.zeros(e4.N * e4.NM)
    for slot in np.flatnonzero(c3):
        eq, m = divmod(int(slot), 10)
        mono = tuple(list(s3.MONOS[m]) + [0])
        c[e4.NM * eq + e4._MIDX[mono]] = c3[slot]
    c[e4.NM * 3 + e4._MIDX[(0, 0, 0, 1)]] = -1.0  # w' = -w
    return c


out = {}
systems = [("LS", c_from_system("LS"))] + [("emb_" + k, embed(k)) for k in "BCDE"]
for name, c in systems:
    any0, pick = screen4(np.array([c]))
    x0 = e4.STARTS4[pick[0]] if pick[0] >= 0 else None
    lab, r = confirm4(c, x0, 20000.0) if x0 is not None else (None, None)
    out[name] = dict(terms=int(np.count_nonzero(c)), stage0=bool(any0[0]), kept=x0 is not None, start=None if x0 is None else x0.tolist(), label=lab, exps=None if r is None else np.round(r.exponents, 4).tolist())
    print(name, out[name], flush=True)
print("kept", sum(v["kept"] for v in out.values()), "of", len(out))
json.dump(out, open(os.path.join(RES, "h3_a2_controls.json"), "w"), indent=1)
