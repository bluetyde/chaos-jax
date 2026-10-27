"""Numbers for the H1 results section. Usage: H1_T=5000|100000 python experiments/h1_report.py results/out.json"""
import json
import os
import sys
import time

import numpy as np

import hyperchaos  # noqa: F401
from hyperchaos.classify import classify
from hyperchaos.spectrum import spectrum
from hyperchaos.systems import ALL, NEGATIVE, POSITIVE
from hyperchaos.verify import verify

T = float(os.environ.get("H1_T", "5000"))
OUT = sys.argv[1]
SEED = 1

rows = []
for s in POSITIVE + NEGATIVE:
    t0 = time.process_time()
    x0 = s.start(SEED)
    r = spectrum(s.f, x0, dt=s.dt, T=T)
    c2 = classify(r, rule="v2")
    c1 = classify(r, rule="v1")
    row = {
        "id": s.id,
        "dt": s.dt,
        "T": T,
        "expected": s.expected,
        "label_v2": c2.label,
        "label_v1": c1.label,
        "exponents": r.exponents.tolist(),
        "se": r.se.tolist(),
        "sum_minus_meandiv": float(r.exponents.sum() - r.mean_div) if r.finite else None,
        "mean_div": r.mean_div if r.finite else None,
        "conservative": c2.conservative,
        "z_err": c2.z_err,
        "published": list(s.published),
    }
    if s.published:
        pub = np.sort(np.array(s.published))[::-1]
        row["diff_to_published"] = (r.exponents - pub).tolist()
    if c2.label in ("chaotic", "hyperchaotic"):
        v = verify(s, r, x0, T=T, v4_T=0.4 * T)
        row["verification"] = v
    row["cpu_seconds"] = time.process_time() - t0
    rows.append(row)
    print(s.id, c2.label, "(v1:", c1.label + ")", np.round(r.exponents, 4).tolist(), "cpu %.0fs" % row["cpu_seconds"], flush=True)
json.dump(rows, open(OUT, "w"), indent=1, default=float)
