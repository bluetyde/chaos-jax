"""Chase of Kopp 2025 (see docs/PREDICTIONS-KOPP.md). Usage: python experiments/kopp_chase.py results/kopp_chase.json [part ...]
Parts: table starts scan dt short long. Without parts, all but long run."""
import json
import sys
import time

import numpy as np

import hyperchaos  # noqa: F401
from hyperchaos.classify import classify
from hyperchaos.spectrum import spectrum
from hyperchaos.systems import kopp25

OUT = sys.argv[1]
PARTS = sys.argv[2:] or ["table", "starts", "scan", "dt", "short"]
rows = []


def run(tag, a, x0, T, dt=0.01, T_spin=200.0, **extra):
    f = kopp25(a=a, b=0.55)
    t0 = time.time()
    r = spectrum(f, np.asarray(x0, dtype=float), dt=dt, T=T, T_spin=T_spin)
    c = classify(r, rule="v2")
    row = dict(tag=tag, a=a, x0=list(map(float, x0)), T=T, dt=dt, T_spin=T_spin, label=c.label,
               exponents=r.exponents.tolist(), se=r.se.tolist(), sum_minus_meandiv=float(r.exponents.sum() - r.mean_div),
               max_abs=r.max_abs, finite=r.finite, secs=time.time() - t0, **extra)
    rows.append(row)
    print(tag, "a=%.3f" % a, c.label, np.round(r.exponents, 4).tolist(), "se2 %.4f" % r.se[1], "%.0fs" % row["secs"], flush=True)
    json.dump(rows, open(OUT, "w"), indent=1, default=float)
    return row


ONES = (1.0, 1.0, 1.0, 1.0)
if "table" in PARTS:
    for a in (0.15, 0.55, 0.6, 0.62, 0.65, 0.8, 1.5):
        run("table", a, ONES, 100000.0)
if "starts" in PARTS:
    rng = np.random.default_rng(7)
    starts = [ONES, (-1.0, -1.0, 1.0, -1.0), (1.0, 1.0, 1.0, 1.01), (0.5, 0.5, 0.5, 0.5), (0.0, 0.0, 0.0, 0.0)]
    for w in (1.0, 3.0, 10.0):
        for _ in range(3):
            starts.append(tuple(rng.uniform(-w, w, 4)))
    for a in (0.62, 0.65):
        for i, x0 in enumerate(starts):
            run("start%d" % i, a, x0, 100000.0)
if "scan" in PARTS:
    for a in np.round(np.arange(0.10, 2.0001, 0.02), 3):
        run("scan", float(a), ONES, 20000.0)
if "dt" in PARTS:
    for a in (0.62, 0.65):
        for dt in (0.01, 0.005, 0.0025):
            run("dt", a, ONES, 20000.0, dt=dt)
if "short" in PARTS:
    for a in (0.62, 0.65):
        for T in (500.0, 1000.0, 2000.0, 5000.0):
            run("short", a, ONES, T, T_spin=0.0)
if "long" in PARTS:
    run("long", 0.62, ONES, 1000000.0)
