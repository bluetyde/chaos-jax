"""Chase of Kopp and Samuilik 2025 (see docs/PREDICTIONS-KOPP-SAMUILIK.md).
Usage: python experiments/kopp_samuilik.py results/kopp_samuilik.json [part ...]
Parts: starts dt short scan. Without parts, all."""
import json
import sys
import time

import jax.numpy as jnp
import numpy as np

import hyperchaos  # noqa: F401
from hyperchaos.classify import classify
from hyperchaos.spectrum import spectrum

OUT = sys.argv[1]
PARTS = sys.argv[2:] or ["starts", "dt", "short", "scan"]
ALPHA, BETA, B = 1.0, 0.1525, 90.0
rows = []


def ks25(a):
    def f(x):
        return jnp.stack([a * x[1] - (a + ALPHA) * x[0] - BETA * jnp.abs(x[3]) * x[0], -x[0] * x[2], x[0] * x[1] - B, x[0]])

    return f


def run(tag, a, x0, T, dt=0.01, T_spin=500.0):
    t0 = time.time()
    r = spectrum(ks25(a), np.asarray(x0, dtype=float), dt=dt, T=T, T_spin=T_spin)
    c = classify(r, rule="v2")
    mean_abs_x4 = (-r.mean_div - (a + ALPHA)) / BETA
    row = dict(tag=tag, a=a, x0=list(map(float, x0)), T=T, dt=dt, T_spin=T_spin, label=c.label,
               exponents=r.exponents.tolist(), se=r.se.tolist(), mean_div=r.mean_div, implied_mean_abs_x4=float(mean_abs_x4),
               sum_minus_meandiv=float(r.exponents.sum() - r.mean_div), max_abs=r.max_abs, finite=r.finite, secs=time.time() - t0)
    rows.append(row)
    print(tag, "a=%.2f" % a, c.label, np.round(r.exponents, 4).tolist(), "se2 %.4f" % r.se[1], "mean|x4| %.2f" % mean_abs_x4,
          "maxabs %.3g" % r.max_abs, "%.0fs" % row["secs"], flush=True)
    json.dump(rows, open(OUT, "w"), indent=1, default=float)
    return row


STARTS = [(1.0, 1.0, 1.0, 1.0), (0.1, 0.1, 0.1, 0.1), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (-1.0, 1.0, 0.0, 1.0)]
rng = np.random.default_rng(11)
for w, n in ((1.0, 4), (10.0, 3)):
    STARTS += [tuple(rng.uniform(-w, w, 4)) for _ in range(n)]

best = None
if "starts" in PARTS:
    for a in (3.0, 5.0):
        for i, x0 in enumerate(STARTS):
            r = run("starts%d" % i, a, x0, 100000.0)
            if a == 3.0 and best is None and r["finite"] and r["max_abs"] < 1e6 and r["exponents"][0] > 0.5:
                best = x0
if "dt" in PARTS:
    x0 = best or STARTS[0]
    for dt in (0.01, 0.005, 0.0025):
        run("dt", 3.0, x0, 50000.0, dt=dt)
if "short" in PARTS:
    for T in (500.0, 1000.0, 2000.0, 5000.0):
        run("short", 3.0, STARTS[0], T, T_spin=0.0)
if "scan" in PARTS:
    for a in np.arange(1.0, 8.01, 0.25):
        run("scan", float(a), best or STARTS[0], 20000.0)
