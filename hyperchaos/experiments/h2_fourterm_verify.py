"""Verify the conservative four-term candidates: both starts, dt 0.01 / 0.005 / 0.0025, T = 20000. Usage: python experiments/h2_fourterm_verify.py"""
import json, os
import numpy as np
from hyperchaos import search3d as s
from hyperchaos.classify import classify
from hyperchaos.spectrum import spectrum_p

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
cons = json.load(open(os.path.join(RES, "h2_fourterm_search.json")))["cons"]
out = []
for i, h in enumerate(cons):
    c = np.array(h["c"])
    row = dict(i=i, c=h["c"], runs={})
    for dt in (0.01, 0.005, 0.0025):
        for name, x0 in (("S1", s.START), ("S2", s.START2)):
            r = spectrum_p(s.flow, c, x0, dt=dt, T=20000.0)
            k = classify(r, rule="v2")
            row["runs"]["%s_%g" % (name, dt)] = dict(label=k.label, cons=bool(k.conservative), exps=[float(v) for v in r.exponents], se=[float(v) for v in r.se], max_abs=float(r.max_abs), late_range=r.late_range.tolist())
            print(i, name, dt, k.label, "cons" if k.conservative else "", np.round(r.exponents, 4).tolist(), "se", np.round(r.se, 4).tolist(), "max|x| %.3g" % r.max_abs, flush=True)
    out.append(row)
json.dump(out, open(os.path.join(RES, "h2_fourterm_verify.json"), "w"), indent=1)
