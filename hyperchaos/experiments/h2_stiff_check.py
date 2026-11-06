"""Re-confirm the night-run hits with first exponent above 10 at smaller steps. Usage: python experiments/h2_stiff_check.py <tag> [lam_min]"""
import glob, json, os, sys
import numpy as np
from hyperchaos import search3d as s
from hyperchaos.classify import classify
from hyperchaos.spectrum import spectrum_p

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
tag = sys.argv[1]
lam_min = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
hits = [h for p in sorted(glob.glob(os.path.join(RES, "h2_%s_b*.json" % tag))) for h in json.load(open(p))["hits"]]
sel = [h for h in hits if h["exps"][0] > lam_min]
print("selected", len(sel), "of", len(hits), flush=True)
out = []
for i, h in enumerate(sel):
    c = np.array(h["c"])
    row = dict(i=i, k=h["k"], stored=h["exps"], runs={})
    for dt in (0.01, 0.005, 0.0025):
        for name, x0 in (("S1", s.START), ("S2", s.START2)):
            r = spectrum_p(s.flow, c, x0, dt=dt, T=20000.0)
            k = classify(r, rule="v2")
            row["runs"]["%s_%g" % (name, dt)] = dict(label=k.label, exps=[float(v) for v in r.exponents], div=float(r.mean_div), max_abs=float(r.max_abs))
    out.append(row)
    print(i, "k=%d stored l1=%.2f" % (h["k"], h["exps"][0]), {k: (v["label"], round(v["exps"][0], 2)) for k, v in row["runs"].items()}, flush=True)
json.dump(out, open(os.path.join(RES, "h2_%s_stiff.json" % tag), "w"), indent=1)
