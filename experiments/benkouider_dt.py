"""Follow-up (not in the predictions): step-size convergence of the second exponent at c = 15 and 14. Usage: python experiments/benkouider_dt.py"""
import json, os
import jax.numpy as jnp
from hyperchaos.spectrum import spectrum_p

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

def F(x, p):
    a, b, c, d = p[0], p[1], p[2], p[3]
    return jnp.array([a * x[0] - x[1] * x[2] + x[3], x[0] * x[2] - b * x[1], x[0] * x[1] - c * x[2], -x[1] + d])

out = []
for c in (15.0, 14.0):
    for dt in (0.0025, 0.00125):
        r = spectrum_p(F, [8.0, 40.0, c, -0.1], (10.0, 1.0, 10.0, 1.0), dt=dt, T=40000.0)
        row = dict(c=c, dt=dt, exps=[float(v) for v in r.exponents], se=[float(v) for v in r.se], div=float(r.mean_div))
        out.append(row)
        print("c=%4.1f dt=%.5f exps=%s se=%s div=%.4f sum=%.4f" % (c, dt, [round(v, 4) for v in row["exps"]], [round(v, 4) for v in row["se"]], row["div"], sum(row["exps"])), flush=True)
json.dump(out, open(os.path.join(RES, "benkouider_dt.json"), "w"), indent=1)
