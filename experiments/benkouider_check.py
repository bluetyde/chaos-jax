"""Benkouider et al. 2024 system (1): spectrum at the published parameters. Usage: python experiments/benkouider_check.py"""
import json, os, time
import jax.numpy as jnp
from hyperchaos.spectrum import spectrum_p

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

def F(x, p):
    a, b, c, d = p[0], p[1], p[2], p[3]
    return jnp.array([a * x[0] - x[1] * x[2] + x[3], x[0] * x[2] - b * x[1], x[0] * x[1] - c * x[2], -x[1] + d])

out = []
for c in (15.0, 14.0, 9.4, 7.0):
    for x0 in ((10.0, 1.0, 10.0, 1.0), (1.0, 1.0, 1.0, 1.0)):
        for dt in (0.01, 0.005):
            t = time.time()
            r = spectrum_p(F, [8.0, 40.0, c, -0.1], x0, dt=dt, T=20000.0)
            row = dict(c=c, x0=x0, dt=dt, exps=[float(v) for v in r.exponents], se=[float(v) for v in r.se], max_abs=float(r.max_abs), div=float(r.mean_div), s=time.time() - t)
            out.append(row)
            print("c=%4.1f x0=%s dt=%.3f exps=%s se=%s max|x|=%.3g (%.0f s)" % (c, x0, dt, [round(v, 4) for v in row["exps"]], [round(v, 4) for v in row["se"]], row["max_abs"], row["s"]), flush=True)
json.dump(out, open(os.path.join(RES, "benkouider_check.json"), "w"), indent=1)
