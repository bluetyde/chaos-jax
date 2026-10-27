"""A3 part 2: our own LSS for rho, sigma, beta. Design in docs/PREDICTIONS-A3.md.

python experiments/phaseA3_lss.py run          T = 40, 80, 160, 320 at alpha = 10, and alpha = 3, 30 at T = 160
python experiments/phaseA3_lss.py control      C4 only
python experiments/phaseA3_lss.py report       gates A3.3 and A3.4
"""
import json
import pathlib
import sys
import time

import numpy as np

import jax
import jax.numpy as jnp
import chaos_jax  # noqa: F401
from chaos_jax.fixes import lss_derivative
from chaos_jax.integrate import integrate, integrate_final
from chaos_jax.systems import lorenz, lorenz_params

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
DT, N, CHUNK = 0.01, 200, 25
P = lorenz_params()
NAMES = {0: "sigma", 1: "rho", 2: "beta"}
NILSS = {"rho": (1.01708, 0.00007), "sigma": (0.13375, 0.00008), "beta": (-1.65927, 0.00116)}   # A2 stage 2
ORACLE = {"rho": (1.00317, 0.00074), "sigma": (0.14882, 0.00072), "beta": (-1.65814, 0.00595)}  # A1


def zobs(x):
    return x[2]


def start_points():
    k = jax.random.split(jax.random.PRNGKey(90), 3)
    s = jnp.stack([jax.random.uniform(k[0], (N,), minval=-15.0, maxval=15.0),
                   jax.random.uniform(k[1], (N,), minval=-15.0, maxval=15.0),
                   jax.random.uniform(k[2], (N,), minval=5.0, maxval=45.0)], axis=1)
    return jax.vmap(lambda x: integrate_final(lorenz, x, P, DT, 5000))(s)


def lss_all(x0s, T, alpha):
    n = int(round(T / DT))

    def one(x0):
        xs = integrate(lorenz, x0, P, DT, n)[1:]
        return jnp.stack([lss_derivative(lorenz, zobs, xs, P, i, DT, alpha)[0] for i in (0, 1, 2)])

    f = jax.jit(jax.vmap(one))
    return np.concatenate([np.asarray(f(x0s[i:i + CHUNK])) for i in range(0, x0s.shape[0], CHUNK)])  # (N, 3) sigma, rho, beta


mode = sys.argv[1]
if mode == "run":
    x0s = start_points()
    out, t0 = {}, time.time()
    for T, alpha in [(40, 10.0), (80, 10.0), (160, 10.0), (320, 10.0), (160, 3.0), (160, 30.0)]:
        d = lss_all(x0s, T, alpha)
        out[f"T{T}_a{alpha:g}"] = {"T": T, "alpha": alpha, "per_start": {NAMES[i]: d[:, i].tolist() for i in range(3)}}
        print(f"T={T} alpha={alpha:g} done t={time.time()-t0:.0f}s", flush=True)
    (OUT / "phaseA3_lss_raw.json").write_text(json.dumps(out) + "\n")
elif mode == "control":
    d = json.loads((OUT / "phaseA3_lss_raw.json").read_text())["T80_a10"]["per_start"]["rho"]
    m = float(np.mean(d))
    print("C4 LSS rho T=80 mean", round(m, 5), "pass", abs(m - 1.0158) <= 0.005)
elif mode == "report":
    raw = json.loads((OUT / "phaseA3_lss_raw.json").read_text())
    res = {}
    print(f"{'case':<10} {'rho':>18} {'sigma':>18} {'beta':>18}")
    for key, r in raw.items():
        row = {}
        for nm in ("rho", "sigma", "beta"):
            v = np.array(r["per_start"][nm])
            row[nm] = {"mean": float(v.mean()), "sem": float(v.std(ddof=1) / np.sqrt(len(v))), "sd": float(v.std(ddof=1))}
        res[key] = row
        print(f"{key:<10} " + " ".join(f"{row[nm]['mean']:+.5f}+-{row[nm]['sem']:.5f}" for nm in ("rho", "sigma", "beta")))
    gates = {}
    for nm in ("rho", "sigma", "beta"):
        a, b = res["T160_a10"][nm], res["T320_a10"][nm]
        mn, sn = NILSS[nm]
        mo, so = ORACLE[nm]
        d33 = b["mean"] - mn
        lim33 = 2 * np.hypot(b["sem"], sn)
        d34 = abs(a["mean"] - b["mean"])
        lim34 = 2 * np.hypot(a["sem"], b["sem"])
        gates[nm] = {"A3.3_diff_to_nilss": float(d33), "A3.3_rel": float(d33 / abs(mn)), "A3.3_limit": float(lim33), "A3.3_pass": bool(abs(d33) <= lim33),
                     "diff_to_oracle": float(b["mean"] - mo), "rel_to_oracle": float((b["mean"] - mo) / abs(mo)),
                     "A3.4_diff": float(d34), "A3.4_limit": float(lim34), "A3.4_pass": bool(d34 <= lim34)}
        g = gates[nm]
        print(f"{nm}: T320 LSS {b['mean']:+.5f} | nilss {mn:+.5f} diff {g['A3.3_diff_to_nilss']:+.5f} ({100*g['A3.3_rel']:+.2f}%) limit {g['A3.3_limit']:.5f} A3.3 {'pass' if g['A3.3_pass'] else 'FAIL'}"
              f" | oracle {mo:+.5f} diff {100*g['rel_to_oracle']:+.2f}% | T160 vs T320 {g['A3.4_diff']:.5f} limit {g['A3.4_limit']:.5f} A3.4 {'pass' if g['A3.4_pass'] else 'FAIL'}")
    (OUT / "phaseA3_lss.json").write_text(json.dumps({"cases": res, "gates": gates}, indent=2) + "\n")
