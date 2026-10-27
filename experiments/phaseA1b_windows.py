"""A1b: does the finite-difference oracle depend on the averaging window, dt or spin-up? Design in docs/PREDICTIONS-A1b.md.

python experiments/phaseA1b_windows.py run rho|sigma|beta
python experiments/phaseA1b_windows.py control     gate C-W only
python experiments/phaseA1b_windows.py report      all variants, gates W1 to W3
"""
import json
import pathlib
import sys
import time

import numpy as np

import jax
import jax.numpy as jnp
import chaos_jax  # noqa: F401
from chaos_jax.response import ensemble_time_average
from chaos_jax.systems import lorenz, lorenz_params

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
CHUNK, N_BOOT = 16000, 2000
PARAMS = {"rho": (0, 1, 28.0, 0.5), "sigma": (1, 0, 10.0, 0.5), "beta": (2, 2, 8.0 / 3.0, 0.125)}  # index, lorenz_params slot, theta0, h
# name, T_avg, dt, T_spin, N
VARIANTS = [("baseline", 100.0, 0.01, 50.0, 8000), ("T25", 25.0, 0.01, 50.0, 32000), ("T400", 400.0, 0.01, 50.0, 2000),
            ("dt0.02", 100.0, 0.02, 50.0, 8000), ("dt0.005", 100.0, 0.005, 50.0, 8000),
            ("spin10", 100.0, 0.01, 10.0, 8000), ("spin200", 100.0, 0.01, 200.0, 8000)]
A1 = {"rho": (1.00305, 0.00150), "sigma": (0.14887, 0.00142), "beta": (-1.65030, 0.00581)}


def zobs(x):
    return x[2]


def start_points(key, n):
    k = jax.random.split(key, 3)
    return jnp.stack([jax.random.uniform(k[0], (n,), minval=-15.0, maxval=15.0),
                      jax.random.uniform(k[1], (n,), minval=-15.0, maxval=15.0),
                      jax.random.uniform(k[2], (n,), minval=5.0, maxval=45.0)], axis=1)


def slope_se(data, h, rng):
    n = data.shape[0]
    idx = rng.integers(0, n, (N_BOOT, n))
    means = data[idx].mean(axis=1)
    s = (means[:, 1] - means[:, 0]) / (2 * h)
    m = data.mean(axis=0)
    return float((m[1] - m[0]) / (2 * h)), float(s.std(ddof=1))


mode = sys.argv[1]
if mode == "run":
    name = sys.argv[2]
    pidx, slot, theta0, h = PARAMS[name]
    t0 = time.time()
    for vi, (vname, T, dt, Tspin, N) in enumerate(VARIANTS):
        x0s = start_points(jax.random.fold_in(jax.random.PRNGKey(80), 10 * pidx + vi), N)
        cols = []
        for s in (-1.0, 1.0):
            p = lorenz_params().at[slot].set(theta0 + s * h)
            cols.append(jnp.concatenate([ensemble_time_average(lorenz, zobs, x0s[i:i + CHUNK], p, dt,
                                                               int(round(Tspin / dt)), int(round(T / dt)))
                                         for i in range(0, N, CHUNK)]))
        np.save(OUT / f"phaseA1b_{name}_{vname}.npy", np.asarray(jnp.stack(cols, axis=1)))
        print(f"{name} {vname} done t={time.time()-t0:.0f}s", flush=True)

elif mode in ("control", "report"):
    rng = np.random.default_rng(81)
    res, ok = {}, True
    for name, (pidx, slot, theta0, h) in PARAMS.items():
        rows = {}
        for vname, T, dt, Tspin, N in VARIANTS:
            s, se = slope_se(np.load(OUT / f"phaseA1b_{name}_{vname}.npy"), h, rng)
            rows[vname] = {"slope": s, "se": se, "T": T, "dt": dt, "T_spin": Tspin, "N": N}
        b = rows["baseline"]
        a, sa = A1[name]
        d = abs(b["slope"] - a) / np.hypot(b["se"], sa)
        res[name] = {"rows": rows, "C-W_diff_in_SE": float(d), "C-W_pass": bool(d <= 2)}
        ok &= d <= 2
        if mode == "control":
            print(f"C-W {name}: baseline {b['slope']:+.5f} +- {b['se']:.5f}  A1 rung {a:+.5f} +- {sa:.5f}  diff {d:.2f} SE  {'pass' if d <= 2 else 'FAIL'}")
    if mode == "report":
        for name, r in res.items():
            b = r["rows"]["baseline"]
            print(f"== {name}")
            for vname, v in r["rows"].items():
                z = abs(v["slope"] - b["slope"]) / np.hypot(v["se"], b["se"]) if vname != "baseline" else 0.0
                v["diff_in_SE_vs_baseline"] = float(z)
                print(f"  {vname:<9} slope {v['slope']:+.5f} +- {v['se']:.5f}   vs baseline {z:4.2f} SE")
        rs = [v["diff_in_SE_vs_baseline"] for n in ("rho", "sigma") for k, v in res[n]["rows"].items() if k != "baseline"]
        allz = [v["diff_in_SE_vs_baseline"] for n in res for k, v in res[n]["rows"].items() if k != "baseline"]
        res["W2_window_dependence_found"] = bool(max(rs) >= 3)
        res["n_beyond_2SE_of_18"] = int(sum(z > 2 for z in allz))
        res["n_beyond_3SE_of_18"] = int(sum(z > 3 for z in allz))
        tgt = {"rho": 1.017, "sigma": 0.134}
        res["W3_toward_shadowing"] = bool(any(abs(v["slope"] - tgt[n]) <= 2 * np.hypot(v["se"], 0.0001)
                                              for n in tgt for k, v in res[n]["rows"].items()))
        print({k: v for k, v in res.items() if k.startswith(("W", "n_"))})
        (OUT / "phaseA1b_windows.json").write_text(json.dumps(res, indent=2) + "\n")
