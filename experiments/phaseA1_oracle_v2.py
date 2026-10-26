"""A1: finite-difference oracle v2 for d<z>/d(theta), theta in (rho, sigma, beta). Gates in docs/PREDICTIONS-A1.md.

python experiments/phaseA1_oracle_v2.py run rho|sigma|beta     all four rungs for one parameter
python experiments/phaseA1_oracle_v2.py report                 slopes, curvatures, gates
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
DT, N_SPIN, N_AVG, CHUNK = 0.01, 5000, 10000, 16000
PARAMS = {"rho": (0, 28.0, [1.0, 0.5, 0.25, 0.125]),
          "sigma": (1, 10.0, [1.0, 0.5, 0.25, 0.125]),
          "beta": (2, 8.0 / 3.0, [0.25, 0.125, 0.0625, 0.03125])}
NAMES = {"rho": "rho", "sigma": "sigma", "beta": "beta"}
THETA_INDEX = {"sigma": 0, "rho": 1, "beta": 2}  # position in lorenz_params
N0, N_BOOT = 2000, 2000


def zobs(x):
    return x[2]


def start_points(key, n):
    k = jax.random.split(key, 3)
    return jnp.stack([jax.random.uniform(k[0], (n,), minval=-15.0, maxval=15.0),
                      jax.random.uniform(k[1], (n,), minval=-15.0, maxval=15.0),
                      jax.random.uniform(k[2], (n,), minval=5.0, maxval=45.0)], axis=1)


def run_means(theta_name, value, x0s):
    p = lorenz_params().at[THETA_INDEX[theta_name]].set(value)
    return jnp.concatenate([ensemble_time_average(lorenz, zobs, x0s[i:i + CHUNK], p, DT, N_SPIN, N_AVG)
                            for i in range(0, x0s.shape[0], CHUNK)])


mode = sys.argv[1]
if mode == "run":
    name = sys.argv[2]
    pidx, theta0, hs = PARAMS[name]
    t0 = time.time()
    for k, h in enumerate(hs):
        n = N0 * 4 ** k
        x0s = start_points(jax.random.fold_in(jax.random.PRNGKey(60), 10 * pidx + k), n)
        cols = [run_means(name, theta0 + s * h, x0s) for s in (-1.0, 0.0, 1.0)]
        data = np.asarray(jnp.stack(cols, axis=1))
        np.save(OUT / f"phaseA1_{name}_k{k}.npy", data)
        print(f"{name} k={k} h={h} N={n} done t={time.time()-t0:.0f}s", flush=True)

elif mode == "report":
    rng = np.random.default_rng(61)
    res = {}
    for name, (pidx, theta0, hs) in PARAMS.items():
        rows = []
        for k, h in enumerate(hs):
            d = np.load(OUT / f"phaseA1_{name}_k{k}.npy")
            n = d.shape[0]
            slope = lambda m: (m[:, 2] - m[:, 0]) / (2 * h)
            curv = lambda m: (m[:, 2] + m[:, 0] - 2 * m[:, 1]) / h ** 2
            idx = rng.integers(0, n, (N_BOOT, n))
            means = d[idx].mean(axis=1)  # (boot, 3)
            s_b, c_b = slope(means), curv(means)
            m = d.mean(axis=0)[None]
            rows.append({"k": k, "h": h, "N": n, "slope": float(slope(m)[0]), "slope_se": float(s_b.std(ddof=1)),
                         "curvature": float(curv(m)[0]), "curvature_se": float(c_b.std(ddof=1)),
                         "mean_z_center": float(m[0, 1])})
        pairs = []
        for a, b in zip(rows[:-1], rows[1:]):
            diff = abs(a["slope"] - b["slope"])
            lim = 2 * np.hypot(a["slope_se"], b["slope_se"])
            pairs.append({"h_pair": [a["h"], b["h"]], "diff": float(diff), "limit": float(lim), "pass": bool(diff <= lim)})
        w = np.array([1 / r["slope_se"] ** 2 for r in rows])
        wmean = float(np.sum(w * np.array([r["slope"] for r in rows])) / w.sum())
        last = rows[-1]
        res[name] = {"theta0": theta0, "rungs": rows, "pairs": pairs, "converged": bool(all(p["pass"] for p in pairs)),
                     "final_slope": last["slope"], "final_se": last["slope_se"],
                     "weighted_mean_all_rungs": wmean, "weighted_mean_se": float(1 / np.sqrt(w.sum())),
                     "A1.2_pass": bool(last["slope_se"] < 0.05 * abs(last["slope"]))}
        print(f"== {name} (theta0 = {theta0:.4g})")
        for r in rows:
            print(f"  h={r['h']:<8} N={r['N']:<7} slope={r['slope']:+.5f} +- {r['slope_se']:.5f}   curvature={r['curvature']:+.4f} +- {r['curvature_se']:.4f}")
        for p in pairs:
            print(f"  pair {p['h_pair']}: diff {p['diff']:.5f} limit {p['limit']:.5f} {'ok' if p['pass'] else 'FAIL'}")
        print(f"  converged={res[name]['converged']}  final={last['slope']:+.5f} +- {last['slope_se']:.5f}  weighted mean={wmean:+.5f} +- {res[name]['weighted_mean_se']:.5f}  A1.2={res[name]['A1.2_pass']}")
    (OUT / "phaseA1_oracle_v2.json").write_text(json.dumps(res, indent=2) + "\n")
