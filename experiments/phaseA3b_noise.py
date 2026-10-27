"""A3b: noise-regularised finite differences for rho and sigma. Design in docs/PREDICTIONS-A3b.md.

python experiments/phaseA3b_noise.py run rho|sigma     all noise levels and both steps for one parameter
python experiments/phaseA3b_noise.py control           C-N2 only (eps = 0 anchor against A1)
python experiments/phaseA3b_noise.py report            gates N1 to N3
"""
import json
import pathlib
import sys
import time

import numpy as np

import jax
import jax.numpy as jnp
import chaos_jax  # noqa: F401
from chaos_jax.noisy import ensemble_noisy_time_average
from chaos_jax.systems import lorenz, lorenz_params

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
DT, N_SPIN, N_AVG, CHUNK, N_BOOT = 0.01, 5000, 10000, 8000, 2000
PARAMS = {"rho": (0, 1, 28.0), "sigma": (1, 0, 10.0)}  # index, slot in lorenz_params, theta0
EPS = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0]
HS = [(1.0, 2000), (0.5, 8000)]
A1 = {"rho": {1.0: (1.00331, 0.00151), 0.5: (1.00305, 0.00150)}, "sigma": {1.0: (0.15145, 0.00138), 0.5: (0.14887, 0.00142)}}
F = {"rho": (1.00317, 0.00074), "sigma": (0.14882, 0.00072)}
SH = {"rho": (1.01708, 0.00007), "sigma": (0.13375, 0.00008)}


def zobs(x):
    return x[2]


def start_points(key, n):
    k = jax.random.split(key, 3)
    return jnp.stack([jax.random.uniform(k[0], (n,), minval=-15.0, maxval=15.0),
                      jax.random.uniform(k[1], (n,), minval=-15.0, maxval=15.0),
                      jax.random.uniform(k[2], (n,), minval=5.0, maxval=45.0)], axis=1)


def slope_se(d, h, rng):
    n = d.shape[0]
    means = d[rng.integers(0, n, (N_BOOT, n))].mean(axis=1)
    s = (means[:, 1] - means[:, 0]) / (2 * h)
    m = d.mean(axis=0)
    return float((m[1] - m[0]) / (2 * h)), float(s.std(ddof=1))


mode = sys.argv[1]
if mode == "run":
    name = sys.argv[2]
    pidx, slot, theta0 = PARAMS[name]
    t0 = time.time()
    for ei, eps in enumerate(EPS):
        for hi, (h, N) in enumerate(HS):
            base = jax.random.fold_in(jax.random.PRNGKey(95), 100 * pidx + 10 * ei + hi)
            x0s = start_points(jax.random.fold_in(base, 0), N)
            keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.fold_in(base, 1), i))(jnp.arange(N))
            cols = []
            for s in (-1.0, 1.0):
                p = lorenz_params().at[slot].set(theta0 + s * h)
                cols.append(jnp.concatenate([ensemble_noisy_time_average(lorenz, zobs, x0s[i:i + CHUNK], keys[i:i + CHUNK], p, DT, N_SPIN, N_AVG, eps)
                                             for i in range(0, N, CHUNK)]))
            np.save(OUT / f"phaseA3b_{name}_e{eps:g}_h{h:g}.npy", np.asarray(jnp.stack(cols, axis=1)))
            print(f"{name} eps={eps:g} h={h:g} N={N} done t={time.time()-t0:.0f}s", flush=True)

elif mode in ("control", "report"):
    rng = np.random.default_rng(96)
    res = {}
    for name in PARAMS:
        rows = {}
        for eps in EPS:
            for h, N in HS:
                s, se = slope_se(np.load(OUT / f"phaseA3b_{name}_e{eps:g}_h{h:g}.npy"), h, rng)
                rows.setdefault(f"{eps:g}", {})[f"h{h:g}"] = {"slope": s, "se": se, "N": N}
        res[name] = {"rows": rows}
        if mode == "control":
            for h, _ in HS:
                r = rows["0"][f"h{h:g}"]
                a, sa = A1[name][h]
                d = abs(r["slope"] - a) / np.hypot(r["se"], sa)
                print(f"C-N2 {name} h={h:g}: eps=0 {r['slope']:+.5f} +- {r['se']:.5f}  A1 {a:+.5f} +- {sa:.5f}  diff {d:.2f} SE  {'pass' if d <= 2 else 'FAIL'}")
    if mode == "report":
        for name in PARAMS:
            rows = res[name]["rows"]
            print(f"== {name}")
            conv = {}
            for eps in EPS:
                r1, r2 = rows[f"{eps:g}"]["h1"], rows[f"{eps:g}"]["h0.5"]
                lim = 2 * np.hypot(r1["se"], r2["se"])
                conv[f"{eps:g}"] = bool(abs(r1["slope"] - r2["slope"]) <= lim)
                print(f"  eps={eps:<6g} h=1 {r1['slope']:+.5f}+-{r1['se']:.5f}   h=0.5 {r2['slope']:+.5f}+-{r2['se']:.5f}   N1 {'ok' if conv[f'{eps:g}'] else 'FAIL'}")
            res[name]["N1_converged_in_h"] = conv
            S = rows["0.125"]["h0.5"]
            f, sf = F[name]
            sh, ssh = SH[name]
            near_f = abs(S["slope"] - f) <= 2 * np.hypot(S["se"], sf)
            near_sh = abs(S["slope"] - sh) <= 2 * np.hypot(S["se"], ssh)
            usable = conv["0.125"] and conv["0.25"]
            verdict = "unresolved"
            if usable and near_f and not near_sh:
                verdict = "finite-difference side"
            elif usable and near_sh and not near_f:
                verdict = "shadowing side"
            plateau = all(abs(rows[f"{e:g}"]["h0.5"]["slope"] - S["slope"]) <= 2 * np.hypot(rows[f"{e:g}"]["h0.5"]["se"], S["se"]) for e in (0.5, 0.25))
            res[name].update({"N2_S": S["slope"], "N2_se": S["se"], "N2_usable": bool(usable), "N2_near_FD": bool(near_f), "N2_near_shadowing": bool(near_sh),
                              "N2_verdict": verdict, "N3_plateau_eps_le_0.5": bool(plateau)})
            print(f"  N2: S = {S['slope']:+.5f} +- {S['se']:.5f}  usable={usable}  near FD {near_f}  near shadowing {near_sh}  -> {verdict};  N3 plateau (eps 0.5, 0.25, 0.125): {plateau}")
        (OUT / "phaseA3b_noise.json").write_text(json.dumps(res, indent=2) + "\n")
