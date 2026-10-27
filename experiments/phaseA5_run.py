"""A5: our own NILSS on Lorenz-63, 32 orbits, compared with nilss-jax (A2). Design in docs/PREDICTIONS-A5.md.

python experiments/phaseA5_run.py probe     wall time of one orbit
python experiments/phaseA5_run.py run       32 orbits
python experiments/phaseA5_run.py control   C-A5.5
python experiments/phaseA5_run.py report    gates
"""
import json
import pathlib
import sys
import time

import numpy as np

import jax
import jax.numpy as jnp
import chaos_jax  # noqa: F401
from chaos_jax.integrate import integrate_final
from chaos_jax.nilss import nilss
from chaos_jax.systems import lorenz, lorenz_params

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
DT, T_SEG, M, T, T_SPIN = 0.005, 0.5, 1, 1000.0, 20.0
STEPS, K = int(round(T_SEG / DT)), int(round(T / T_SEG))
IDXS = (1, 0, 2)  # rho, sigma, beta, the order of A2
NAMES = ("rho", "sigma", "beta")
P0 = lorenz_params()


def z_of(x):
    return x[2]


def start_points(n):
    k = jax.random.split(jax.random.PRNGKey(70), 3)  # the A2 start points
    return jnp.stack([jax.random.uniform(k[0], (n,), minval=-15.0, maxval=15.0), jax.random.uniform(k[1], (n,), minval=-15.0, maxval=15.0),
                      jax.random.uniform(k[2], (n,), minval=5.0, maxval=45.0)], axis=1)


def one_orbit(x0, i):
    u0 = integrate_final(lorenz, x0, P0, DT, int(round(T_SPIN / DT)))
    return nilss(lorenz, z_of, u0, P0, IDXS, DT, STEPS, K, M, jax.random.PRNGKey(7000 + i))


mode = sys.argv[1]
if mode == "probe":
    x0s = start_points(32)
    jax.block_until_ready(one_orbit(x0s[0], 0)["dJds"])
    t = time.time()
    jax.block_until_ready(one_orbit(x0s[1], 1)["dJds"])
    print(f"one orbit (T = {T:g}, dt = {DT}, {K} segments): {time.time() - t:.2f} s")
elif mode == "run":
    x0s = start_points(32)
    t0 = time.time()
    rows = []
    for i in range(32):
        r = one_orbit(x0s[i], i)
        rows.append({"dJds": [float(v) for v in r["dJds"]], "lyap1": float(r["lyap"][0]), "J": float(r["J"])})
    vals = np.array([r["dJds"] for r in rows])
    res = {"settings": dict(dt=DT, T_seg=T_SEG, M=M, T=T, T_spinup=T_SPIN), "names": list(NAMES), "per_orbit": {n: vals[:, j].tolist() for j, n in enumerate(NAMES)},
           "mean": {n: float(vals[:, j].mean()) for j, n in enumerate(NAMES)}, "sem": {n: float(vals[:, j].std(ddof=1) / np.sqrt(32)) for j, n in enumerate(NAMES)},
           "lyap1_mean": float(np.mean([r["lyap1"] for r in rows])), "J_mean": float(np.mean([r["J"] for r in rows])), "wall_seconds": time.time() - t0}
    (OUT / "phaseA5_own.json").write_text(json.dumps(res, indent=2) + "\n")
    print("done in", round(res["wall_seconds"]), "s")
elif mode == "control":
    d = json.loads((OUT / "phaseA5_own.json").read_text())
    print(f"C-A5.5: lyap1 {d['lyap1_mean']:.4f} (within 0.02 of 0.9041): {'pass' if abs(d['lyap1_mean'] - 0.9041) <= 0.02 else 'FAIL'};  mean z {d['J_mean']:.3f} (within 0.2 of 23.55): {'pass' if abs(d['J_mean'] - 23.55) <= 0.2 else 'FAIL'}")
elif mode == "report":
    own = json.loads((OUT / "phaseA5_own.json").read_text())
    a2 = json.loads((OUT / "phaseA2_stage2.json").read_text())
    lss = json.loads((OUT / "phaseA3_lss.json").read_text())["cases"]["T320_a10"]
    orc = json.loads((OUT / "phaseA1_oracle_v2.json").read_text())
    oracle = {"rho": orc["rho"]["weighted_mean_all_rungs"], "sigma": orc["sigma"]["weighted_mean_all_rungs"], "beta": orc["beta"]["final_slope"]}
    res = {}
    for n in NAMES:
        m, s = own["mean"][n], own["sem"][n]
        mn, sn = a2["mean"][n], a2["sem"][n]
        pa = np.array(own["per_orbit"][n]); pb = np.array(a2["per_orbit"][n])
        row = {"own": m, "own_sem": s, "own_sd_orbits": float(pa.std(ddof=1)), "nilss_jax": mn, "nilss_jax_sem": sn,
               "rel_diff_to_nilss_jax": (m - mn) / abs(mn), "A5.1a_pass": bool(abs(m - mn) <= 2 * np.hypot(s, sn)), "A5.1b_pass": bool(abs(m - mn) / abs(mn) < 0.005),
               "median_per_orbit_rel_diff": float(np.median(np.abs(pa - pb) / np.abs(pb))),
               "rel_diff_to_lss_T320": (m - lss[n]["mean"]) / abs(lss[n]["mean"]), "A5.3_pass": bool(abs(m - lss[n]["mean"]) / abs(lss[n]["mean"]) < 0.01),
               "rel_diff_to_oracle": (m - oracle[n]) / abs(oracle[n])}
        row["A5.2_pass"] = bool(row["median_per_orbit_rel_diff"] < 0.01) if n in ("rho", "sigma") else None
        res[n] = row
        print(f"{n:<6} own {m:+.5f} +- {s:.5f} (orbit sd {row['own_sd_orbits']:.5f}) | nilss-jax {mn:+.5f} +- {sn:.5f} | diff {100*row['rel_diff_to_nilss_jax']:+.3f}%  A5.1a {'pass' if row['A5.1a_pass'] else 'FAIL'}  A5.1b {'pass' if row['A5.1b_pass'] else 'FAIL'}"
              f" | per-orbit median {100*row['median_per_orbit_rel_diff']:.3f}%  A5.2 {row['A5.2_pass']} | vs LSS(T=320) {100*row['rel_diff_to_lss_T320']:+.3f}%  A5.3 {'pass' if row['A5.3_pass'] else 'FAIL'} | vs oracle {100*row['rel_diff_to_oracle']:+.2f}%")
    (OUT / "phaseA5_report.json").write_text(json.dumps(res, indent=2) + "\n")
