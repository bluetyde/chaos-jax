"""A2: run nilss-jax (pinned commit, see docs/PREDICTIONS-A2.md) on Lorenz-63 for rho, sigma, beta.

python experiments/phaseA2_run.py probe     wall time of one orbit of T = 100 (prints only the time)
python experiments/phaseA2_run.py stage1    README quickstart, 8 orbits, T = 200
python experiments/phaseA2_run.py stage2    gate run, 32 orbits, T = 1000
python experiments/phaseA2_run.py controls  C1 and C2 only, no sensitivities
python experiments/phaseA2_run.py report    controls and gates
"""
import json
import pathlib
import sys
import time

import numpy as np

import jax

PIN = "ad5b635f5798d89f5d31afa4118d31b778ab86f7"
OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
PARAMS = ("rho", "sigma", "beta")
KW = dict(dt=0.005, T_seg=0.5, nus=1)


def make():
    from nilss_jax import NILSS
    from nilss_jax.systems import lorenz63
    return NILSS(lorenz63.rhs, lorenz63.J, params=PARAMS, **KW), lorenz63


def start_points(n):
    k = jax.random.split(jax.random.PRNGKey(70), 3)
    return np.stack([np.asarray(jax.random.uniform(k[0], (n,), minval=-15.0, maxval=15.0)),
                     np.asarray(jax.random.uniform(k[1], (n,), minval=-15.0, maxval=15.0)),
                     np.asarray(jax.random.uniform(k[2], (n,), minval=5.0, maxval=45.0))], axis=1)


def summarise(ens, extra):
    vals = np.asarray(ens.values())
    lyap = np.array([float(np.max(r.lyapunov)) for r in ens.results])
    J = np.array([float(r.J) for r in ens.results])
    sem = ens.sem()
    mean = ens.mean()
    out = {"pin": PIN, "settings": dict(KW), "n_orbits": len(ens.results), "n_used": int(np.sum(ens.kept)) if hasattr(ens, "kept") else None,
           "mean": {k: float(mean[k]) for k in PARAMS}, "sem": {k: float(sem[k]) for k in PARAMS},
           "per_orbit": {k: vals[:, i].tolist() for i, k in enumerate(PARAMS)},
           "lyap1_mean": float(lyap.mean()), "lyap1_all": lyap.tolist(), "J_mean": float(J.mean()), "J_all": J.tolist(),
           "verdict": str(ens.report().verdict)}
    out.update(extra)
    return out


mode = sys.argv[1]
if mode == "probe":
    nilss, L = make()
    u0 = start_points(1)[0]
    nilss.run(L.DEFAULTS, u0, T=20.0, T_spinup=20.0)          # compile
    t = time.time()
    r = nilss.run(L.DEFAULTS, u0, T=100.0, T_spinup=20.0)
    print(f"one orbit, T = 100, T_spinup = 20: {time.time() - t:.2f} s")
elif mode in ("stage1", "stage2"):
    from nilss_jax import run_ensemble
    nilss, L = make()
    t0 = time.time()
    if mode == "stage1":
        u0s = np.array([L.initial_condition(i) for i in range(8)])
        T = 200.0
    else:
        u0s = start_points(32)
        T = 1000.0
    ens = run_ensemble(nilss, L.DEFAULTS, u0s, T=T, T_spinup=20.0)
    res = summarise(ens, {"mode": mode, "T": T, "T_spinup": 20.0, "wall_seconds": time.time() - t0})
    (OUT / f"phaseA2_{mode}.json").write_text(json.dumps(res, indent=2) + "\n")
    print(mode, "done in", round(res["wall_seconds"]), "s; orbits", res["n_orbits"], "used", res["n_used"])
elif mode == "controls":
    for stage in ("stage1", "stage2"):
        p = OUT / f"phaseA2_{stage}.json"
        if p.exists():
            d = json.loads(p.read_text())
            print(stage, "C1 lyap1_mean", round(d["lyap1_mean"], 4), "pass", abs(d["lyap1_mean"] - 0.9041) <= 0.02,
                  "| C2 J_mean", round(d["J_mean"], 3), "pass", abs(d["J_mean"] - 23.55) <= 0.2,
                  "| orbits", d["n_orbits"], "used", d["n_used"])
elif mode == "report":
    oracle = json.loads((OUT / "phaseA1_oracle_v2.json").read_text())
    ref = {"rho": (oracle["rho"]["weighted_mean_all_rungs"], oracle["rho"]["weighted_mean_se"]),
           "sigma": (oracle["sigma"]["weighted_mean_all_rungs"], oracle["sigma"]["weighted_mean_se"]),
           "beta": (oracle["beta"]["final_slope"], oracle["beta"]["final_se"])}
    res = {}
    for stage in ("stage1", "stage2"):
        p = OUT / f"phaseA2_{stage}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        row = {"n_orbits": d["n_orbits"], "n_used": d["n_used"], "lyap1_mean": d["lyap1_mean"], "J_mean": d["J_mean"],
               "verdict": d["verdict"], "params": {}}
        row["C1_pass"] = bool(abs(d["lyap1_mean"] - 0.9041) <= 0.02)
        row["C2_pass"] = bool(abs(d["J_mean"] - 23.55) <= 0.2)
        for k in PARAMS:
            m, s = d["mean"][k], d["sem"][k]
            o, so = ref[k]
            lim = 2 * np.hypot(s, so)
            row["params"][k] = {"nilss": m, "sem": s, "oracle": o, "oracle_se": so, "diff": m - o, "rel_diff": (m - o) / abs(o),
                                "limit": float(lim), "A2.1_pass": bool(abs(m - o) <= lim)}
        res[stage] = row
        print(stage, {k: v for k, v in row.items() if k != "params"})
        for k, v in row["params"].items():
            print(f"   {k}: nilss {v['nilss']:+.5f} +- {v['sem']:.5f}  oracle {v['oracle']:+.5f} +- {v['oracle_se']:.5f}  diff {v['diff']:+.5f} ({100*v['rel_diff']:+.2f}%)  limit {v['limit']:.5f}  A2.1 {'pass' if v['A2.1_pass'] else 'FAIL'}")
    if "stage2" in res:
        d = json.loads((OUT / "phaseA2_stage2.json").read_text())
        m, s = d["mean"]["sigma"], d["sem"]["sigma"]
        res["A2.2a_reproduces_0.133"] = bool(abs(m - 0.133) <= 2 * np.hypot(s, 0.004))
        res["A2.2b_differs_from_oracle"] = bool(abs(m - ref["sigma"][0]) > 2 * np.hypot(s, ref["sigma"][1]))
        print("A2.2a", res["A2.2a_reproduces_0.133"], " A2.2b", res["A2.2b_differs_from_oracle"])
    (OUT / "phaseA2_report.json").write_text(json.dumps(res, indent=2) + "\n")
