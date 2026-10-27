"""A3 part 1: vary the nilss-jax settings one at a time. Design in docs/PREDICTIONS-A3.md.

python experiments/phaseA3_settings.py run         all variants (the baseline is the A2 stage 2 result)
python experiments/phaseA3_settings.py controls    C1 and C2 only
python experiments/phaseA3_settings.py report      gates A3.1 and A3.2
"""
import json
import pathlib
import sys
import time

import numpy as np

import jax

PIN = "ad5b635f5798d89f5d31afa4118d31b778ab86f7"
OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
PARAMS = ("rho", "sigma", "beta")
BASE = dict(dt=0.005, T_seg=0.5, nus=1, T=1000.0)
VARIANTS = {"dt0.01": dict(dt=0.01), "dt0.0025": dict(dt=0.0025), "seg0.25": dict(T_seg=0.25), "seg1.0": dict(T_seg=1.0),
            "nus2": dict(nus=2), "T250": dict(T=250.0), "T4000": dict(T=4000.0)}
BASE_MEAN = {"rho": 1.01708, "sigma": 0.13375, "beta": -1.65927}
THRESH = {"rho": 0.004, "sigma": 0.004, "beta": 0.02}
ORACLE_SIGMA, ORACLE_SIGMA_SE = 0.14882, 0.00072


def start_points(n):
    k = jax.random.split(jax.random.PRNGKey(70), 3)
    return np.stack([np.asarray(jax.random.uniform(k[0], (n,), minval=-15.0, maxval=15.0)),
                     np.asarray(jax.random.uniform(k[1], (n,), minval=-15.0, maxval=15.0)),
                     np.asarray(jax.random.uniform(k[2], (n,), minval=5.0, maxval=45.0))], axis=1)


mode = sys.argv[1]
if mode == "run":
    from nilss_jax import NILSS, run_ensemble
    from nilss_jax.systems import lorenz63
    u0s = start_points(32)
    for name, change in VARIANTS.items():
        cfg = {**BASE, **change}
        t0 = time.time()
        nilss = NILSS(lorenz63.rhs, lorenz63.J, params=PARAMS, dt=cfg["dt"], T_seg=cfg["T_seg"], nus=cfg["nus"])
        ens = run_ensemble(nilss, lorenz63.DEFAULTS, u0s, T=cfg["T"], T_spinup=20.0)
        mean, sem = ens.mean(), ens.sem()
        lyap = np.array([float(np.max(r.lyapunov)) for r in ens.results])
        J = np.array([float(r.J) for r in ens.results])
        res = {"pin": PIN, "variant": name, "settings": cfg, "n_orbits": len(ens.results), "n_used": int(np.sum(ens.kept)),
               "mean": {k: float(mean[k]) for k in PARAMS}, "sem": {k: float(sem[k]) for k in PARAMS},
               "lyap1_mean": float(lyap.mean()), "J_mean": float(J.mean()), "verdict": str(ens.report().verdict),
               "wall_seconds": time.time() - t0}
        (OUT / f"phaseA3_nilss_{name}.json").write_text(json.dumps(res, indent=2) + "\n")
        print(name, "done", round(res["wall_seconds"]), "s; used", res["n_used"], "of", res["n_orbits"], flush=True)
elif mode == "controls":
    for name in VARIANTS:
        d = json.loads((OUT / f"phaseA3_nilss_{name}.json").read_text())
        print(f"{name:<9} C1 lyap1 {d['lyap1_mean']:.4f} {'pass' if abs(d['lyap1_mean'] - 0.9041) <= 0.02 else 'FAIL'}"
              f" | C2 mean z {d['J_mean']:.3f} {'pass' if abs(d['J_mean'] - 23.55) <= 0.2 else 'FAIL'} | used {d['n_used']}/{d['n_orbits']}")
elif mode == "report":
    base = json.loads((OUT / "phaseA2_stage2.json").read_text())
    rows = {"baseline": {"mean": base["mean"], "sem": base["sem"], "verdict": base["verdict"], "n_used": base["n_used"], "settings": BASE}}
    for name in VARIANTS:
        d = json.loads((OUT / f"phaseA3_nilss_{name}.json").read_text())
        rows[name] = {"mean": d["mean"], "sem": d["sem"], "verdict": d["verdict"], "n_used": d["n_used"], "settings": d["settings"],
                      "wall_seconds": d["wall_seconds"]}
    ok31, ok32 = True, True
    print(f"{'variant':<9} {'rho':>18} {'sigma':>18} {'beta':>18}  verdict  used")
    for name, r in rows.items():
        m, s = r["mean"], r["sem"]
        r["A3.1_pass"] = all(abs(m[k] - BASE_MEAN[k]) <= THRESH[k] for k in PARAMS)
        r["sigma_gap_to_oracle_in_SE"] = (ORACLE_SIGMA - m["sigma"]) / np.hypot(s["sigma"], ORACLE_SIGMA_SE)
        r["A3.2_pass"] = bool(abs(r["sigma_gap_to_oracle_in_SE"]) > 2)
        if name != "baseline":
            ok31 &= r["A3.1_pass"]
            ok32 &= r["A3.2_pass"]
        print(f"{name:<9} {m['rho']:+.5f}+-{s['rho']:.5f} {m['sigma']:+.5f}+-{s['sigma']:.5f} {m['beta']:+.5f}+-{s['beta']:.5f}  {r['verdict']:<8} {r['n_used']}"
              f"  A3.1 {'pass' if r['A3.1_pass'] else 'FAIL'}  sigma gap {r['sigma_gap_to_oracle_in_SE']:.0f} SE")
    print("A3.1 all variants:", ok31, " A3.2 all variants:", ok32)
    (OUT / "phaseA3_settings.json").write_text(json.dumps({"rows": rows, "A3.1_all": bool(ok31), "A3.2_all": bool(ok32)}, indent=2, default=float) + "\n")
