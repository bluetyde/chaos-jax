"""A4: calibrate the nilss-jax reliability report against finite-difference errors. Design in docs/PREDICTIONS-A4.md.

python experiments/phaseA4_calibration.py oracle <case>     adaptive finite-difference oracle for one case (1..13)
python experiments/phaseA4_calibration.py nilss <case>      4 ensembles of 16 orbits (+ one nus+1 ensemble) for one case
python experiments/phaseA4_calibration.py controls          C-A4.1 and C-A4.3
python experiments/phaseA4_calibration.py report            table and gates
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
from chaos_jax.systems import lorenz, lorenz96, lorenz_params, lorenz96_params, rossler, rossler_params

PIN = "ad5b635f5798d89f5d31afa4118d31b778ab86f7"
OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)


def z_of(x):
    return x[2]


def mean_sq(x):
    return jnp.mean(x ** 2)


# ---- case table (docs/PREDICTIONS-A4.md) ----
# sys, param name, theta0, other system settings, nilss settings, oracle spec
L63 = dict(nilss=dict(dt=0.005, T_seg=0.5, nus=1, T=1000.0, T_spinup=20.0), oracle=dict(n_spin=5000, n_avg=10000, dt=0.01))
L96 = dict(nilss=dict(dt=0.01, T_seg=0.5, T=500.0, T_spinup=50.0), oracle=dict(n_spin=5000, n_avg=10000, dt=0.01))
ROS = dict(nilss=dict(dt=0.01, T_seg=1.0, nus=1, T=1000.0, T_spinup=100.0), oracle=dict(n_spin=10000, n_avg=40000, dt=0.01))
CASES = {
    1: dict(sys="L63", name="rho", rho=28.0, hs=(1.0, 0.5)), 2: dict(sys="L63", name="rho", rho=45.0, hs=(1.0, 0.5)),
    3: dict(sys="L63", name="rho", rho=50.0, hs=(1.0, 0.5)), 4: dict(sys="L63", name="rho", rho=60.0, hs=(1.0, 0.5)),
    5: dict(sys="L63", name="sigma", rho=28.0, hs=(1.0, 0.5)), 6: dict(sys="L63", name="sigma", rho=45.0, hs=(1.0, 0.5)),
    7: dict(sys="L63", name="beta", rho=28.0, hs=(0.25, 0.125)),
    8: dict(sys="L96", name="F", N=8, F=6.0, nus=2, hs=(0.5, 0.25)), 9: dict(sys="L96", name="F", N=8, F=10.0, nus=2, hs=(0.5, 0.25)),
    10: dict(sys="L96", name="F", N=16, F=8.0, nus=5, hs=(0.5, 0.25)),
    11: dict(sys="ROS", name="c", c=5.0, hs=(0.2, 0.1)), 12: dict(sys="ROS", name="c", c=7.0, hs=(0.2, 0.1)),
    13: dict(sys="ROS", name="c", c=9.0, hs=(0.2, 0.1)),
}
BASE_SIGMA, BASE_BETA = 10.0, 8.0 / 3.0


def spec(case):
    c = CASES[case]
    S = {"L63": L63, "L96": L96, "ROS": ROS}[c["sys"]]
    nil = dict(S["nilss"])
    if c["sys"] == "L96":
        nil["nus"] = c["nus"]
    return c, nil, S["oracle"]


def oracle_system(c):
    """f, observable, parameter array, slot of the differentiated parameter, start-point generator."""
    if c["sys"] == "L63":
        slot = {"sigma": 0, "rho": 1, "beta": 2}[c["name"]]
        p = lorenz_params(rho=c["rho"])
        def gen(key, n):
            k = jax.random.split(key, 3)
            return jnp.stack([jax.random.uniform(k[0], (n,), minval=-15.0, maxval=15.0), jax.random.uniform(k[1], (n,), minval=-15.0, maxval=15.0),
                              jax.random.uniform(k[2], (n,), minval=5.0, maxval=c["rho"] + 20.0)], axis=1)
        return lorenz, z_of, p, slot, gen
    if c["sys"] == "L96":
        def gen(key, n):
            return c["F"] + 0.5 * jax.random.normal(key, (n, c["N"]))
        return lorenz96, mean_sq, lorenz96_params(c["F"]), 0, gen
    def gen(key, n):
        k = jax.random.split(key, 3)
        return jnp.stack([jax.random.uniform(k[0], (n,), minval=-8.0, maxval=8.0), jax.random.uniform(k[1], (n,), minval=-8.0, maxval=8.0),
                          jax.random.uniform(k[2], (n,), minval=0.0, maxval=10.0)], axis=1)
    return rossler, z_of, rossler_params(c=c["c"]), 2, gen


def nilss_system(c):
    from nilss_jax.systems import lorenz63
    if c["sys"] == "L63":
        p = dict(lorenz63.DEFAULTS)
        p["rho"] = c["rho"]
        return lorenz63.rhs, lorenz63.J, p
    if c["sys"] == "L96":
        return l96_rhs, l96_J, {"F": c["F"]}
    return ros_rhs, ros_J, {"a": 0.2, "b": 0.2, "c": c["c"]}


def l96_rhs(u, p):
    return (jnp.roll(u, -1) - jnp.roll(u, 2)) * jnp.roll(u, 1) - u + p["F"]


def l96_J(u):
    return jnp.mean(u ** 2)


def ros_rhs(u, p):
    return jnp.array([-u[1] - u[2], u[0] + p["a"] * u[1], p["b"] + u[2] * (u[0] - p["c"])])


def ros_J(u):
    return u[2]


def boot_slope(d, h, rng, n_boot=1000):
    n = d.shape[0]
    means = d[rng.integers(0, n, (n_boot, n))].mean(axis=1)
    s = (means[:, 1] - means[:, 0]) / (2 * h)
    m = d.mean(axis=0)
    return float((m[1] - m[0]) / (2 * h)), float(s.std(ddof=1))


mode = sys.argv[1]
if mode == "oracle":
    case = int(sys.argv[2])
    c, nil, osp = spec(case)
    f, obs, p0, slot, gen = oracle_system(c)
    theta0 = float(p0[slot])
    rng = np.random.default_rng(1000 + case)
    t0 = time.time()
    data = {h: [] for h in c["hs"]}
    N, rnd = 0, 0
    while True:
        key = jax.random.fold_in(jax.random.PRNGKey(100 + case), rnd)
        x0s = gen(jax.random.fold_in(key, 0), 4000)
        for h in c["hs"]:
            cols = []
            for s in (-1.0, 1.0):
                cols.append(jnp.concatenate([ensemble_time_average(f, obs, x0s[i:i + 2000], p0.at[slot].set(theta0 + s * h), osp["dt"], osp["n_spin"], osp["n_avg"])
                                             for i in range(0, 4000, 2000)]))
            data[h].append(np.asarray(jnp.stack(cols, axis=1)))
        N += 4000
        rnd += 1
        d_small = np.concatenate(data[c["hs"][1]])
        s, se = boot_slope(d_small, c["hs"][1], rng)
        print(f"case {case} N={N} slope(h={c['hs'][1]})={s:+.5f} +- {se:.5f} t={time.time()-t0:.0f}s", flush=True)
        if se < 0.05 * abs(s) or N >= 64000:
            break
    est = {}
    for h in c["hs"]:
        est[h] = boot_slope(np.concatenate(data[h]), h, rng)
    (h_big, h_small) = c["hs"]
    diff = abs(est[h_big][0] - est[h_small][0])
    lim = 2 * np.hypot(est[h_big][1], est[h_small][1])
    res = {"case": case, "spec": c, "N": N, "slope": est[h_small][0], "se": est[h_small][1], "slope_big_h": est[h_big][0], "se_big_h": est[h_big][1],
           "h_pair_diff": diff, "h_pair_limit": lim, "h_converged": bool(diff <= lim), "se_rule_met": bool(est[h_small][1] < 0.05 * abs(est[h_small][0])),
           "oracle_converged": bool(diff <= lim and est[h_small][1] < 0.05 * abs(est[h_small][0])), "wall_seconds": time.time() - t0}
    (OUT / f"phaseA4_oracle_{case}.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
    print({k: v for k, v in res.items() if k != "spec"})

elif mode == "nilss":
    from nilss_jax import NILSS, run_ensemble
    case = int(sys.argv[2])
    c, nil, osp = spec(case)
    rhs, J, p = nilss_system(c)
    f, obs, p0, slot, gen = oracle_system(c)
    t0 = time.time()
    def make(nus):
        return NILSS(rhs, J, params=(c["name"],), dt=nil["dt"], T_seg=nil["T_seg"], nus=nus)
    ens_out = []
    for e in range(4):
        u0s = np.asarray(gen(jax.random.fold_in(jax.random.PRNGKey(200 + case), e), 16))
        ens = run_ensemble(make(nil["nus"]), p, u0s, T=nil["T"], T_spinup=nil["T_spinup"])
        vals = np.asarray(ens.values())[:, 0]
        lyap = np.array([float(np.max(r.lyapunov)) for r in ens.results])
        row = {"mean": float(ens.mean()[c["name"]]), "sem": float(ens.sem()[c["name"]]), "median": float(np.median(vals)),
               "n_used": int(np.sum(ens.kept)), "verdict": str(ens.report().verdict), "lyap1_mean": float(lyap.mean()),
               "J_mean": float(np.mean([float(r.J) for r in ens.results])), "values": vals.tolist()}
        if e == 0:
            ens_plus = run_ensemble(make(nil["nus"] + 1), p, u0s, T=nil["T"], T_spinup=nil["T_spinup"])
            row["verdict_nus_check"] = str(ens.report(nus_check=ens_plus).verdict)
        ens_out.append(row)
        print(f"case {case} ensemble {e}: mean {row['mean']:+.5f} +- {row['sem']:.5f} verdict {row['verdict']} used {row['n_used']}/16 t={time.time()-t0:.0f}s", flush=True)
    (OUT / f"phaseA4_nilss_{case}.json").write_text(json.dumps({"pin": PIN, "case": case, "spec": c, "settings": nil, "ensembles": ens_out,
                                                                 "wall_seconds": time.time() - t0}, indent=2, default=str) + "\n")

elif mode == "controls":
    n1 = json.loads((OUT / "phaseA4_nilss_1.json").read_text())
    l1 = np.mean([e["lyap1_mean"] for e in n1["ensembles"]])
    j1 = np.mean([e["J_mean"] for e in n1["ensembles"]])
    print(f"C-A4.1 case 1: lambda_1 {l1:.4f} (gate within 0.02 of 0.9041): {'pass' if abs(l1 - 0.9041) <= 0.02 else 'FAIL'};  mean z {j1:.3f} (within 0.2 of 23.55): {'pass' if abs(j1 - 23.55) <= 0.2 else 'FAIL'}")
    o1 = json.loads((OUT / "phaseA4_oracle_1.json").read_text())
    print(f"C-A4.3 case 1 oracle at h=0.5: {o1['slope']:+.5f} +- {o1['se']:.5f}  (A1: 1.003 +- 0.002): {'pass' if abs(o1['slope'] - 1.003) <= 2 * np.hypot(o1['se'], 0.002) else 'FAIL'}")

elif mode == "report":
    rows, table = [], {}
    labels = {v: k for k, v in {"ok": "ok", "caution": "caution", "unreliable": "unreliable"}.items()}
    ok_big, ok_total, misses, false_alarms, stab = 0, 0, [], [], []
    print(f"{'case':>4} {'system':<22} {'param':<6} {'oracle':>10} {'conv':>5} | verdicts (4 ens)            | rel. errors (%)                | nus-check")
    for case in range(1, 14):
        o = json.loads((OUT / f"phaseA4_oracle_{case}.json").read_text())
        n = json.loads((OUT / f"phaseA4_nilss_{case}.json").read_text())
        c = CASES[case]
        name = {"L63": f"Lorenz-63 rho={c.get('rho')}", "L96": f"Lorenz-96 N={c.get('N')} F={c.get('F')}", "ROS": f"Roessler c={c.get('c')}"}[c["sys"]]
        vs = [e["verdict"] for e in n["ensembles"]]
        errs = [abs(e["mean"] - o["slope"]) / abs(o["slope"]) for e in n["ensembles"]]
        stab.append(len(set(vs)) == 1)
        for v, er in zip(vs, errs):
            b = "<5%" if er < 0.05 else ("5-25%" if er <= 0.25 else ">25%")
            table.setdefault((v, b), []).append(case)
            if o["oracle_converged"]:
                ok_total += v == "ok"
                if v == "ok" and er > 0.25:
                    ok_big += 1
                if v == "ok" and er > 0.10:
                    misses.append((case, er))
                if v == "unreliable" and er < 0.05:
                    false_alarms.append((case, er))
        print(f"{case:>4} {name:<22} {c['name']:<6} {o['slope']:>+10.4f} {'yes' if o['oracle_converged'] else 'NO':>5} | {','.join(vs):<28}| {', '.join(f'{100*x:6.1f}' for x in errs):<31}| {n['ensembles'][0].get('verdict_nus_check')}")
        rows.append({"case": case, "name": name, "param": c["name"], "oracle": o["slope"], "oracle_se": o["se"], "oracle_converged": o["oracle_converged"],
                     "h_converged": o["h_converged"], "se_rule_met": o["se_rule_met"], "verdicts": vs, "rel_errors": errs,
                     "means": [e["mean"] for e in n["ensembles"]], "verdict_nus_check": n["ensembles"][0].get("verdict_nus_check")})
    print("\nverdict x error bin (counts of ensembles, all cases):")
    for v in ("ok", "caution", "unreliable"):
        print(f"  {v:<11}", {b: len(table.get((v, b), [])) for b in ("<5%", "5-25%", ">25%")})
    res = {"rows": rows, "A4.1_confident_misses": ok_big, "A4.1_pass": ok_big == 0, "ok_ensembles_in_converged_cases": int(ok_total),
           "A4.2_tested": bool(ok_total >= 6), "misses_ok_err_gt_10pct": misses, "false_alarms_unreliable_err_lt_5pct": false_alarms,
           "cases_with_stable_verdict": int(sum(stab)), "oracle_converged_cases": int(sum(r["oracle_converged"] for r in rows))}
    print({k: v for k, v in res.items() if k != "rows"})
    (OUT / "phaseA4_calibration.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
