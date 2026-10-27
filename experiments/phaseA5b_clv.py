"""A5b: hyperbolicity test for Lorenz-63. Design and decision rules in docs/PREDICTIONS-A5b.md.

python experiments/phaseA5b_clv.py angles    D1: angle distributions at rho = 28, two orbits of T = 10000
python experiments/phaseA5b_clv.py link      D2: per-segment contributions against angles, 32 orbits, T = 1000
python experiments/phaseA5b_clv.py scan      D3: rho in {26, 28, 30, 33, 36, 40}
python experiments/phaseA5b_clv.py report    summary
"""
import json
import pathlib
import sys
import time

import numpy as np
from scipy.stats import spearmanr

import jax
import jax.numpy as jnp
import chaos_jax  # noqa: F401
from chaos_jax.clv import covariant_vectors, line_angle, vector_plane_angle
from chaos_jax.integrate import integrate_final
from chaos_jax.nilss import nilss
from chaos_jax.response import ensemble_time_average
from chaos_jax.systems import lorenz, lorenz_params

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
DT, SPB = 0.005, 20            # block = 0.1 time units
FWD_BLOCKS, BACK_BLOCKS = 500, 1000   # discarded: 50 and 100 time units
GAP_RHO, ORACLE_RHO, OWN_RHO = 0.01384, 1.00317, 1.01701


def z_of(x):
    return x[2]


def angles(xs, V, p):
    f = jax.vmap(lambda x: lorenz(x, p))(xs)
    z1, z3 = V[:, :, 0], V[:, :, 2]
    return {"theta1": line_angle(z1, z3), "theta2": vector_plane_angle(z3, z1, f), "theta3": line_angle(z1, f), "fnorm": jnp.linalg.norm(f, axis=-1)}


def clv_orbit(x0, p, T, key):
    n_blocks = int(round(T / (DT * SPB)))
    xs, V, growth, Rs = covariant_vectors(lorenz, x0, p, DT, n_blocks, SPB, key)
    return xs, V


def summarise(a):
    out = {}
    for k in ("theta1", "theta2", "theta3"):
        v = np.asarray(a[k])
        out[k] = {"min": float(v.min()), "q0.1%": float(np.quantile(v, 0.001)), "q1%": float(np.quantile(v, 0.01)), "q5%": float(np.quantile(v, 0.05)),
                  "median": float(np.median(v)), "P<0.01": float((v < 0.01).mean()), "P<0.05": float((v < 0.05).mean()), "P<0.1": float((v < 0.1).mean()), "n": int(v.size)}
    fn = np.asarray(a["fnorm"])
    out["fnorm"] = {"min": float(fn.min()), "q1%": float(np.quantile(fn, 0.01)), "median": float(np.median(fn))}
    return out


def start_point(key, rho):
    k = jax.random.split(key, 3)
    return jnp.stack([jax.random.uniform(k[0], (), minval=-15.0, maxval=15.0), jax.random.uniform(k[1], (), minval=-15.0, maxval=15.0),
                      jax.random.uniform(k[2], (), minval=5.0, maxval=rho + 20.0)])


mode = sys.argv[1]
t0 = time.time()
if mode == "angles":
    p = lorenz_params()
    res = {}
    for o in range(2):
        x0 = integrate_final(lorenz, start_point(jax.random.PRNGKey(500 + o), 28.0), p, DT, int(50 / DT))
        xs, V = clv_orbit(x0, p, 10000.0, jax.random.PRNGKey(600 + o))
        sl = slice(FWD_BLOCKS, V.shape[0] - BACK_BLOCKS)
        a = angles(xs[sl], V[sl], p)
        res[f"orbit{o}"] = summarise(a)
        np.save(OUT / f"phaseA5b_angles_orbit{o}.npy", np.stack([np.asarray(a[k]) for k in ("theta1", "theta2", "theta3", "fnorm")], axis=1))
        print(f"orbit {o}: {json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items() if kk in ('min', 'P<0.05', 'P<0.1')} for k, v in res[f'orbit{o}'].items() if k.startswith('theta')})}  t={time.time()-t0:.0f}s", flush=True)
    (OUT / "phaseA5b_angles.json").write_text(json.dumps(res, indent=2) + "\n")

elif mode == "link":
    p = lorenz_params()
    k3 = jax.random.split(jax.random.PRNGKey(70), 3)   # the A2 and A5 start points
    starts = jnp.stack([jax.random.uniform(k3[0], (32,), minval=-15.0, maxval=15.0), jax.random.uniform(k3[1], (32,), minval=-15.0, maxval=15.0),
                        jax.random.uniform(k3[2], (32,), minval=5.0, maxval=45.0)], axis=1)
    STEPS, K, TSEG = 100, 2000, 0.5
    rows = {k: [] for k in ("c", "th1", "th2", "th3", "fmin", "vnorm", "orbit")}
    for i in range(32):
        u0 = integrate_final(lorenz, starts[i], p, DT, int(20.0 / DT))
        r = nilss(lorenz, z_of, u0, p, (1,), DT, STEPS, K, 1, jax.random.PRNGKey(7000 + i))
        rec, a = r["rec"], r["a"]
        seg = rec["JV"][:, 0] + jnp.einsum("km,km->k", rec["JW"], a[:, :, 0])
        contrib = (seg + (rec["xi_star"][:, 0] + jnp.einsum("km,km->k", rec["xi_W"], a[:, :, 0])) * (r["J"] - rec["J_end"])) / TSEG
        vperp = rec["Vp_end"][:, :, 0] + jnp.einsum("kab,kb->ka", rec["Wp_end"], a[:, :, 0])
        xs, V = clv_orbit(u0, p, 1000.0, jax.random.PRNGKey(800 + i))
        an = angles(xs, V, p)
        nb = V.shape[0]
        # segment j covers blocks 5j .. 5j+4. Valid CLV blocks: FWD_BLOCKS .. nb - BACK_BLOCKS.
        seg_ok = [j for j in range(K) if 5 * j >= FWD_BLOCKS and 5 * j + 5 <= nb - BACK_BLOCKS]
        for key, arr in (("th1", an["theta1"]), ("th2", an["theta2"]), ("th3", an["theta3"])):
            rows[key].append(np.array([float(jnp.min(arr[5 * j:5 * j + 5])) for j in seg_ok]))
        rows["fmin"].append(np.array([float(jnp.min(an["fnorm"][5 * j:5 * j + 5])) for j in seg_ok]))
        rows["c"].append(np.asarray(contrib)[seg_ok])
        rows["vnorm"].append(np.asarray(jnp.linalg.norm(vperp, axis=1))[seg_ok])
        rows["orbit"].append(np.full(len(seg_ok), i))
        if i % 8 == 7:
            print(f"orbit {i} done t={time.time()-t0:.0f}s", flush=True)
    d = {k: np.concatenate(v) for k, v in rows.items()}
    np.savez(OUT / "phaseA5b_link.npz", **d)
    rng = np.random.default_rng(5)
    c = d["c"]
    est_all = float(c.mean())
    n_rm = int(0.05 * len(c))
    null = np.array([c[np.setdiff1d(np.arange(len(c)), rng.choice(len(c), n_rm, replace=False))].mean() - est_all for _ in range(1000)])
    lo, hi = np.quantile(null, [0.025, 0.975])
    res = {"n_segments": int(len(c)), "est_all": est_all, "gap_prereg": GAP_RHO, "null_95": [float(lo), float(hi)], "criteria": {}}
    for k in ("th1", "th2", "th3", "fmin"):
        idx = np.argsort(d[k])[:n_rm]
        keep = np.ones(len(c), bool); keep[idx] = False
        delta = float(c[keep].mean() - est_all)
        res["criteria"][k] = {"delta": delta, "outside_null": bool(delta < lo or delta > hi), "toward_oracle_half_gap": bool(delta <= -0.5 * GAP_RHO),
                              "explains": bool((delta < lo or delta > hi) and delta <= -0.5 * GAP_RHO),
                              "spearman_log_crit_vs_abs_c": float(spearmanr(np.log(d[k]), np.abs(c))[0]),
                              "spearman_log_crit_vs_log_vnorm": float(spearmanr(np.log(d[k]), np.log(d["vnorm"]))[0])}
    (OUT / "phaseA5b_link.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))

elif mode == "scan":
    out = {}
    rng = np.random.default_rng(9)
    for rho in (26.0, 28.0, 30.0, 33.0, 36.0, 40.0):
        p = lorenz_params(rho=rho)
        # own NILSS, 32 orbits
        vals = []
        for i in range(32):
            u0 = integrate_final(lorenz, start_point(jax.random.fold_in(jax.random.PRNGKey(900), 100 * int(rho) + i), rho), p, DT, int(20.0 / DT))
            vals.append(float(nilss(lorenz, z_of, u0, p, (1,), DT, 100, 2000, 1, jax.random.PRNGKey(7000 + i))["dJds"][0]))
        own, own_sem = float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(32))
        # finite-difference oracle, steps 1.0 and 0.5, N doubled from 4000 until SE < 5%
        data = {1.0: [], 0.5: []}
        N, rnd = 0, 0
        while True:
            key = jax.random.fold_in(jax.random.PRNGKey(950), 1000 * int(rho) + rnd)
            x0s = jax.vmap(lambda k: start_point(k, rho))(jax.random.split(key, 4000))
            for h in data:
                cols = [jnp.concatenate([ensemble_time_average(lorenz, z_of, x0s[j:j + 2000], p.at[1].set(rho + s * h), 0.01, 5000, 10000) for j in range(0, 4000, 2000)]) for s in (-1.0, 1.0)]
                data[h].append(np.asarray(jnp.stack(cols, axis=1)))
            N += 4000; rnd += 1
            d5 = np.concatenate(data[0.5])
            means = d5[rng.integers(0, len(d5), (500, len(d5)))].mean(axis=1)
            se5 = float(((means[:, 1] - means[:, 0]) / 1.0).std(ddof=1))
            s5 = float((d5.mean(0)[1] - d5.mean(0)[0]) / 1.0)
            if se5 < 0.05 * abs(s5) or N >= 64000:
                break
        d1 = np.concatenate(data[1.0])
        m1 = d1[rng.integers(0, len(d1), (500, len(d1)))].mean(axis=1)
        se1 = float(((m1[:, 1] - m1[:, 0]) / 2.0).std(ddof=1)); s1 = float((d1.mean(0)[1] - d1.mean(0)[0]) / 2.0)
        s5 = float((d5.mean(0)[1] - d5.mean(0)[0]) / 1.0); se5 = float(((means[:, 1] - means[:, 0]) / 1.0).std(ddof=1))
        conv = bool(abs(s1 - s5) <= 2 * np.hypot(se1, se5))
        # CLV statistics, one orbit of T = 10000
        x0 = integrate_final(lorenz, start_point(jax.random.PRNGKey(990 + int(rho)), rho), p, DT, int(50 / DT))
        xs, V = clv_orbit(x0, p, 10000.0, jax.random.PRNGKey(995))
        sl = slice(FWD_BLOCKS, V.shape[0] - BACK_BLOCKS)
        a = angles(xs[sl], V[sl], p)
        st = summarise(a)
        out[str(int(rho))] = {"rho": rho, "own_nilss": own, "own_sem": own_sem, "fd_h0.5": s5, "fd_h0.5_se": se5, "fd_h1": s1, "fd_h1_se": se1, "oracle_converged": conv, "N": N,
                              "gap_rel": (own - s5) / abs(s5), "gap_rel_sys": abs(s1 - s5) / abs(s5), "clv": st}
        print(f"rho={rho:g}: own {own:.4f}, FD(0.5) {s5:.4f}+-{se5:.4f}, FD(1) {s1:.4f}, gap {100*(own-s5)/abs(s5):+.2f}% conv={conv}, P(th1<0.1)={st['theta1']['P<0.1']:.4f} P(th2<0.1)={st['theta2']['P<0.1']:.4f} P(th3<0.1)={st['theta3']['P<0.1']:.4f} t={time.time()-t0:.0f}s", flush=True)
    keys = list(out)
    gaps = np.array([out[k]["gap_rel"] for k in keys])
    corr = {}
    for stat in ("theta1", "theta2", "theta3"):
        s = np.array([out[k]["clv"][stat]["P<0.1"] for k in keys])
        corr[f"spearman_gap_vs_P({stat}<0.1)"] = float(spearmanr(gaps, s)[0])
    corr["spearman_gap_vs_fnorm_q1%"] = float(spearmanr(gaps, [out[k]["clv"]["fnorm"]["q1%"] for k in keys])[0])
    out["correlations"] = corr
    (OUT / "phaseA5b_scan.json").write_text(json.dumps(out, indent=2) + "\n")
    print(corr)
