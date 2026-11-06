"""A5c: S3 on the Solenoid map against a noise-free oracle. Design in docs/PREDICTIONS-A5c.md.

python experiments/phaseA5c_solenoid.py probe    wall time of one chunk
python experiments/phaseA5c_solenoid.py run      S3 and shadowing at s = (1, 0) and (1, 0.5), plus orbit means at s2 = 0.45, 0.55
python experiments/phaseA5c_solenoid.py report   gates
"""
import json
import pathlib
import sys
import time

import numpy as np

import jax
import jax.numpy as jnp
import chaos_jax  # noqa: F401
from chaos_jax import s3, solenoid

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
M, N, K_RUN, N_LAGS, TAIL, CHUNK = 512, 20000, 100, 24, 100, 32
ES = jnp.eye(2)


def starts(seed=5):
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    th = jax.random.uniform(k1, (M,), minval=0.0, maxval=2 * jnp.pi)
    x0 = jnp.stack([jnp.ones(M), th, jnp.zeros(M)], axis=1)
    q0 = jax.random.normal(k2, (M, 3))
    return x0, q0


def one_orbit(x0, q0, s):
    arr = s3.orbit_arrays(solenoid.phi, solenoid.objective, s, ES, x0, q0, K_RUN + N + TAIL)
    return s3.estimates(arr, K_RUN, N, N_LAGS)


@jax.jit
def chunk(x0s, q0s, s):
    return jax.vmap(one_orbit, in_axes=(0, 0, None))(x0s, q0s, s)


@jax.jit
def mean_J_chunk(x0s, s):
    def one(x0):
        def step(x, _):
            x = solenoid.phi(x, s)
            return x, solenoid.objective(x)

        x, _ = jax.lax.scan(step, x0, None, length=K_RUN)
        _, js = jax.lax.scan(step, x, None, length=N)
        return js.mean()

    return jax.vmap(one)(x0s)


def run_point(s, x0, q0):
    outs = [chunk(x0[i:i + CHUNK], q0[i:i + CHUNK], jnp.array(s)) for i in range(0, M, CHUNK)]
    return {k: np.concatenate([np.asarray(o[k]) for o in outs]) for k in outs[0]}


def summarize(d):
    res = {}
    for j, name in enumerate(("s1", "s2")):
        tot = d["stable"][:, j] + d["unstable"][:, j]
        r = {}
        for key, arr in (("stable", d["stable"][:, j]), ("unstable", d["unstable"][:, j]), ("total", tot), ("shadow", d["shadow"][:, j])):
            r[key] = {"mean": float(arr.mean()), "se": float(arr.std(ddof=1) / np.sqrt(arr.size))}
        res[name] = r
    res["meanJ"] = {"mean": float(d["meanJ"].mean()), "se": float(d["meanJ"].std(ddof=1) / np.sqrt(M))}
    return res


def probe():
    x0, q0 = starts()
    chunk(x0[:CHUNK], q0[:CHUNK], jnp.array([1.0, 0.0]))  # compile
    t = time.time()
    jax.block_until_ready(chunk(x0[CHUNK:2 * CHUNK], q0[CHUNK:2 * CHUNK], jnp.array([1.0, 0.0])))
    dt = time.time() - t
    print("one chunk of", CHUNK, "orbits:", dt, "s; whole run of 2 points:", 2 * dt * M / CHUNK, "s plus orbit means")


def run():
    x0, q0 = starts()
    res = {"settings": dict(M=M, N=N, K_run=K_RUN, n_lags=N_LAGS, tail=TAIL, seed=5), "points": {}}
    t = time.time()
    for s in ((1.0, 0.0), (1.0, 0.5)):
        res["points"][str(s)] = summarize(run_point(s, x0, q0))
        print(s, res["points"][str(s)], time.time() - t, flush=True)
    for s2 in (0.45, 0.55):
        m = np.concatenate([np.asarray(mean_J_chunk(x0[i:i + CHUNK], jnp.array([1.0, s2]))) for i in range(0, M, CHUNK)])
        res[f"orbit_mean_s2_{s2}"] = {"mean": float(m.mean()), "se": float(m.std(ddof=1) / np.sqrt(M))}
    res["seconds"] = time.time() - t
    (OUT / "phaseA5c_s3.json").write_text(json.dumps(res, indent=1))
    print("seconds", res["seconds"])


def report():
    d = json.loads((OUT / "phaseA5c_s3.json").read_text())
    out = {"settings": d["settings"]}
    g0 = {}
    for nb in (48, 96):
        g0[nb] = {"E_10": solenoid.expected_J(1.0, 0.0, nb=nb)[0], "E_105": solenoid.expected_J(1.0, 0.5, nb=nb)[0],
                  "d_10": solenoid.oracle_derivatives(1.0, 0.0, nb=nb), "d_105": solenoid.oracle_derivatives(1.0, 0.5, nb=nb)}
    o = {"(1.0, 0.0)": g0[96]["d_10"], "(1.0, 0.5)": g0[96]["d_105"]}
    EJ = {"(1.0, 0.0)": g0[96]["E_10"], "(1.0, 0.5)": g0[96]["E_105"]}
    out["G-M0a_basis_diff"] = {"E": max(abs(g0[48]["E_10"] - g0[96]["E_10"]), abs(g0[48]["E_105"] - g0[96]["E_105"])),
                               "derivs": max(abs(a - b) for k in ("d_10", "d_105") for a, b in zip(g0[48][k], g0[96][k]))}
    out["G-M0a_pass"] = bool(out["G-M0a_basis_diff"]["E"] < 1e-9 and out["G-M0a_basis_diff"]["derivs"] < 1e-9)
    out["G-M0b_ds1_minus_2"] = abs(g0[96]["d_10"][0] - 2.0)
    out["G-M0b_pass"] = bool(out["G-M0b_ds1_minus_2"] < 1e-8)
    gc = {}
    for key in o:
        m = d["points"][key]["meanJ"]
        gc[key] = {"orbit_mean": m["mean"], "se": m["se"], "oracle": EJ[key], "z": (m["mean"] - EJ[key]) / m["se"]}
    out["G-M0c"] = gc
    out["G-M0c_pass"] = bool(all(abs(v["z"]) < 3 for v in gc.values()))
    a, b = d["orbit_mean_s2_0.45"], d["orbit_mean_s2_0.55"]
    fd, se = (b["mean"] - a["mean"]) / 0.1, np.hypot(a["se"], b["se"]) / 0.1
    out["G-M0d"] = {"fd_slope": fd, "se": float(se), "oracle": o["(1.0, 0.5)"][1], "z": float((fd - o["(1.0, 0.5)"][1]) / se)}
    out["G-M0d_pass"] = bool(abs(out["G-M0d"]["z"]) < 3)
    pts = {}
    for key, pr in d["points"].items():
        r = {}
        for j, name in enumerate(("s1", "s2")):
            t = pr[name]
            orc = o[key][j]
            r[name] = {"oracle": orc, "S3_total": t["total"], "S3_stable": t["stable"], "S3_unstable": t["unstable"], "shadow": t["shadow"],
                       "total_rel_err": (t["total"]["mean"] - orc) / abs(orc), "total_z": (t["total"]["mean"] - orc) / t["total"]["se"],
                       "shadow_rel_err": (t["shadow"]["mean"] - orc) / abs(orc), "shadow_z": (t["shadow"]["mean"] - orc) / t["shadow"]["se"]}
        pts[key] = r
    out["points"] = pts
    p10, p105 = pts["(1.0, 0.0)"], pts["(1.0, 0.5)"]
    out["G-M1_pass"] = bool(abs(p10["s1"]["total_z"]) < 3 and abs(p10["s1"]["total_rel_err"]) < 0.01)
    out["G-M2_pass"] = bool(abs(p10["s2"]["total_z"]) < 3 and abs(p10["s2"]["total_rel_err"]) < 0.02)
    out["G-M3_pass"] = bool(abs(p105["s2"]["total_z"]) < 3 and abs(p105["s2"]["total_rel_err"]) < 0.02)
    out["G-M4_pass"] = bool(abs(p10["s1"]["shadow_rel_err"]) < 0.01)
    (OUT / "phaseA5c_report.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    {"probe": probe, "run": run, "report": report}[sys.argv[1]]()
