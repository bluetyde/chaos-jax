"""A6 stage 1: spread of a long-time average of the double pendulum across starts on one energy shell. Design in docs/PREDICTIONS-A6.md.

python experiments/phaseA6_spread.py controls   C-A6.1 and C-A6.2
python experiments/phaseA6_spread.py run        main scan (32 starts, T = 1000) and Lyapunov exponents
python experiments/phaseA6_spread.py report     gates
"""
import json
import pathlib
import sys
import time
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
import chaos_jax  # noqa: F401
from chaos_jax.integrate import rk4_step
from chaos_jax.lyapunov import lyapunov_spectrum
from chaos_jax.systems import double_pendulum, double_pendulum_energy, pendulum_params

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
P = pendulum_params()
DT = 0.00125
BLOCK_STEPS = int(round(50.0 / DT))  # 50 time units per block
N_BLOCKS = 20  # T = 1000
G, L1, L2, M1, M2 = (float(v) for v in P)
E_TARGET = float(double_pendulum_energy(jnp.array([2.5, 2.5, 0.0, 0.0]), P))


def starts(n=32, seed=60):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        t1, t2 = rng.uniform(-np.pi, np.pi, 2)
        v = -(M1 + M2) * G * L1 * np.cos(t1) - M2 * G * L2 * np.cos(t2)
        k = E_TARGET - v
        if k <= 0:
            continue
        d = t1 - t2
        mat = np.array([[(M1 + M2) * L1 ** 2, M2 * L1 * L2 * np.cos(d)], [M2 * L1 * L2 * np.cos(d), M2 * L2 ** 2]])
        a = rng.uniform(0, 2 * np.pi)
        u = np.array([np.cos(a), np.sin(a)])
        w = u * np.sqrt(2.0 * k / (u @ mat @ u))
        out.append([t1, t2, w[0], w[1]])
    return jnp.array(out)


@partial(jax.jit, static_argnames=("n_blocks",))
def scan_one(x0, n_blocks):
    def step(c, _):
        x, s2, s1 = c
        x = rk4_step(double_pendulum, x, DT, P)
        return (x, s2 + jnp.cos(x[1]), s1 + jnp.cos(x[0])), None

    def block(x, _):
        (x, s2, s1), _ = jax.lax.scan(step, (x, 0.0, 0.0), None, length=BLOCK_STEPS)
        return x, jnp.stack([s2 / BLOCK_STEPS, s1 / BLOCK_STEPS])

    x, means = jax.lax.scan(block, x0, None, length=n_blocks)
    return x, means


def energy(x):
    return double_pendulum_energy(x, P)


def controls():
    x0 = starts()
    err = jnp.abs(energy(x0) - E_TARGET) / abs(E_TARGET)
    c1 = float(err.max())
    xf = jax.vmap(lambda x: scan_one(x, 2)[0])(x0[:4])  # T = 100
    c2 = float((jnp.abs(energy(xf) - E_TARGET) / abs(E_TARGET)).max())
    res = {"E_target": E_TARGET, "C-A6.1_max_rel_energy_error": c1, "C-A6.1_pass": c1 < 1e-12, "C-A6.2_max_rel_drift_T100": c2, "C-A6.2_pass": c2 < 1e-6}
    print(json.dumps(res, indent=1))
    (OUT / "phaseA6_controls.json").write_text(json.dumps(res, indent=1))


def run():
    x0 = starts()
    t = time.time()
    xf, means = jax.vmap(lambda x: scan_one(x, N_BLOCKS))(x0)
    jax.block_until_ready(means)
    t_scan = time.time() - t
    drift = np.asarray(jnp.abs(energy(xf) - E_TARGET) / abs(E_TARGET))
    t = time.time()
    lam = []
    for i in range(x0.shape[0]):
        e, _, _ = lyapunov_spectrum(double_pendulum, x0[i], P, DT, 16000, 10, k=1, discard_blocks=100)
        lam.append(float(e[0]))
    res = {"E_target": E_TARGET, "dt": DT, "T": 50.0 * N_BLOCKS, "n_starts": int(x0.shape[0]), "block_time": 50.0, "seconds_scan": t_scan, "seconds_lyapunov": time.time() - t,
           "start_points": np.asarray(x0).tolist(), "block_means_cos_theta2": np.asarray(means[:, :, 0]).tolist(), "block_means_cos_theta1": np.asarray(means[:, :, 1]).tolist(),
           "final_rel_energy_drift": drift.tolist(), "lambda1_T200": lam}
    (OUT / "phaseA6_spread.json").write_text(json.dumps(res, indent=1))
    print("scan s", t_scan, "lyap s", res["seconds_lyapunov"], "max drift", drift.max())


def spread(b):
    b = np.asarray(b)  # (starts, blocks)
    m = b.mean(axis=1)
    se = b.std(axis=1, ddof=1) / np.sqrt(b.shape[1])
    m250 = b[:, :5].mean(axis=1)
    return m, se, m250


def report():
    d = json.loads((OUT / "phaseA6_spread.json").read_text())
    out = {"E_target": d["E_target"], "T": d["T"], "n_starts": d["n_starts"]}
    out["C-A6.3_max_final_drift"] = float(max(d["final_rel_energy_drift"]))
    out["C-A6.3_pass"] = bool(out["C-A6.3_max_final_drift"] < 1e-4)
    for name in ("cos_theta2", "cos_theta1"):
        m, se, m250 = spread(d[f"block_means_{name}"])
        sd, sd250 = float(m.std(ddof=1)), float(m250.std(ddof=1))
        med = float(np.median(m))
        z = np.abs(m - med) / se
        r = {"ensemble_mean": float(m.mean()), "range": float(m.max() - m.min()), "sd_between_T1000": sd, "sd_between_T250": sd250, "mean_within_se": float(se.mean()),
             "R": sd / float(se.mean()), "ratio_T1000_over_T250": sd / sd250, "max_z_from_median": float(z.max()), "worst_starts": [int(i) for i in np.argsort(-z)[:3]]}
        r["G-A6.1_pass"], r["G-A6.2_pass"], r["G-A6.3_pass"] = bool(r["R"] < 2.0), bool(r["ratio_T1000_over_T250"] < 0.7), bool(r["max_z_from_median"] < 4.0)
        out[name] = r
    lam = np.asarray(d["lambda1_T200"])
    out["lambda1_T200"] = {"min": float(lam.min()), "median": float(np.median(lam)), "max": float(lam.max()), "n_above_0.5": int((lam > 0.5).sum())}
    p = out["cos_theta2"]
    out["stage1_pass"] = bool(p["G-A6.1_pass"] and p["G-A6.2_pass"] and p["G-A6.3_pass"])
    (OUT / "phaseA6_report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"controls": controls, "run": run, "report": report}[sys.argv[1]]()
