"""Phase 3: gradient blow-up. Method and gates in docs/PREDICTIONS-3.md.

Usage:  python experiments/phase3_blowup.py f64      (controls, Jacobian growth, float64 gradients)
        python experiments/phase3_blowup.py f32      (float32 gradients, separate process)
        python experiments/phase3_blowup.py report   (gates, departure times)
"""
import json
import pathlib
import sys
import time

import numpy as np

import jax
import chaos_jax  # noqa: F401
from chaos_jax import cases

mode = sys.argv[1]
if mode == "f32":
    jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp

from chaos_jax.blowup import log_jacobian_norms, naive_gradients
from chaos_jax.integrate import integrate_final
from chaos_jax.systems import (damped_oscillator, damped_oscillator_params, double_pendulum, lorenz,
                               lorenz_params, pendulum_params)

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
N, DT = 200, 0.01
LAMBDA1 = 0.9041
ORACLE = 1.0029
T_GRID = np.arange(1, 31)
T_J = 0.5 * np.arange(1, 41)  # Jacobian times 0.5 .. 20
t0 = time.time()


def fit_slope(t, logs):
    sel = t >= 2.0
    return float(np.polyfit(t[sel], logs[sel], 1)[0])


def slope_with_se(t, logs_by_start, n_boot=2000, seed=0):
    slope = fit_slope(t, logs_by_start.mean(axis=0))
    rng = np.random.default_rng(seed)
    n = logs_by_start.shape[0]
    boots = [fit_slope(t, logs_by_start[rng.integers(0, n, n)].mean(axis=0)) for _ in range(n_boot)]
    return slope, float(np.std(boots, ddof=1))


if mode == "f64":
    for f in OUT.glob("phase3_f64.npz"):
        f.unlink()
    res = {}
    # ---- controls first ----
    key = jax.random.PRNGKey(31)
    osc0 = jax.random.uniform(key, (50, 2), minval=-2.0, maxval=2.0)
    po = damped_oscillator_params()
    logs = np.asarray(jax.vmap(lambda x: log_jacobian_norms(damped_oscillator, x, po, 0.01, 40, 50))(osc0))
    res["oscillator_slope"], res["oscillator_slope_se"] = slope_with_se(T_J, logs)
    print("control oscillator slope", res["oscillator_slope"], flush=True)

    pp = pendulum_params()
    th = jax.random.uniform(jax.random.PRNGKey(32), (50, 2), minval=0.05, maxval=0.15)
    pend0 = jnp.concatenate([th, jnp.zeros((50, 2))], axis=1)
    spb = int(round(0.5 / cases.PENDULUM_DT))
    logs = np.asarray(jax.vmap(lambda x: log_jacobian_norms(double_pendulum, x, pp, cases.PENDULUM_DT, 40, spb))(pend0))
    res["pendulum_low_slope"], res["pendulum_low_slope_se"] = slope_with_se(T_J, logs)
    print("control low-energy pendulum slope", res["pendulum_low_slope"], flush=True)

    # ---- Lorenz ----
    p = lorenz_params()
    kx, ky, kz = jax.random.split(jax.random.PRNGKey(30), 3)
    start = jnp.stack([jax.random.uniform(kx, (N,), minval=-15.0, maxval=15.0),
                       jax.random.uniform(ky, (N,), minval=-15.0, maxval=15.0),
                       jax.random.uniform(kz, (N,), minval=5.0, maxval=45.0)], axis=1)
    x0s = jax.vmap(lambda x: integrate_final(lorenz, x, p, DT, 5000))(start)
    logsL = np.asarray(jax.vmap(lambda x: log_jacobian_norms(lorenz, x, p, DT, 40, 50))(x0s))
    res["lorenz_slope"], res["lorenz_slope_se"] = slope_with_se(T_J, logsL)
    res["lorenz_slope_vs_lambda1_rel"] = res["lorenz_slope"] / LAMBDA1 - 1.0
    print("lorenz slope", res["lorenz_slope"], flush=True)

    g = np.stack([np.asarray(naive_gradients(x0s, 28.0, DT, int(round(T / DT)))) for T in T_GRID], axis=1)
    print(f"gradients done t={time.time()-t0:.0f}s", flush=True)
    np.savez(OUT / "phase3_f64.npz", x0s=np.asarray(x0s), g=g, logs_lorenz=logsL)
    res["wall_seconds_f64"] = time.time() - t0
    (OUT / "phase3_f64.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))

elif mode == "f32":
    d = np.load(OUT / "phase3_f64.npz")
    x0s = jnp.asarray(d["x0s"], dtype=jnp.float32)
    assert x0s.dtype == jnp.float32
    g = np.stack([np.asarray(naive_gradients(x0s, jnp.float32(28.0), jnp.float32(DT), int(round(T / DT)))) for T in T_GRID], axis=1)
    assert g.dtype == np.float32
    np.savez(OUT / "phase3_f32.npz", g=g)
    print(f"float32 gradients done in {time.time()-t0:.0f}s, dtype {g.dtype}")

elif mode == "report":
    r64 = json.loads((OUT / "phase3_f64.json").read_text())
    g64 = np.load(OUT / "phase3_f64.npz")["g"]
    g32 = np.load(OUT / "phase3_f32.npz")["g"].astype(np.float64)
    med_abs = np.median(np.abs(g64), axis=0)
    e64 = np.median(np.abs(g64 - ORACLE) / ORACLE, axis=0)
    e32 = np.median(np.abs(g32 - g64) / np.abs(g64), axis=0)

    def permanent_departure(e, tmin):
        for i, T in enumerate(T_GRID):
            if T >= tmin and np.all(e[i:] > 0.25):
                return int(T)
        return None

    exceed = np.where(med_abs > 10 * ORACLE)[0]
    out = dict(r64)
    out.update({
        "T_grid": T_GRID.tolist(), "median_abs_g64": med_abs.tolist(), "e64": e64.tolist(), "e32": e32.tolist(),
        "T_star": int(T_GRID[exceed[0]]) if len(exceed) else None,
        "T_leave64": permanent_departure(e64, 2), "T_leave32": permanent_departure(e32, 1),
        "frac_start_points_with_sign_flip_vs_oracle_at_T": {int(T): float(np.mean(g64[:, i] < 0)) for i, T in enumerate(T_GRID) if T in (5, 10, 20, 30)},
    })
    out["P3.1_pass"] = bool(abs(out["lorenz_slope_vs_lambda1_rel"]) <= 0.15)
    out["P3.2_pass"] = bool(out["oscillator_slope"] <= 0.02 and out["pendulum_low_slope"] <= 0.02)
    out["P3.3_pass"] = out["T_star"] is not None
    out["P3.4_pass"] = (out["T_leave32"] is not None) and (out["T_leave64"] is None or out["T_leave32"] < out["T_leave64"])
    print(json.dumps({k: v for k, v in out.items() if k not in ("median_abs_g64", "e64", "e32", "T_grid")}, indent=2))
    for T, a, b, c in zip(T_GRID, med_abs, e64, e32):
        print(f"T={T:2d}  median|g64|={a:12.4g}  e64={b:10.4g}  e32={c:10.4g}")
    (OUT / "phase3.json").write_text(json.dumps(out, indent=2) + "\n")
