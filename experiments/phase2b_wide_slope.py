"""Phase 2 follow-up: slope of <z> against rho over [25, 50]. Method fixed in docs/PREDICTIONS-2b.md."""
import json
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np

import chaos_jax  # noqa: F401
from chaos_jax.response import bootstrap_se, ensemble_time_average, time_average
from chaos_jax.systems import lorenz, lorenz_params

zobs = lambda x: x[2]
t0 = time.time()

# ---- A: ensemble over a wide range (Phase 2 pipeline) ----
N, DT, N_SPIN, N_AVG = 1000, 0.01, 5000, 10000
kx, ky, kz = jax.random.split(jax.random.PRNGKey(0), 3)
x0s = jnp.stack([jax.random.uniform(kx, (N,), minval=-15.0, maxval=15.0),
                 jax.random.uniform(ky, (N,), minval=-15.0, maxval=15.0),
                 jax.random.uniform(kz, (N,), minval=5.0, maxval=45.0)], axis=1)
rhos = np.arange(25.0, 50.5, 1.0)
cols = [ensemble_time_average(lorenz, zobs, x0s, lorenz_params(rho=float(r)), DT, N_SPIN, N_AVG) for r in rhos]
data = jnp.stack(cols, axis=1)
data.block_until_ready()
print(f"A done t={time.time()-t0:.0f}s", flush=True)
means = data.mean(axis=0)


def fit_slope(r, m):
    return float(np.polyfit(r, m, 1)[0])


V1 = np.stack([np.ones_like(rhos), rhos - 37.5], axis=1)
pinv1 = jnp.asarray(np.linalg.pinv(V1))
line = lambda m: (pinv1 @ m)[1]
V2 = np.stack([np.ones_like(rhos), rhos - 37.5, (rhos - 37.5) ** 2], axis=1)
pinv2 = jnp.asarray(np.linalg.pinv(V2))
quad_local_at_28 = lambda m: (pinv2 @ m)[1] + 2.0 * (pinv2 @ m)[2] * (28.0 - 37.5)
A = {"rhos": rhos.tolist(), "mean_z": [float(v) for v in means],
     "line_slope": float(line(means)), "line_slope_se": float(bootstrap_se(line, data, jax.random.PRNGKey(1))),
     "quad_slope_at_28": float(quad_local_at_28(means)),
     "quad_slope_at_28_se": float(bootstrap_se(quad_local_at_28, data, jax.random.PRNGKey(5)))}
wins = {}
for lo in (25, 30, 35, 40, 45):
    sel = (rhos >= lo) & (rhos <= lo + 5)
    Vw = np.stack([np.ones(sel.sum()), rhos[sel] - (lo + 2.5)], axis=1)
    pw = jnp.asarray(np.linalg.pinv(Vw))
    f = lambda m, sel=sel, pw=pw: (pw @ m[np.where(sel)[0]])[1]
    wins[f"[{lo},{lo+5}]"] = {"slope": float(f(means)), "se": float(bootstrap_se(f, data, jax.random.PRNGKey(10 + lo)))}
A["window_slopes"] = wins

# ---- B: Lea-like single run, many rho, one start point; then 19 more start points ----
DTB, N_AVG_B = 0.005, 26272
rho_b = np.round(np.arange(25.0, 50.0 + 1e-9, 0.005), 6)
pb = jnp.stack([jnp.full(rho_b.shape, 10.0), jnp.asarray(rho_b), jnp.full(rho_b.shape, 8.0 / 3.0)], axis=1)
run = jax.jit(jax.vmap(lambda p, x0: time_average(lorenz, zobs, x0, p, DTB, 0, N_AVG_B), in_axes=(0, None)))
k2 = jax.random.split(jax.random.PRNGKey(2), 3)
extra = jnp.stack([jax.random.uniform(k2[0], (19,), minval=-15.0, maxval=15.0),
                   jax.random.uniform(k2[1], (19,), minval=-15.0, maxval=15.0),
                   jax.random.uniform(k2[2], (19,), minval=5.0, maxval=45.0)], axis=1)
starts = jnp.concatenate([jnp.array([[-2.40, -3.70, 14.98]]), extra], axis=0)
slopes = []
for i in range(20):
    zb = np.asarray(run(pb, starts[i]))
    slopes.append(fit_slope(rho_b, zb))
    print(f"B start {i}: slope={slopes[-1]:.4f} t={time.time()-t0:.0f}s", flush=True)
B = {"lea_start_slope": slopes[0], "slopes": slopes, "min": min(slopes), "max": max(slopes),
     "mean": float(np.mean(slopes)), "sd": float(np.std(slopes, ddof=1))}

A_line = A["line_slope"]
verdict = ("range explains the gap" if 0.94 <= A_line <= 0.98 else
           "range explains part of the gap" if 0.98 < A_line <= 1.00 else "range does not explain the gap")
res = {"A": A, "B": B, "verdict_A": verdict, "wall_seconds": time.time() - t0}
print(json.dumps(res, indent=2))
out = pathlib.Path(__file__).resolve().parents[1] / "results"
out.mkdir(exist_ok=True)
(out / "phase2b_wide_slope.json").write_text(json.dumps(res, indent=2) + "\n")
