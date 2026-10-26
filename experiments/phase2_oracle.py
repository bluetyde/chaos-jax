"""Phase 2: ensemble finite-difference oracle for d<z>/drho at rho = 28. Gates in docs/PREDICTIONS-2.md."""
import json
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np

import chaos_jax  # noqa: F401
from chaos_jax.response import bootstrap_se, ensemble_time_average
from chaos_jax.systems import lorenz, lorenz_params

N, DT, T_SPIN, T_AVG = 1000, 0.01, 50.0, 100.0
RHOS = np.array([27.0, 27.5, 28.0, 28.5, 29.0])
N_BOOT = 2000

key = jax.random.PRNGKey(0)
kx, ky, kz = jax.random.split(key, 3)
x0s = jnp.stack([jax.random.uniform(kx, (N,), minval=-15.0, maxval=15.0),
                 jax.random.uniform(ky, (N,), minval=-15.0, maxval=15.0),
                 jax.random.uniform(kz, (N,), minval=5.0, maxval=45.0)], axis=1)

t0 = time.time()
cols = []
for rho in RHOS:
    zbar = ensemble_time_average(lorenz, lambda x: x[2], x0s, lorenz_params(rho=float(rho)), DT,
                                 int(round(T_SPIN / DT)), int(round(T_AVG / DT)))
    zbar.block_until_ready()
    cols.append(zbar)
    print(f"rho={rho}: mean<z>={float(zbar.mean()):.4f}  sd over starts={float(zbar.std(ddof=1)):.4f}  t={time.time()-t0:.0f}s", flush=True)
data = jnp.stack(cols, axis=1)
wall = time.time() - t0

u = RHOS - 28.0
V = np.stack([np.ones_like(u), u, u ** 2], axis=1)
pinv = jnp.asarray(np.linalg.pinv(V))
slope = lambda m: (m[4] - m[0]) / 2.0
c2 = lambda m: (pinv @ m)[2]
lin_slope = lambda m: (pinv @ m)[1]
means = data.mean(axis=0)
k1, k2, k3 = jax.random.split(jax.random.PRNGKey(1), 3)
res = {
    "N": N, "dt": DT, "T_spin": T_SPIN, "T_avg": T_AVG, "rhos": RHOS.tolist(), "wall_seconds": wall,
    "mean_z": [float(v) for v in means],
    "estimate_central_h1": float(slope(means)), "se_central_h1": float(bootstrap_se(slope, data, k1, N_BOOT)),
    "quad_c2": float(c2(means)), "se_c2": float(bootstrap_se(c2, data, k2, N_BOOT)),
    "quad_linear_term": float(lin_slope(means)), "se_linear_term": float(bootstrap_se(lin_slope, data, k3, N_BOOT)),
    "slope_27_28": float((means[2] - means[0]) / 1.0), "slope_28_29": float((means[4] - means[2]) / 1.0),
}
res["rel_se"] = res["se_central_h1"] / abs(res["estimate_central_h1"])
res["G2.1_pass"] = bool(res["rel_se"] < 0.10)
res["G2.2_pass"] = bool(abs(res["quad_c2"]) < 2.0 * res["se_c2"])
print(json.dumps(res, indent=2))
out = pathlib.Path(__file__).resolve().parents[1] / "results"
out.mkdir(exist_ok=True)
(out / "phase2_oracle.json").write_text(json.dumps(res, indent=2) + "\n")
