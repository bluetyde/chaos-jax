"""Phase 4: compare naive, truncated, ensemble and LSS gradients of the time average of z. Gates in docs/PREDICTIONS-4.md.

python experiments/phase4_fixes.py gen          5000 start points
python experiments/phase4_fixes.py grads        naive and truncated (L = 2.0), all T, all start points (about 10 min)
python experiments/phase4_fixes.py lss          LSS, first 200 start points, all T
python experiments/phase4_fixes.py alpha        exploratory sweep of alpha at T = 80
python experiments/phase4_fixes.py report       gates and tables
python experiments/phase4_fixes.py cost         wall time and peak memory, one process per (method, T)
"""
import json
import pathlib
import resource
import subprocess
import sys
import time

import numpy as np

import jax
import jax.numpy as jnp
import chaos_jax  # noqa: F401
from chaos_jax.fixes import batched_param_gradient, lss_derivative, param_gradient
from chaos_jax.integrate import integrate, integrate_final
from chaos_jax.systems import lorenz, lorenz_params

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
DT, ORACLE, TS = 0.01, 1.0029, (5, 10, 20, 40, 80)
N_ALL, N_SINGLE, M, R, L_TRUNC, ALPHA, RHO_IDX = 5000, 200, 100, 50, 2.0, 10.0, 1
P = lorenz_params()


def zobs(x):
    return x[2]


def load_x0s():
    return jnp.asarray(np.load(OUT / "phase4_x0s.npy"))


def lss_batch(x0s, T, alpha, chunk=50):
    n = int(round(T / DT))
    one = jax.jit(lambda x0: lss_derivative(lorenz, zobs, integrate(lorenz, x0, P, DT, n)[1:], P, RHO_IDX, DT, alpha)[0])
    f = jax.vmap(one)
    return np.concatenate([np.asarray(f(x0s[i:i + chunk])) for i in range(0, x0s.shape[0], chunk)])


def stats(est):
    q1, med, q3 = np.percentile(est, [25, 50, 75])
    rel_err = abs(med - ORACLE) / ORACLE
    rel_iqr = (q3 - q1) / ORACLE
    return {"median": float(med), "rel_err_of_median": float(rel_err), "rel_iqr": float(rel_iqr),
            "works": bool(rel_err <= 0.25 and rel_iqr <= 1.0)}


mode = sys.argv[1]
t0 = time.time()
if mode == "gen":
    k = jax.random.split(jax.random.PRNGKey(40), 3)
    s = jnp.stack([jax.random.uniform(k[0], (N_ALL,), minval=-15.0, maxval=15.0),
                   jax.random.uniform(k[1], (N_ALL,), minval=-15.0, maxval=15.0),
                   jax.random.uniform(k[2], (N_ALL,), minval=5.0, maxval=45.0)], axis=1)
    x0s = jax.vmap(lambda x: integrate_final(lorenz, x, P, DT, 5000))(s)
    np.save(OUT / "phase4_x0s.npy", np.asarray(x0s))
    print("saved", x0s.shape)

elif mode == "grads":
    x0s = load_x0s()
    seg = int(round(L_TRUNC / DT))
    for T in TS:
        n = int(round(T / DT))
        a = np.asarray(batched_param_gradient(lorenz, zobs, x0s, P, RHO_IDX, DT, n))
        b = np.asarray(batched_param_gradient(lorenz, zobs, x0s, P, RHO_IDX, DT, n, seg))
        np.savez(OUT / f"phase4_g_T{T}.npz", naive=a, trunc=b)
        print(f"T={T}: naive median {np.median(a):.4g}, trunc median {np.median(b):.4g}, t={time.time()-t0:.0f}s", flush=True)

elif mode == "lss":
    x0s = load_x0s()[:N_SINGLE]
    for T in TS:
        e = lss_batch(x0s, T, ALPHA)
        np.save(OUT / f"phase4_lss_T{T}.npy", e)
        print(f"T={T}: lss median {np.median(e):.5g} t={time.time()-t0:.0f}s", flush=True)

elif mode == "alpha":
    x0s = load_x0s()[:50]
    res = {}
    for a in (1.0, 3.0, 10.0, 30.0, 100.0):
        e = lss_batch(x0s, 80, a)
        res[str(a)] = {"median": float(np.median(e)), "rel_iqr": float((np.percentile(e, 75) - np.percentile(e, 25)) / ORACLE)}
        print("alpha", a, res[str(a)], flush=True)
    (OUT / "phase4_alpha.json").write_text(json.dumps(res, indent=2) + "\n")

elif mode == "report":
    res = {}
    for T in TS:
        d = np.load(OUT / f"phase4_g_T{T}.npz")
        naive, trunc = d["naive"], d["trunc"]
        lss = np.load(OUT / f"phase4_lss_T{T}.npy")
        ens = lambda g: g.reshape(R, M).mean(axis=1)
        res[T] = {"a_naive": stats(naive[:N_SINGLE]), "b_truncated": stats(trunc[:N_SINGLE]),
                  "c_ensemble_naive": stats(ens(naive)), "c_ensemble_truncated": stats(ens(trunc)),
                  "d_lss": stats(lss)}
    print(f"{'T':>3} {'method':<22} {'median':>10} {'err|med|':>9} {'IQR/O':>9} works")
    for T, r in res.items():
        for k, v in r.items():
            print(f"{T:>3} {k:<22} {v['median']:>10.4g} {v['rel_err_of_median']:>9.3g} {v['rel_iqr']:>9.3g} {v['works']}")
    (OUT / "phase4.json").write_text(json.dumps({str(k): v for k, v in res.items()}, indent=2) + "\n")

elif mode == "cost-one":
    method, T = sys.argv[2], int(sys.argv[3])
    x0 = load_x0s()[0]
    n = int(round(T / DT))
    if method == "naive":
        run = lambda: param_gradient(lorenz, zobs, x0, P, RHO_IDX, DT, n)
    elif method == "truncated":
        run = lambda: param_gradient(lorenz, zobs, x0, P, RHO_IDX, DT, n, int(round(L_TRUNC / DT)))
    else:
        run = lambda: lss_derivative(lorenz, zobs, integrate(lorenz, x0, P, DT, n)[1:], P, RHO_IDX, DT, ALPHA)[0]
    jax.block_until_ready(run())
    t = time.time()
    for _ in range(5):
        jax.block_until_ready(run())
    print(json.dumps({"method": method, "T": T, "wall_s": (time.time() - t) / 5,
                      "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}))

elif mode == "cost":
    rows = []
    base = json.loads(subprocess.run([sys.executable, "-c",
        "import resource, jax, chaos_jax, jax.numpy as jnp; jnp.zeros(1).block_until_ready(); import json;"
        "print(json.dumps({'maxrss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}))"],
        capture_output=True, text=True).stdout)["maxrss_mb"]
    for method in ("naive", "truncated", "lss"):
        for T in TS:
            out = subprocess.run([sys.executable, __file__, "cost-one", method, str(T)], capture_output=True, text=True).stdout
            r = json.loads(out.strip().splitlines()[-1])
            r["peak_above_bare_mb"] = r["maxrss_mb"] - base
            rows.append(r)
            print(r, flush=True)
    (OUT / "phase4_cost.json").write_text(json.dumps({"bare_process_mb": base, "rows": rows}, indent=2) + "\n")
