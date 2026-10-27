"""A7b: fds as a black box on Lorenz-63. Design in docs/PREDICTIONS-A7b.md.

Run the fds steps with the scratch environment (it holds fds, numpy, scipy):
  python experiments/phaseA7b_run.py x0s                 (main environment, needs JAX: saves the 32 A2 start points)
  <venv>/bin/python experiments/phaseA7b_run.py control  C-F2 (Van der Pol) and a short Lorenz call
  <venv>/bin/python experiments/phaseA7b_run.py probe    wall time of one orbit
  <venv>/bin/python experiments/phaseA7b_run.py run N    main run, RK4 wrapper, N orbits, three parameters
  <venv>/bin/python experiments/phaseA7b_run.py dop      second integrator, first 8 orbits
  python experiments/phaseA7b_run.py report              gates (main environment)
"""
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE.parent / "results"
OUT.mkdir(exist_ok=True)
NAMES = ("rho", "sigma", "beta")
SEG_STEPS, RUNUP = 100, 4000


def versions():
    import importlib.metadata as m
    return {p: m.version(p) for p in ("fds", "numpy", "scipy", "mpi4py", "mpich", "dill")}


def x0s():
    import jax
    import jax.numpy as jnp
    k = jax.random.split(jax.random.PRNGKey(70), 3)  # the A2 start points, as in phaseA5_run.py
    x = jnp.stack([jax.random.uniform(k[0], (32,), minval=-15.0, maxval=15.0), jax.random.uniform(k[1], (32,), minval=-15.0, maxval=15.0),
                   jax.random.uniform(k[2], (32,), minval=5.0, maxval=45.0)], axis=1)
    np.save(OUT / "phaseA7b_x0s.npy", np.asarray(x))


def vdp_run(u, mu, nsteps):
    from scipy.integrate import odeint
    x = np.empty(nsteps)
    for i in range(nsteps):
        u = odeint(lambda v, t: np.array([v[1], mu * (1 - v[0] ** 2) * v[1] - v[0]]), u, [0, 0.01])[1]
        x[i] = u[0]
    return u, x


def vdp_x2(u, mu, nsteps):
    u, x = vdp_run(u, mu, nsteps)
    return u, x ** 2


def control():
    from fds import shadowing
    res = {"versions": versions()}
    t = time.time()
    J, G = shadowing(vdp_x2, [1, 1], 1, 1, 10, 500, 500)
    res["vdp_fds"] = {"J": float(np.ravel(J)[0]), "G": np.asarray(G).tolist(), "seconds": time.time() - t}
    m = {}
    for mu in (0.95, 1.05):
        _, x = vdp_run([1, 1], mu, 205000)
        m[mu] = float(np.mean(x[5000:] ** 2))
    fd = (m[1.05] - m[0.95]) / 0.1
    g = float(np.ravel(G)[0])
    res["vdp_fd"] = {"means": {str(k): v for k, v in m.items()}, "central_diff": fd, "rel_diff": (g - fd) / fd, "C-F2_pass": bool(abs((g - fd) / fd) < 0.10)}
    print(json.dumps(res, indent=1))
    (OUT / "phaseA7b_control.json").write_text(json.dumps(res, indent=1))


def lorenz_call(name, u0, integrator, T_steps_segments, seg_steps=SEG_STEPS, runup=RUNUP):
    from fds import shadowing
    import fds_lorenz as W
    t = time.time()
    J, G = shadowing(W.make_run(name, integrator), np.asarray(u0, dtype=float), W.DEFAULTS[name], 1, T_steps_segments, seg_steps, runup)
    return float(np.ravel(J)[0]), float(np.ravel(G)[0]), time.time() - t


def probe():
    x = np.load(OUT / "phaseA7b_x0s.npy")
    out = {}
    for integ in ("rk4",):
        out[integ] = [dict(zip(("J", "G", "seconds"), lorenz_call(n, x[0], integ, 200))) for n in ("sigma",)]
    print(json.dumps(out, indent=1))


def run(N, integrator, tag, n_seg, names=NAMES):
    x = np.load(OUT / "phaseA7b_x0s.npy")
    per = {n: [] for n in names}
    Js = {n: [] for n in names}
    secs = {n: [] for n in names}
    for i in range(N):
        for n in names:
            J, G, s = lorenz_call(n, x[i], integrator, n_seg)
            per[n].append(G)
            Js[n].append(J)
            secs[n].append(s)
        print(i, {n: per[n][-1] for n in names}, flush=True)
    res = {"versions": versions(), "integrator": integrator, "settings": dict(subspace_dimension=1, steps_per_segment=SEG_STEPS, runup_steps=RUNUP,
           num_segments=n_seg, epsilon="default 1e-6", dt=0.005, T=n_seg * SEG_STEPS * 0.005), "n_orbits": N, "per_orbit_G": per, "per_orbit_J": Js, "seconds": secs}
    (OUT / f"phaseA7b_{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "x0s":
        x0s()
    elif mode == "control":
        control()
    elif mode == "probe":
        probe()
    elif mode == "run":
        run(int(sys.argv[2]), "rk4", "main", int(sys.argv[3]) if len(sys.argv) > 3 else 2000)
    elif mode == "dop":
        run(8, "dop853", "dop853", int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
