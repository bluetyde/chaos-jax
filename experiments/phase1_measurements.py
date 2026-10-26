"""Print and save every Lyapunov number quoted in HANDOFF.md. Takes about 10 s on CPU."""
import json
import pathlib

import jax.numpy as jnp

import chaos_jax  # noqa: F401
from chaos_jax import cases
from chaos_jax.integrate import integrate_final
from chaos_jax.lyapunov import lyapunov_spectrum, lyapunov_two_trajectory
from chaos_jax.systems import (damped_oscillator, damped_oscillator_params, double_pendulum, lorenz,
                               lorenz_params, pendulum_params)

out = {}
nb = lambda T, dt: int(round(T / (cases.STEPS_PER_BLOCK * dt)))
D = cases.DISCARD_BLOCKS

p = damped_oscillator_params()
e, _, _ = lyapunov_spectrum(damped_oscillator, jnp.array(cases.OSCILLATOR_START), p, 0.01, 2000, k=2)
out["damped_oscillator_spectrum_T20"] = [float(v) for v in e]

p = lorenz_params(); dt = cases.LORENZ_DT
for i, s in enumerate(cases.LORENZ_STARTS):
    x0 = integrate_final(lorenz, cases.vec(s), p, dt, int(round(cases.LORENZ_SPINUP_T / dt)))
    e, _, _ = lyapunov_spectrum(lorenz, x0, p, dt, nb(cases.LORENZ_LAMBDA_T, dt), cases.STEPS_PER_BLOCK, k=3, discard_blocks=D)
    two = lyapunov_two_trajectory(lorenz, x0, p, dt, nb(cases.LORENZ_LAMBDA_T, dt), cases.STEPS_PER_BLOCK, discard_blocks=D)
    out[f"lorenz_start{i}"] = {"start": s, "spectrum_tangent": [float(v) for v in e], "lambda1_two_trajectory": float(two), "sum": float(e.sum())}

p = pendulum_params(); dt = cases.PENDULUM_DT
for name in ("HIGH", "LOW"):
    x0 = cases.vec(getattr(cases, f"PENDULUM_{name}"))
    n = nb(cases.PENDULUM_LAMBDA_T, dt)
    e, _, _ = lyapunov_spectrum(double_pendulum, x0, p, dt, n, cases.STEPS_PER_BLOCK, k=1, discard_blocks=D)
    two = lyapunov_two_trajectory(double_pendulum, x0, p, dt, n, cases.STEPS_PER_BLOCK, discard_blocks=D)
    out[f"pendulum_{name.lower()}"] = {"lambda1_tangent": float(e[0]), "lambda1_two_trajectory": float(two)}

print(json.dumps(out, indent=2))
r = pathlib.Path(__file__).resolve().parents[1] / "results"
r.mkdir(exist_ok=True)
(r / "phase1_measurements.json").write_text(json.dumps(out, indent=2) + "\n")
