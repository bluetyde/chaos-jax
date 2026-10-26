"""Energy drift of the high-energy double pendulum over T=10 for a ladder of dt."""
import json
import pathlib

import jax.numpy as jnp

import chaos_jax  # noqa: F401
from chaos_jax import cases
from chaos_jax.integrate import integrate
from chaos_jax.systems import double_pendulum, double_pendulum_energy, pendulum_params

p = pendulum_params()
x0 = cases.vec(cases.PENDULUM_HIGH)
rows = []
for dt in cases.PENDULUM_DT_LADDER:
    n = int(round(cases.PENDULUM_ENERGY_T / dt))
    xs = integrate(double_pendulum, x0, p, dt, n)
    e = double_pendulum_energy(xs, p)
    drift = float(jnp.max(jnp.abs(e - e[0])) / jnp.abs(e[0]))
    rows.append({"dt": dt, "steps": n, "max_rel_energy_drift": drift})
    print(f"dt={dt:<8} steps={n:<6} drift={drift:.3e}")
ok = [r for r in rows if r["max_rel_energy_drift"] < 1e-6 / cases.ENERGY_DRIFT_MARGIN]
chosen = max(r["dt"] for r in ok) if ok else None
print("chosen dt:", chosen)
out = pathlib.Path(__file__).resolve().parents[1] / "results"
out.mkdir(exist_ok=True)
(out / "pendulum_dt_scan.json").write_text(json.dumps({"rows": rows, "chosen_dt": chosen}, indent=2) + "\n")
