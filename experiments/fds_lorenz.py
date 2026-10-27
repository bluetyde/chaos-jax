"""Lorenz-63 wrappers in the form fds expects: run(u0, parameter, nsteps) -> (final state, z after each step).

Plain numpy and scipy, no JAX, so the file also loads in the scratch environment that holds fds.
make_run(name, integrator) fixes the other two parameters at (sigma, rho, beta) = (10, 28, 8/3).
Design in docs/PREDICTIONS-A7b.md.
"""
import numpy as np

DEFAULTS = {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0}
DT = 0.005


def rhs(u, sigma, rho, beta):
    return np.array([sigma * (u[1] - u[0]), u[0] * (rho - u[2]) - u[1], u[0] * u[1] - beta * u[2]])


def _params(name, value):
    p = dict(DEFAULTS)
    p[name] = float(value)
    return p["sigma"], p["rho"], p["beta"]


def run_rk4(name, u0, value, nsteps, dt=DT):
    sigma, rho, beta = _params(name, value)
    u = np.array(u0, dtype=float)
    z = np.empty(nsteps)
    for i in range(nsteps):
        k1 = rhs(u, sigma, rho, beta)
        k2 = rhs(u + 0.5 * dt * k1, sigma, rho, beta)
        k3 = rhs(u + 0.5 * dt * k2, sigma, rho, beta)
        k4 = rhs(u + dt * k3, sigma, rho, beta)
        u = u + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        z[i] = u[2]
    return u, z


def run_dop853(name, u0, value, nsteps, dt=DT, tol=1e-12):
    from scipy.integrate import solve_ivp
    sigma, rho, beta = _params(name, value)
    t_eval = dt * np.arange(1, nsteps + 1)
    sol = solve_ivp(lambda t, u: rhs(u, sigma, rho, beta), (0.0, t_eval[-1]), np.array(u0, dtype=float), method="DOP853",
                    t_eval=t_eval, rtol=tol, atol=tol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:, -1], sol.y[2]


def make_run(name, integrator="rk4"):
    fn = {"rk4": run_rk4, "dop853": run_dop853}[integrator]
    return lambda u0, value, nsteps: fn(name, u0, value, nsteps)
