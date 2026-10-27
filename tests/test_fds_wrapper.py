"""A7b control C-F1: the numpy and scipy Lorenz wrappers for fds match the JAX integrator."""
import sys
import pathlib

import jax.numpy as jnp
import numpy as np
import pytest

import chaos_jax  # noqa: F401
from chaos_jax.integrate import integrate
from chaos_jax.systems import lorenz

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments"))
import fds_lorenz as W  # noqa: E402

U0 = np.array([1.0, 2.0, 25.0])


def _jax_reference(name, value, nsteps):
    p = dict(W.DEFAULTS)
    p[name] = value
    xs = np.asarray(integrate(lorenz, jnp.array(U0), jnp.array([p["sigma"], p["rho"], p["beta"]]), W.DT, nsteps))
    return xs[-1], xs[1:, 2]


@pytest.mark.parametrize("name", ["rho", "sigma", "beta"])
@pytest.mark.parametrize("shift", [0.0, 1e-3])
def test_c_f1_rk4_wrapper_matches_jax(name, shift):
    value = W.DEFAULTS[name] + shift
    u, z = W.make_run(name, "rk4")(U0, value, 1000)
    u_ref, z_ref = _jax_reference(name, value, 1000)
    assert np.linalg.norm(u - u_ref) / np.linalg.norm(u_ref) < 1e-9
    assert np.max(np.abs(z - z_ref)) / np.max(np.abs(z_ref)) < 1e-9


@pytest.mark.parametrize("name", ["rho", "sigma", "beta"])
def test_c_f1_dop853_wrapper_matches_jax(name):
    value = W.DEFAULTS[name] + 1e-3
    u, z = W.make_run(name, "dop853")(U0, value, 200)
    u_ref, z_ref = _jax_reference(name, value, 200)
    assert np.linalg.norm(u - u_ref) / np.linalg.norm(u_ref) < 1e-6
    assert np.max(np.abs(z - z_ref)) / np.max(np.abs(z_ref)) < 1e-6
