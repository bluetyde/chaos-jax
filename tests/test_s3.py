"""A5c controls C-M1, C-M2 and C-M4 for the S3 algorithm on the Solenoid map."""
import jax
import jax.numpy as jnp
import numpy as np

import chaos_jax  # noqa: F401
from chaos_jax import s3, solenoid

S0 = jnp.array([1.0, 0.0])
ES = jnp.eye(2)


def _fd(fun, x, h=1e-6):
    cols = []
    for i in range(x.shape[0]):
        d = jnp.zeros_like(x).at[i].set(h)
        cols.append((fun(x + d) - fun(x - d)) / (2 * h))
    return jnp.stack(cols, axis=-1)


def test_c_m1_derivatives_match_finite_differences():
    f, dphi, d2, dsphi, ddsphi_q = s3.make_ops(solenoid.phi, S0)
    x = jnp.array([1.3, 2.1, -0.4])
    u, v, e = jnp.array([0.3, 1.0, 0.2]), jnp.array([-0.5, 0.7, 0.9]), jnp.array([0.0, 1.0])
    assert np.max(np.abs(dphi(x) - _fd(f, x))) < 1e-7
    assert np.max(np.abs(d2(x, u, v) - _fd(lambda y: dphi(y) @ u, x) @ v)) < 1e-6
    ds = (solenoid.phi(x, S0 + 1e-6 * e) - solenoid.phi(x, S0 - 1e-6 * e)) / 2e-6
    assert np.max(np.abs(dsphi(x, e) - ds)) < 1e-7
    assert np.max(np.abs(ddsphi_q(x, e, u) - _fd(lambda y: dsphi(y, e), x) @ u)) < 1e-6


def _arrays(n=1500, s=S0):
    return s3.orbit_arrays(solenoid.phi, solenoid.objective, s, ES, jnp.array([1.0, 1.0, 0.0]), jnp.array([0.3, 1.0, 0.2]), n)


def test_c_m2_orthogonality_identities():
    a = _arrays()
    q, v, p, y = (np.asarray(a[k]) for k in ("q", "v", "p", "y"))
    assert np.max(np.abs(np.einsum("ni,ni->n", q, q) - 1.0)) < 1e-12
    assert np.max(np.abs(np.einsum("npi,ni->np", v, q))) < 1e-12
    assert np.max(np.abs(np.einsum("npi,ni->np", y, q) + np.einsum("npi,ni->np", v, p))) < 1e-12


def test_c_m4_shadow_solution_solves_tangent_equation_and_is_bounded():
    s = jnp.array([1.0, 0.3])
    a = _arrays(2000, s)
    f, dphi, d2, dsphi, ddsphi_q = s3.make_ops(solenoid.phi, s)
    e = np.asarray(s3.shadow_coefficient(a["a"], a["alpha"]))  # (L, P)
    v, q, x = np.asarray(a["v"]), np.asarray(a["q"]), np.asarray(a["x"])
    u = v + e[:, :, None] * q[:, None, :]  # shadowing direction at point i
    worst = 0.0
    for i in range(1, 1900):
        J = np.asarray(dphi(jnp.asarray(x[i - 1])))
        for P in range(2):
            lhs = u[i, P]
            rhs = J @ u[i - 1, P] + np.asarray(a["chi"][i, P])
            worst = max(worst, float(np.max(np.abs(lhs - rhs))))
    assert worst < 1e-12
    assert np.max(np.linalg.norm(u[:1900], axis=-1)) < 50.0
