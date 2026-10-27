"""Smale-Williams Solenoid map (Eq. 7.2 of arXiv 2103.08816) and a noise-free oracle for E[r^2].

State x = (r, theta, z), parameters s = (s1, s2), reference s = (1, 0):
    r'     = s1 + (r - s1) / 4 + cos(theta) / 2
    theta' = (2 theta + (s2 / 4) sin(4 theta)) mod 2 pi
    z'     = z / 4 + sin(theta) / 2
Objective J = r^2. The theta map does not depend on r, z or s1, so it is an expanding circle map.
"""
import jax.numpy as jnp
import numpy as np

REF = (1.0, 0.0)


def phi(x, s):
    r, th, z = x[0], x[1], x[2]
    s1, s2 = s[0], s[1]
    return jnp.array([s1 + (r - s1) / 4.0 + jnp.cos(th) / 2.0,
                      jnp.mod(2.0 * th + 0.25 * s2 * jnp.sin(4.0 * th), 2.0 * jnp.pi),
                      z / 4.0 + jnp.sin(th) / 2.0])


def objective(x):
    return x[0] ** 2


def _transfer_matrix(s2, nb, nq):
    th = 2.0 * np.pi * np.arange(nq) / nq
    T = 2.0 * th + 0.25 * s2 * np.sin(4.0 * th)
    ks = np.arange(-nb, nb + 1)
    # A_kl = mean_j exp(i (l th_j - k T_j))
    return np.exp(1j * (ks[None, :, None] * th[None, None, :] - ks[:, None, None] * T[None, None, :])).mean(axis=-1)


def expected_J(s1, s2, nb=48, nq=1024, n_lags=60):
    """E[r^2] under the SRB measure, from the Fourier transfer operator of the theta map (no sampling noise)."""
    A = _transfer_matrix(s2, nb, nq)
    w, V = np.linalg.eig(A)
    rho = V[:, np.argmin(np.abs(w - 1.0))]
    rho = rho / rho[nb] / (2.0 * np.pi)  # coefficient of k = 0 is 1 / (2 pi), so the density integrates to one
    Ec = 2.0 * np.pi * rho[nb + 1].real  # E[cos theta]
    u = np.zeros_like(rho)
    u[1:-1] = 0.5 * (rho[:-2] + rho[2:])  # Fourier coefficients of cos(theta) * rho
    C = []
    for _ in range(n_lags + 1):
        C.append(np.pi * (u[nb + 1] + u[nb - 1]).real)  # C(m) = E[cos theta_0 cos theta_m]
        u = A @ u
    var = (4.0 / 15.0) * (C[0] + 2.0 * sum(4.0 ** (-m) * C[m] for m in range(1, n_lags + 1)))
    return s1 ** 2 + 2.0 * s1 * (2.0 / 3.0) * Ec + var, Ec


def oracle_derivatives(s1, s2, h=0.02, **kw):
    """(d E[J] / d s1, d E[J] / d s2). The s1 derivative is exact: 2 s1 + (4/3) E[cos theta].
    The s2 derivative uses a four-point central stencil of the noise-free expected value."""
    e = lambda t: expected_J(s1, t, **kw)[0]
    d1 = 2.0 * s1 + (4.0 / 3.0) * expected_J(s1, s2, **kw)[1]
    d2 = (-e(s2 + 2 * h) + 8 * e(s2 + h) - 8 * e(s2 - h) + e(s2 - 2 * h)) / (12 * h)
    return d1, d2
