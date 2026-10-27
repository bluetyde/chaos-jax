"""Second integrator for check V1: scipy DOP853 with the variational equation, QR every tau."""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

from hyperchaos.spectrum import SpectrumResult


def spectrum_scipy(f, x0, T=300.0, T_spin=200.0, n_blocks=10, tau=1.0, rtol=1e-10, atol=1e-12):
    x0 = np.asarray(x0, dtype=float)
    n = x0.shape[0]
    jac = jax.jacfwd(f)

    @jax.jit
    def rhs_state(x):
        return f(x)

    @jax.jit
    def rhs_aug(y):
        x = y[:n]
        q = y[n:].reshape(n, n)
        return jnp.concatenate([f(x), (jac(x) @ q).reshape(-1)])

    spin = solve_ivp(
        lambda t, x: np.asarray(rhs_state(x)),
        (0.0, T_spin),
        x0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    x = spin.y[:, -1]
    q = np.eye(n)
    steps_per_block = int(round(T / n_blocks / tau))
    blocks = np.zeros((n_blocks, n))
    max_abs = float(np.max(np.abs(x)))
    for b in range(n_blocks):
        ssum = np.zeros(n)
        for _ in range(steps_per_block):
            y0 = np.concatenate([x, q.reshape(-1)])
            sol = solve_ivp(
                lambda t, y: np.asarray(rhs_aug(y)),
                (0.0, tau),
                y0,
                method="DOP853",
                rtol=rtol,
                atol=atol,
            )
            y1 = sol.y[:, -1]
            x = y1[:n]
            q, r = np.linalg.qr(y1[n:].reshape(n, n))
            ssum += np.log(np.abs(np.diag(r)))
            max_abs = max(max_abs, float(np.max(np.abs(x))))
        blocks[b] = ssum / (steps_per_block * tau)
    exps = blocks.mean(axis=0)
    order = np.argsort(-exps, kind="stable")
    blocks = blocks[:, order]
    exps = exps[order]
    se = blocks.std(axis=0, ddof=1) / np.sqrt(n_blocks)
    return SpectrumResult(
        exponents=exps,
        se=se,
        blocks=blocks,
        mean_div=float("nan"),
        max_abs=max_abs,
        finite=bool(np.all(np.isfinite(blocks))),
        late_range=np.full(n, np.nan),
        x_end=x,
        dt=float("nan"),
        T=float(T),
        n_blocks=int(n_blocks),
    )
