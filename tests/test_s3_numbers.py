"""A5c regression test: S3 and the shadowing contribution on the Solenoid map, small sample, against the noise-free oracle.

256 orbits of 8000 steps. The standard error of the s2 total is about 0.7%, so the tolerances below are 5% (s2 total),
1% (s1 total and s1 shadowing). These tolerances are test settings chosen before planting any bug. They are not the A5c gates.
"""
import sys
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

import chaos_jax  # noqa: F401
from chaos_jax import s3, solenoid

M, N, K_RUN, TAIL, N_LAGS = 256, 8000, 100, 100, 24


def _run(s):
    k1, k2 = jax.random.split(jax.random.PRNGKey(11))
    x0 = jnp.stack([jnp.ones(M), jax.random.uniform(k1, (M,), maxval=2 * jnp.pi), jnp.zeros(M)], axis=1)
    q0 = jax.random.normal(k2, (M, 3))
    es = jnp.eye(2)

    def one(x, q):
        return s3.estimates(s3.orbit_arrays(solenoid.phi, solenoid.objective, jnp.array(s), es, x, q, K_RUN + N + TAIL), K_RUN, N, N_LAGS)

    outs = [jax.jit(jax.vmap(one))(x0[i:i + 32], q0[i:i + 32]) for i in range(0, M, 32)]
    return {k: np.concatenate([np.asarray(o[k]) for o in outs]) for k in outs[0]}


def test_s3_total_and_shadow_against_oracle():
    d = _run((1.0, 0.0))
    o1, o2 = solenoid.oracle_derivatives(1.0, 0.0)
    total = (d["stable"] + d["unstable"]).mean(axis=0)
    shadow = d["shadow"].mean(axis=0)
    assert abs(total[0] - o1) / abs(o1) < 0.01
    assert abs(total[1] - o2) / abs(o2) < 0.05
    assert abs(shadow[0] - o1) / abs(o1) < 0.01
