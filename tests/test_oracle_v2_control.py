"""A1 control: the three-point pipeline (slope and curvature) must match closed-form derivatives."""
import jax
import jax.numpy as jnp
import numpy as np

import chaos_jax  # noqa: F401
from chaos_jax.response import ensemble_time_average
from chaos_jax.systems import driven_oscillator, driven_oscillator_mean_square, driven_oscillator_params

N_PER_WINDOW, PERIODS, W0 = 20000, 10, 1.5


def _x2(x):
    return x[0] ** 2


def _means(w, x0s):
    p = driven_oscillator_params(w=w)
    dt = 2.0 * jnp.pi * PERIODS / (w * N_PER_WINDOW)
    return ensemble_time_average(driven_oscillator, _x2, x0s, p, dt, 3 * N_PER_WINDOW, N_PER_WINDOW)


def _three_point(h, x0s):
    data = jnp.stack([_means(W0 - h, x0s), _means(W0, x0s), _means(W0 + h, x0s)], axis=1)
    m = data.mean(axis=0)
    return float((m[2] - m[0]) / (2 * h)), float((m[2] + m[0] - 2 * m[1]) / h ** 2)


def _closed_form(w):
    ms = lambda ww: driven_oscillator_mean_square(driven_oscillator_params(w=ww))
    return float(jax.grad(ms)(w)), float(jax.hessian(ms)(w))


def _starts():
    xy = jax.random.uniform(jax.random.PRNGKey(0), (4, 2), minval=-2.0, maxval=2.0)
    return jnp.concatenate([xy, jnp.zeros((4, 1))], axis=1)


def test_slope_gate():
    exact, _ = _closed_form(W0)
    slope, _ = _three_point(1e-4, _starts())
    rel = abs(slope - exact) / abs(exact)
    print("A1 control slope rel err:", rel)
    assert rel < 1e-6


def test_curvature_gate():
    _, exact = _closed_form(W0)
    _, curv = _three_point(0.02, _starts())
    rel = abs(curv - exact) / abs(exact)
    print("A1 control curvature rel err:", rel)
    assert rel < 1e-3
