"""G2.3: the ensemble finite-difference pipeline must reproduce a closed-form derivative."""
import jax
import jax.numpy as jnp
import numpy as np

import chaos_jax  # noqa: F401
from chaos_jax.response import bootstrap_se, ensemble_time_average
from chaos_jax.systems import driven_oscillator, driven_oscillator_mean_square, driven_oscillator_params

N_PER_WINDOW = 20000
PERIODS = 10
H = 1e-4


def _x2(x):
    return x[0] ** 2


def _pipeline_mean(w, x0s):
    """Same pipeline as Lorenz, except the window is a whole number of periods."""
    p = driven_oscillator_params(w=w)
    window = 2.0 * jnp.pi * PERIODS / w
    dt = window / N_PER_WINDOW
    return ensemble_time_average(driven_oscillator, _x2, x0s, p, dt, 3 * N_PER_WINDOW, N_PER_WINDOW)


def _closed_form_derivative(w0):
    return jax.grad(lambda w: driven_oscillator_mean_square(driven_oscillator_params(w=w)))(w0)


def test_closed_form_matches_hand_derivative():
    c, k, F, w = 0.5, 4.0, 1.0, 1.5
    D = (k - w ** 2) ** 2 + c ** 2 * w ** 2
    dD = -4.0 * w * (k - w ** 2) + 2.0 * c ** 2 * w
    hand = -F ** 2 * dD / (2.0 * D ** 2)
    np.testing.assert_allclose(float(_closed_form_derivative(w)), hand, rtol=1e-12)


def test_g2_3_driven_oscillator_pipeline_matches_closed_form():
    w0 = 1.5
    key = jax.random.PRNGKey(0)
    x0s = jnp.concatenate([jax.random.uniform(key, (4, 2), minval=-2.0, maxval=2.0), jnp.zeros((4, 1))], axis=1)
    data = jnp.stack([_pipeline_mean(w0 - H, x0s), _pipeline_mean(w0 + H, x0s)], axis=1)
    mean = data.mean(axis=0)
    fd = float((mean[1] - mean[0]) / (2 * H))
    exact = float(_closed_form_derivative(w0))
    rel = abs(fd - exact) / abs(exact)
    print("G2.3 fd, exact, rel err:", fd, exact, rel)
    assert rel < 1e-6, (fd, exact, rel)
    # Start points should not matter once the transient is gone.
    assert float(jnp.std(data[:, 0])) < 1e-10


def test_bootstrap_se_matches_theory_on_normal_data():
    data = jax.random.normal(jax.random.PRNGKey(3), (2000, 1)) * 2.0
    se = float(bootstrap_se(lambda m: m[0], data, jax.random.PRNGKey(4), 2000))
    assert abs(se - 2.0 / np.sqrt(2000)) / (2.0 / np.sqrt(2000)) < 0.1
