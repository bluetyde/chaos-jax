"""Phase 4 control: d/dw of the mean square of the driven damped oscillator, with a closed form."""
import jax
import jax.numpy as jnp

from chaos_jax.fixes import avg_obs, lss_derivative
from chaos_jax.integrate import integrate, integrate_final
from chaos_jax.systems import (driven_oscillator_mean_square, driven_oscillator_params,
                               driven_oscillator_phase)

W0, N_WINDOW, PERIODS, SPIN_WINDOWS, W_IDX = 1.5, 20000, 10, 3, 3


def x2(x):
    return x[0] ** 2


def exact_derivative():
    return float(jax.grad(lambda w: driven_oscillator_mean_square(driven_oscillator_params(w=w)))(W0))


def start_points(key=0, n=4):
    xy = jax.random.uniform(jax.random.PRNGKey(key), (n, 2), minval=-2.0, maxval=2.0)
    return jnp.concatenate([xy, jnp.zeros((n, 1))], axis=1)


def window_dt(w):
    """dt such that N_WINDOW steps are exactly PERIODS whole periods at frequency w."""
    return 2.0 * jnp.pi * PERIODS / (w * N_WINDOW)


def naive_or_truncated(x0, seg_time=None):
    """d/dw of the exact-window mean square, through the spin-up as well, so no transient is left.

    seg_time=None: plain reverse-mode autodiff. seg_time: stop the gradient every seg_time time units.
    """
    p0 = driven_oscillator_params(w=W0)

    def q(w):
        dt = window_dt(w)
        seg = None if seg_time is None else max(1, int(round(seg_time / float(window_dt(W0)))))
        return avg_obs(driven_oscillator_phase, x2, x0, p0.at[W_IDX].set(w), dt,
                       (SPIN_WINDOWS + 1) * N_WINDOW, seg_steps=seg, n_skip=SPIN_WINDOWS * N_WINDOW)

    return float(jax.grad(q)(W0))


def lss(x0, alpha=1.0):
    p0 = driven_oscillator_params(w=W0)
    dt = window_dt(W0)
    xs = integrate(driven_oscillator_phase, x0, p0, dt, (SPIN_WINDOWS + 1) * N_WINDOW)[SPIN_WINDOWS * N_WINDOW + 1:]
    return float(lss_derivative(driven_oscillator_phase, x2, xs, p0, W_IDX, dt, alpha)[0])


def lss_scaling_error(periods, per_period, alpha=1.0, c=0.5, w=1.5, spin_periods=60):
    """Relative error of LSS on the control for a window of `periods` whole periods. Used to show 1/T convergence."""
    p0 = driven_oscillator_params(c=c, w=w)
    exact = float(jax.grad(lambda ww: driven_oscillator_mean_square(driven_oscillator_params(c=c, w=ww)))(w))
    dt = 2.0 * jnp.pi / (w * per_period)
    spin = spin_periods * per_period
    xs = integrate(driven_oscillator_phase, jnp.array([0.3, -0.2, 0.0]), p0, dt, spin + periods * per_period)[spin + 1:]
    d = lss_derivative(driven_oscillator_phase, x2, xs, p0, W_IDX, dt, alpha)[0]
    return abs(float(d) - exact) / abs(exact)
