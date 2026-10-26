"""Vector fields. Each has the signature f(x, p) -> dx/dt, with p a 1-D parameter array."""
import jax.numpy as jnp

LORENZ_SIGMA, LORENZ_RHO, LORENZ_BETA = 10.0, 28.0, 8.0 / 3.0
GRAVITY = 9.81


def lorenz_params(sigma=LORENZ_SIGMA, rho=LORENZ_RHO, beta=LORENZ_BETA):
    return jnp.array([sigma, rho, beta])


def lorenz(x, p):
    """Lorenz-63. p = (sigma, rho, beta)."""
    sigma, rho, beta = p[0], p[1], p[2]
    return jnp.array([
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2],
    ])


def damped_oscillator_params(c=0.4, k=4.0):
    return jnp.array([c, k])


def damped_oscillator(x, p):
    """x'' + c x' + k x = 0 with state (x, v). p = (c, k)."""
    c, k = p[0], p[1]
    return jnp.array([x[1], -c * x[1] - k * x[0]])


def pendulum_params(g=GRAVITY, l1=1.0, l2=1.0, m1=1.0, m2=1.0):
    return jnp.array([g, l1, l2, m1, m2])


def double_pendulum(x, p):
    """Planar double pendulum. State (th1, th2, w1, w2), angles from the downward vertical.

    p = (g, l1, l2, m1, m2).
    """
    g, l1, l2, m1, m2 = p[0], p[1], p[2], p[3], p[4]
    th1, th2, w1, w2 = x[0], x[1], x[2], x[3]
    d = th1 - th2
    den = 2.0 * m1 + m2 - m2 * jnp.cos(2.0 * d)
    a1 = (-g * (2.0 * m1 + m2) * jnp.sin(th1)
          - m2 * g * jnp.sin(th1 - 2.0 * th2)
          - 2.0 * jnp.sin(d) * m2 * (w2 ** 2 * l2 + w1 ** 2 * l1 * jnp.cos(d))) / (l1 * den)
    a2 = (2.0 * jnp.sin(d) * (w1 ** 2 * l1 * (m1 + m2)
                              + g * (m1 + m2) * jnp.cos(th1)
                              + w2 ** 2 * l2 * m2 * jnp.cos(d))) / (l2 * den)
    return jnp.array([w1, w2, a1, a2])


def double_pendulum_energy(x, p):
    """Total energy (kinetic + potential) of the double pendulum."""
    g, l1, l2, m1, m2 = p[0], p[1], p[2], p[3], p[4]
    th1, th2, w1, w2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    kinetic = (0.5 * (m1 + m2) * l1 ** 2 * w1 ** 2
               + 0.5 * m2 * l2 ** 2 * w2 ** 2
               + m2 * l1 * l2 * w1 * w2 * jnp.cos(th1 - th2))
    potential = -(m1 + m2) * g * l1 * jnp.cos(th1) - m2 * g * l2 * jnp.cos(th2)
    return kinetic + potential


def driven_oscillator_params(c=0.5, k=4.0, F=1.0, w=1.5):
    return jnp.array([c, k, F, w])


def driven_oscillator(x, p):
    """x'' + c x' + k x = F cos(w t), made autonomous with state (x, v, t). p = (c, k, F, w)."""
    c, k, F, w = p[0], p[1], p[2], p[3]
    return jnp.array([x[1], -c * x[1] - k * x[0] + F * jnp.cos(w * x[2]), 1.0])


def driven_oscillator_mean_square(p):
    """Closed-form steady-state time average of x^2."""
    c, k, F, w = p[0], p[1], p[2], p[3]
    return F ** 2 / (2.0 * ((k - w ** 2) ** 2 + c ** 2 * w ** 2))


def driven_oscillator_phase(x, p):
    """The driven oscillator with a phase state (x, v, theta), theta' = w. p = (c, k, F, w).

    Same motion as `driven_oscillator`. Forcing is bounded, so it suits LSS.
    """
    c, k, F, w = p[0], p[1], p[2], p[3]
    return jnp.array([x[1], -c * x[1] - k * x[0] + F * jnp.cos(x[2]), w])
