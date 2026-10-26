"""Tangent propagation through one RK4 step with jax.jvp."""
import jax

from chaos_jax.integrate import rk4_step


def step_and_tangent(f, x, V, dt, p):
    """Advance x one RK4 step and push the columns of V through the step Jacobian.

    V has shape (n, k). Returns (x_next, V_next) with V_next = (d x_next / d x) V.
    """
    def g(y):
        return rk4_step(f, y, dt, p)

    x_next = g(x)
    push = jax.vmap(lambda v: jax.jvp(g, (x,), (v,))[1], in_axes=1, out_axes=1)
    return x_next, push(V)


def state_jacobian(f, x0, p, dt, n_steps):
    """d x(n_steps) / d x0 by forward-mode autodiff through the whole unrolled scan."""
    from chaos_jax.integrate import integrate_final
    return jax.jacfwd(lambda y: integrate_final(f, y, p, dt, n_steps))(x0)
