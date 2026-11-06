"""Space-split sensitivity (S3) for a map with a one-dimensional unstable manifold, and the shadowing contribution.

Written from the mathematics of arXiv 2103.08816, Section 4.1, Eqs. 4.5 to 4.10 (checked on the page images).
For a map phi(x, s) on R^m and a unit parameter direction e, the perturbation field at x_{n+1} is
chi_{n+1} = (d phi / d s)(x_n) . e.

Per orbit, one step n -> n + 1 does:
    alpha q'  = dphi q                                   (4.5)
    t         = dphi v + chi'
    a'        = q' . t,   v' = t - a' q'                 (4.6)
    p'        = (d2phi(q, q) + dphi p) / alpha^2         (4.7)
    Y0        = (d2phi(q, v) + dphi y) / alpha + (d chi)' q'
    c'        = Y0 . q' - a' (p' . q') + v' . p'         (from y' . q' = - v' . p')
    y'        = Y0 - c' q' - a' p'                       (4.8)
with (d chi)' q' = [ d(d phi / d s . e) q ] / alpha, because chi(phi(x)) = (d phi / d s)(x) and dphi q = alpha q'.

The shadowing direction (the only bounded solution of u' = dphi u + chi') is u_n = v_n + e_n q_n, where
e_n = (e_{n+1} - a_{n+1}) / alpha_{n+1} is run backward from e = 0 at the end of the orbit.
"""
import jax
import jax.numpy as jnp


def make_ops(phi, s0):
    f = lambda x: phi(x, s0)

    def dphi(x):
        return jax.jacfwd(f)(x)

    def d2(x, u, v):  # second derivative of phi at x in directions u and v
        g = lambda y: jax.jvp(f, (y,), (u,))[1]
        return jax.jvp(g, (x,), (v,))[1]

    def dsphi(x, e):  # d phi / d s . e
        return jax.jvp(lambda s: phi(x, s), (s0,), (e,))[1]

    def ddsphi_q(x, e, q):  # d( d phi / d s . e ) in direction q
        return jax.jvp(lambda y: dsphi(y, e), (x,), (q,))[1]

    return f, dphi, d2, dsphi, ddsphi_q


def orbit_arrays(phi, obj, s0, es, x0, q0, n_steps):
    """Run n_steps of the recursions for the parameter directions es, shape (P, ns). Returns arrays indexed by step n + 1."""
    f, dphi, d2, dsphi, ddsphi_q = make_ops(phi, s0)
    dobj = jax.grad(obj)
    P = es.shape[0]

    def step(carry, _):
        x, q, p, v, y = carry
        J1 = dphi(x)
        dq = J1 @ q
        alpha = jnp.linalg.norm(dq)
        q1 = dq / alpha
        p1 = (d2(x, q, q) + J1 @ p) / alpha ** 2

        def per_param(e, v, y):
            chi1 = dsphi(x, e)
            t = J1 @ v + chi1
            a1 = q1 @ t
            v1 = t - a1 * q1
            dchiq = ddsphi_q(x, e, q) / alpha
            Y0 = (d2(x, q, v) + J1 @ y) / alpha + dchiq
            c1 = Y0 @ q1 - a1 * (p1 @ q1) + v1 @ p1
            y1 = Y0 - c1 * q1 - a1 * p1
            return v1, y1, a1, c1, chi1, t

        v1, y1, a1, c1, chi1, t = jax.vmap(per_param)(es, v, y)
        x1 = f(x)
        gJ = dobj(x1)
        out = dict(J=obj(x1), dJv=v1 @ gJ, dJq=q1 @ gJ, c=c1, a=a1, alpha=alpha, q=q1, v=v1, p=p1, y=y1, chi=chi1, t=t, x=x1)
        return (x1, q1, p1, v1, y1), out

    m = x0.shape[0]
    carry0 = (x0, q0 / jnp.linalg.norm(q0), jnp.zeros(m), jnp.zeros((P, m)), jnp.zeros((P, m)))
    _, out = jax.lax.scan(step, carry0, None, length=n_steps)
    return out


def shadow_coefficient(a, alpha):
    """e_n from e_n = (e_{n+1} - a_{n+1}) / alpha_{n+1}, e = 0 at the last index. a has shape (L, P), alpha shape (L,).
    Index n of the result is the value at the point with index n (the arrays are indexed by step n + 1 = point index)."""
    def back(e_next, inp):
        a_next, al_next = inp
        e = (e_next - a_next) / al_next
        return e, e

    L = alpha.shape[0]
    _, es = jax.lax.scan(back, jnp.zeros(a.shape[1]), (a[1:][::-1], alpha[1:][::-1]))
    return jnp.concatenate([es[::-1], jnp.zeros((1, a.shape[1]))], axis=0)


def estimates(arr, n_burn, n_avg, n_lags, sign_u=1.0, use_c=True):
    """Per-orbit S3 and shadowing estimates from the arrays of orbit_arrays.
    Points n = 0 .. n_avg - 1 after n_burn burn-in steps are averaged; the arrays must hold n_burn + n_avg + tail steps."""
    sl = slice(n_burn, None)
    J, c, dJv, dJq, a, alpha = (arr[k][sl] for k in ("J", "c", "dJv", "dJq", "a", "alpha"))
    e = shadow_coefficient(a, alpha)  # (L, P)
    stable = dJv[:n_avg].mean(axis=0)
    unstable = jnp.zeros_like(stable)
    for k in range(n_lags):
        unstable = unstable - (J[k:k + n_avg][:, None] * c[:n_avg]).mean(axis=0)
    shadow = (dJv[:n_avg] + e[:n_avg] * dJq[:n_avg][:, None]).mean(axis=0)
    return dict(stable=stable, unstable=sign_u * unstable, shadow=shadow, meanJ=J[:n_avg].mean())
