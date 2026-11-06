"""Gate G7: a pinned lyapax (black box, public API only) against our spectrum on L, LS and R79."""
import numpy as np
import pytest

from hyperchaos.spectrum import spectrum  # noqa: F401  (float64 is switched on on import)
from hyperchaos.systems import ALL

from conftest import SEED, T_MAIN, main_run


@pytest.mark.parametrize("sid", ["L", "LS", "R79"])
def test_g7_lyapax_agrees(sid):
    import lyapax  # an ImportError here is a failure of G7, not a skip

    s = ALL[sid]
    x0 = s.start(SEED + 50)
    prob = lyapax.ode_problem(s.f, x0, s.dt, "rk4")
    n_steps = int(round(min(T_MAIN, 20000.0) / s.dt))
    renorm = int(round(0.1 / s.dt))
    n_steps = (n_steps // renorm) * renorm
    res = lyapax.lyapunov_spectrum(prob, n_steps=n_steps, renorm_every=renorm, t_transient=200.0)
    theirs = np.sort(np.asarray(res.exponents))[::-1]
    ours = main_run(sid).exponents
    for i, (a, b) in enumerate(zip(ours, theirs)):
        tol = 0.01 if i < 2 else max(0.02, 0.05 * abs(a))
        assert abs(a - b) <= tol, (sid, i, ours.tolist(), theirs.tolist())
