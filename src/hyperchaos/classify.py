"""Classification of a spectrum result (rules fixed in docs/PREDICTIONS-H1.md)."""
from dataclasses import dataclass

import numpy as np

EPS_ABS = 0.005
INCONSISTENT_ZERO = 0.05
CONSERVATIVE_SUM = 0.005
SUM_TOL_REL = 1e-3  # same tolerance as gate G1


@dataclass
class Classification:
    label: str
    n_pos: int = 0
    n_zero: int = 0
    n_neg: int = 0
    eps: float = float("nan")
    z_err: float = float("nan")
    conservative: bool = False


def label_from_exponents(lam, se, eps_abs=EPS_ABS, rule="v2"):
    """Label a bounded, non-fixed orbit from its sorted exponents and their standard errors.

    rule "v1" (pre-registered, kept to reproduce the first run): one eps for all exponents,
    eps = max(eps_abs, 3 * largest SE, 3 * |z_err|).
    rule "v2" (Amendment 2): exponent i is positive if lambda_i > max(eps_abs, 3 * SE_i, 3 * |z_err|).
    """
    lam = np.asarray(lam, dtype=float)
    se = np.asarray(se, dtype=float)
    iz = int(np.argmin(np.abs(lam)))
    z_err = float(lam[iz])
    if abs(z_err) > INCONSISTENT_ZERO:
        return Classification("inconsistent", z_err=z_err)
    if rule == "v1":
        eps = max(eps_abs, 3.0 * float(np.max(se)), 3.0 * abs(z_err))
        eps_i = np.full(len(lam), eps)
    else:
        eps_i = np.maximum(eps_abs, np.maximum(3.0 * se, 3.0 * abs(z_err)))
        eps = float(np.max(eps_i[np.arange(len(lam)) != iz])) if len(lam) > 1 else eps_abs
    pos = lam > eps_i
    neg = lam < -eps_i
    pos[iz] = False
    neg[iz] = False
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    n_zero = len(lam) - n_pos - n_neg
    if n_pos == 0:
        label = "periodic" if n_zero == 1 else "quasi-periodic"
    elif n_pos == 1:
        label = "chaotic"
    else:
        label = "hyperchaotic"
    conservative = abs(float(lam.sum())) < CONSERVATIVE_SUM
    return Classification(label, n_pos, n_zero, n_neg, eps, z_err, conservative)


def classify(res, eps_abs=EPS_ABS, rule="v2"):
    if (not res.finite) or res.max_abs > 1e8:
        return Classification("diverges")
    if np.all(res.late_range < 1e-6 * (1.0 + res.max_abs)):
        return Classification("fixed point")
    # Amendment 4: the same sum check as gate G1. An orbit that fails it (e.g. a linear drift
    # on an invariant subspace) is not a bounded attractor and its exponents mean nothing.
    if abs(float(np.sum(res.exponents)) - res.mean_div) > SUM_TOL_REL * max(1.0, abs(res.mean_div)):
        return Classification("inconsistent")
    return label_from_exponents(res.exponents, res.se, eps_abs, rule)
