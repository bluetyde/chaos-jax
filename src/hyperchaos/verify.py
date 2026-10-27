"""The four verification checks V1 to V4 (definitions in docs/PREDICTIONS-H1.md)."""
import numpy as np

from hyperchaos.classify import classify, label_from_exponents
from hyperchaos.reference_scipy import spectrum_scipy
from hyperchaos.spectrum import spectrum


def _combined_ok(a, b, k=3.0, floor=0.0):
    tol = np.maximum(floor, k * np.sqrt(a.se**2 + b.se**2))
    return bool(np.all(np.abs(a.exponents - b.exponents) <= tol))


def verify(system, main, x0, T=5000.0, T_spin=200.0, v1_T=300.0, v4_T=2000.0):
    """`main` is the spectrum of `system` from `x0` with the default settings. Returns a dict."""
    out = {}
    main_label = classify(main).label

    short = spectrum(system.f, x0, dt=system.dt, T=v1_T, T_spin=T_spin)
    sc = spectrum_scipy(system.f, x0, T=v1_T, T_spin=T_spin)
    lab_sc = label_from_exponents(sc.exponents, sc.se).label
    lab_short = classify(short).label
    out["V1"] = _combined_ok(short, sc) and lab_sc == lab_short
    out["V1_detail"] = {"rk4": short.exponents.tolist(), "dop853": sc.exponents.tolist(), "label_dop853": lab_sc}

    half = spectrum(system.f, x0, dt=system.dt / 2.0, T=T, T_spin=T_spin)
    out["V2"] = _combined_ok(main, half, floor=0.002) and classify(half).label == main_label
    out["V2_detail"] = {"half": half.exponents.tolist()}

    k = main.n_blocks // 2
    h1 = main.blocks[:k]
    h2 = main.blocks[k:]
    se1 = h1.std(axis=0, ddof=1) / np.sqrt(k)
    se2 = h2.std(axis=0, ddof=1) / np.sqrt(len(h2))
    d = np.abs(h1.mean(axis=0) - h2.mean(axis=0))
    out["V3"] = bool(np.all(d <= 3.0 * np.sqrt(se1**2 + se2**2)))
    out["V3_detail"] = {"first_half": h1.mean(axis=0).tolist(), "second_half": h2.mean(axis=0).tolist()}

    runs = [spectrum(system.f, system.start(100 + i), dt=system.dt, T=v4_T, T_spin=T_spin) for i in range(4)]
    labels = [classify(r).label for r in runs]
    ex = np.array([r.exponents for r in runs])
    sev = np.array([r.se for r in runs])
    w = 1.0 / np.maximum(sev, 1e-12) ** 2
    pooled = (ex * w).sum(axis=0) / w.sum(axis=0)
    within = bool(np.all(np.abs(ex - pooled) <= 3.0 * np.sqrt(sev**2 + (1.0 / np.sqrt(w.sum(axis=0))) ** 2)))
    out["V4"] = bool(len(set(labels)) == 1 and labels[0] == main_label and within)
    out["V4_detail"] = {"labels": labels, "exponents": ex.tolist()}
    out["verified"] = bool(out["V1"] and out["V2"] and out["V3"] and out["V4"])
    return out
