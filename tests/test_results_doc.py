"""RESULTS.md quotes headline numbers. This test checks that each one still matches its result file."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEXT = (ROOT / "RESULTS.md").read_text()


def load(name):
    return json.loads((ROOT / "results" / name).read_text())


def _check(value, digits):
    s = f"{value:.{digits}f}"
    assert s in TEXT, f"{s} (from a result file) is not in RESULTS.md"


def test_phase1_numbers():
    p = load("phase1_measurements.json")
    _check(p["lorenz_start0"]["spectrum_tangent"][0], 4)
    _check(p["lorenz_start1"]["spectrum_tangent"][0], 4)
    _check(p["pendulum_high"]["lambda1_tangent"], 4)


def test_phase2_and_follow_up():
    p = load("phase2_oracle.json")
    _check(p["estimate_central_h1"], 4)
    _check(p["se_central_h1"], 4)
    _check(load("phase2b_wide_slope.json")["A"]["line_slope"], 4)


def test_phase3_and_phase4():
    p3 = load("phase3.json")
    _check(p3["lorenz_slope"], 4)
    assert f"T* = {p3['T_star']}" in TEXT
    p4 = load("phase4.json")["80"]
    _check(p4["d_lss"]["median"], 3)
    _check(p4["b_truncated"]["median"], 3)
    _check(p4["c_ensemble_truncated"]["median"], 3)
    cost = {r["method"]: r["wall_s"] * 1000 for r in load("phase4_cost.json")["rows"] if r["T"] == 80}
    assert f"{cost['lss']:.0f} ms" in TEXT


def test_oracle_v2_and_noise():
    a1 = load("phaseA1_oracle_v2.json")
    _check(a1["rho"]["weighted_mean_all_rungs"], 4)
    _check(a1["sigma"]["weighted_mean_all_rungs"], 4)
    _check(a1["beta"]["final_slope"], 3)
    a3b = load("phaseA3b_noise.json")
    _check(a3b["rho"]["N2_S"], 4)
    _check(a3b["sigma"]["N2_S"], 4)


def test_shadowing_codes():
    a2 = load("phaseA2_report.json")["stage2"]["params"]
    _check(a2["rho"]["nilss"], 5)
    _check(a2["sigma"]["nilss"], 5)
    _check(a2["beta"]["nilss"], 5)
    lss = load("phaseA3_lss.json")["cases"]["T320_a10"]
    _check(lss["rho"]["mean"], 5)
    _check(lss["sigma"]["mean"], 5)
    _check(lss["beta"]["mean"], 5)
    own = load("phaseA5_report.json")
    _check(own["rho"]["own"], 5)
    _check(own["sigma"]["own"], 5)
    _check(own["beta"]["own"], 5)


def test_calibration_counts():
    a4 = load("phaseA4_calibration.json")
    assert f"{a4['oracle_converged_cases']} of 13" in TEXT
    assert f"{a4['ok_ensembles_in_converged_cases']} `ok`" in TEXT
    assert a4["A4.1_pass"] and a4["cases_with_stable_verdict"] == 13
