"""H2F W3a analysis of results/h2f_profile1b.npz: which simple stop rules decide stage 1b early, and what they would save. Usage: python experiments/h2f_analyze1b.py"""
import json
import os

import numpy as np

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
d = np.load(os.path.join(RES, "h2f_profile1b.npz"))
logn, m, tf, keep = d["logn"], d["m"], d["tf"], d["keep"]
N, OUT = logn.shape
SPIN, INNER, DT = 200, 25, 0.02
print("systems", N, "keepers", int(keep.sum()), "escaped (|x|>1e4 or non-finite) at some chunk:", int((tf >= 0).sum()))
cum = np.cumsum(np.nan_to_num(logn, nan=0.0, posinf=0.0, neginf=0.0), axis=1)


def lam_run(c):
    """running exponent estimate after c chunks (needs c > SPIN)"""
    return (cum[:, c - 1] - cum[:, SPIN - 1]) / ((c - SPIN) * INNER * DT)


esc_chunk = np.where(tf >= 0, tf, 10**9)
res = {}
print("share of systems that have escaped by chunk t:", {t: round(float((esc_chunk <= t).mean()), 3) for t in (100, 200, 300, 600, 900)})
for theta in (-np.inf, -0.3, -0.1, -0.05, -0.02, 0.0, 0.005):
    for cps in ([300, 400, 500, 600, 700, 800, 900, 1000, 1100], [400, 800], [600], [300, 600, 900]):
        stop = np.full(N, OUT)
        stop_exit = np.full(N, OUT)  # immediate exit on escape (a GPU thread can break), rounds for the exponent rule
        fired = np.zeros(N, bool)
        for c in cps:
            rule = (esc_chunk <= c) | (lam_run(c) < theta)
            new = rule & ~fired
            stop[new] = c
            stop_exit[new] = np.minimum(c, np.where(esc_chunk[new] <= c, esc_chunk[new], c))
            fired |= rule
        lost = int((fired & keep).sum())
        saving = OUT * N / stop.sum()
        saving_exit = OUT * N / stop_exit.sum()
        res[f"theta={theta} checkpoints={cps}"] = dict(keepers_lost=lost, speedup_rounds=float(saving), speedup_with_exit_on_escape=float(saving_exit),
                                                       share_fired_by_600=float((fired & (stop <= 600)).mean()))
        print("theta %6.3f cps %-45s keepers lost %3d  speedup rounds %.2fx  with immediate escape exit %.2fx  share of systems stopped by chunk 600: %.2f"
              % (theta, str(cps), lost, saving, saving_exit, (fired & (stop <= 600)).mean()))
# share of all stage-1b steps spent on systems that a rule (theta 0, checkpoints every 100) stops by chunk 600
fired600 = np.zeros(N, bool)
for c in (300, 400, 500, 600):
    fired600 |= (esc_chunk <= c) | (lam_run(c) < 0.0)
print("P-F6 quantity: share of all stage-1b steps spent on systems already decided by chunk 600 (rule: escaped or running exponent < 0): %.3f" % fired600.mean())
print("  of which keepers wrongly included:", int((fired600 & keep).sum()))
res["share_steps_on_systems_decided_by_600"] = float(fired600.mean())
json.dump(res, open(os.path.join(RES, "h2f_profile1b_rules.json"), "w"), indent=1)
