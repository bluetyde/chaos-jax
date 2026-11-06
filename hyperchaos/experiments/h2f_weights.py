"""H2F W1b: offline test of frozen sampling weights (docs/PREDICTIONS-H2F.md). Fit on hits of seeds 1000-1249, test on seeds 1250-1499.
Usage: python experiments/h2f_weights.py"""
import json
import os

import numpy as np

from hyperchaos import orbits

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
rows = [json.loads(line) for line in open(os.path.join(RES, "h2_main.jsonl"))]
train = {5: [], 6: []}
test = {5: [], 6: []}
for r in rows:
    for h in r["hits"]:
        sup = (np.abs(np.array(h["c"])) > 0).astype(float)
        (train if r["seed"] < 1250 else test)[h["k"]].append(sup)
print({k: (len(train[k]), len(test[k])) for k in (5, 6)})


def augment(X):
    """Each support enlarged by its 48 symmetric images (hit rate is invariant under the symmetries)."""
    if len(X) == 0:
        return np.zeros((0, 30))
    return np.abs(orbits.images(np.array(X))).reshape(-1, 30)


def uniform_supports(rng, k, m):
    rank = np.argsort(rng.random((m, 30)), axis=1).argsort(axis=1)
    return (rank < k).astype(float)


def fit(X, U, lam, steps=400):
    """Maximise sum_i theta.x_i - n log mean_U exp(theta.u) - lam |theta|^2 by gradient ascent with a decaying step."""
    th = np.zeros(30)
    n = len(X)
    sx = X.sum(0)
    for it in range(steps):
        a = U @ th
        w = np.exp(a - a.max())
        ew = (w[:, None] * U).sum(0) / w.sum()
        g = sx - n * ew - 2 * lam * th
        th += g / (n + 2 * lam + 1e-9) * 1.0 / (1 + 0.01 * it) * 2.0
    return th


def loglik_gain(th, Xt, U):
    a = U @ th
    mx = a.max()
    logz = mx + np.log(np.mean(np.exp(a - mx)))
    return float(np.mean(Xt @ th) - logz)


rng = np.random.default_rng(0)
result = {}
thetas = {}
for k in (5, 6):
    U = uniform_supports(rng, k, 100000)
    Xtr = np.array(train[k])
    Xte = np.array(test[k])
    # choose the ridge penalty by 5-fold cross-validation on training hits only (augmentation inside each fold)
    folds = np.array_split(rng.permutation(len(Xtr)), 5)
    best = None
    for lam in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
        sc = []
        for f in folds:
            mask = np.ones(len(Xtr), bool)
            mask[f] = False
            th = fit(augment(Xtr[mask]), U, lam * 48)
            sc.append(loglik_gain(th, Xtr[f], U))
        m = float(np.mean(sc))
        print("k", k, "lam", lam, "cv mean log-lik gain %.4f" % m)
        if best is None or m > best[0]:
            best = (m, lam)
    th = fit(augment(Xtr), U, best[1] * 48)
    a = U @ th
    w = np.exp(a - a.max())
    w = w / w.mean()
    floor = 0.1
    wf = np.maximum(w, floor)
    w_te = np.maximum(np.exp(Xte @ th - a.max()) / np.exp(a - a.max()).mean(), floor)
    gain = float(w_te.mean() / wf.mean())
    gain_nofloor = float((np.exp(Xte @ th - a.max()) / np.exp(a - a.max()).mean()).mean())
    # control: the same fit on shuffled slot labels of the training data (no real signal) evaluated on the real test hits
    Xsh = Xtr[:, rng.permutation(30)]
    ths = fit(augment(Xsh), U, best[1] * 48)
    as_ = U @ ths
    ws = np.maximum(np.exp(Xte @ ths - as_.max()) / np.exp(as_ - as_.max()).mean(), floor)
    wfs = np.maximum(np.exp(as_ - as_.max()) / np.exp(as_ - as_.max()).mean(), floor)
    ctrl = float(ws.mean() / wfs.mean())
    # cap used by a rejection sampler: effective weight min(w, cap) with cap = 99.9th percentile of w on uniform supports (acceptance is capped weight / cap)
    cap = float(np.quantile(w, 0.999))
    w_te_raw = np.exp(Xte @ th - a.max()) / np.exp(a - a.max()).mean()
    eff = lambda v: np.minimum(np.maximum(v, floor), cap)  # noqa: E731
    gain_cap = float(eff(w_te_raw).mean() / eff(w).mean())
    acc = float(eff(w).mean() / cap)
    thetas[str(k)] = th
    result[str(k)] = dict(cap=cap, gain_floored_capped=gain_cap, acceptance_rate=acc, n_train=len(Xtr), n_test=len(Xte), lam=best[1], gain_floored=gain, gain_no_floor=gain_nofloor, control_shuffled_gain=ctrl,
                          theta=[round(float(v), 3) for v in th])
    print("k", k, "n_train", len(Xtr), "n_test", len(Xte), "best lam", best[1], "gain (floor 0.1): %.3f" % gain, "no floor %.3f" % gain_nofloor, "control %.3f" % ctrl)

# trivial weighting: only the share of k = 5 and 6 (hit counts per k over the whole main run, equal draw shares assumed)
h5 = len(train[5]) + len(test[5])
h6 = len(train[6]) + len(test[6])
result["k_share"] = dict(hits_k5=h5, hits_k6=h6, gain_all_k6_over_uniform_mix=h6 / ((h5 + h6) / 2), gain_all_k5_over_uniform_mix=h5 / ((h5 + h6) / 2))
print(result["k_share"])
json.dump(result, open(os.path.join(RES, "h2f_weights.json"), "w"), indent=1)
np.savez(os.path.join(RES, "h2f_weights_theta.npz"), k5=thetas["5"], k6=thetas["6"], cap5=result["5"]["cap"], cap6=result["6"]["cap"], floor=0.1)
