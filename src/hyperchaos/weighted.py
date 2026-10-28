"""Frozen-weight sampler (H2F W1b, docs/PREDICTIONS-H2F.md). Supports are drawn uniformly, then accepted with probability min(max(w, floor), cap) / cap,
where w(S) = exp(theta . x_S) / mean over uniform supports. Within an accepted support the coefficients are drawn exactly as in search3d.sample.
The weights are frozen (fitted once on training hits, never updated during a run) and the mix of 5- and 6-term systems is left at one half each."""
import numpy as np

from hyperchaos import search3d as s


def uniform_supports(rng, k, m):
    rank = np.argsort(rng.random((m, 30)), axis=1).argsort(axis=1)
    return (rank < k).astype(float)


class Weights:
    def __init__(self, theta, cap, floor=0.1, k=5):
        self.theta = np.asarray(theta, dtype=float)
        self.cap = float(cap)
        self.floor = float(floor)
        pool = uniform_supports(np.random.default_rng(0), k, 100000)
        a = pool @ self.theta
        self.amax = float(a.max())
        self.z = float(np.mean(np.exp(a - self.amax)))

    def eff(self, X):
        w = np.exp(X @ self.theta - self.amax) / self.z
        return np.minimum(np.maximum(w, self.floor), self.cap)


def load(path):
    f = np.load(path)
    return {5: Weights(f["k5"], f["cap5"], float(f["floor"]), 5), 6: Weights(f["k6"], f["cap6"], float(f["floor"]), 6)}


def sample_weighted(rng, n, wts):
    ks = rng.choice(np.array([5, 6]), size=n)
    out = np.zeros((n, 30))
    for k in (5, 6):
        need = int((ks == k).sum())
        if need == 0:
            continue
        got, have, tries = [], 0, 0
        while have < need:
            tries += 1
            if tries > 500:
                raise RuntimeError("weighted sampler accepts almost nothing (broken weights?)")
            X = uniform_supports(rng, k, max(4096, need * 10))
            acc = rng.random(len(X)) < wts[k].eff(X) / wts[k].cap
            got.append(X[acc])
            have += int(acc.sum())
        sup = np.concatenate(got)[:need]
        mag = s.GRID[rng.integers(0, s.N_MAG, size=(need, 30))]
        sign = np.where(rng.random((need, 30)) < 0.5, -1.0, 1.0)
        out[ks == k] = np.where(sup > 0, mag * sign, 0.0)
    return out
