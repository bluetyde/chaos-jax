"""H2 runs. Usage: python experiments/h2_run.py <part> [args]
  g1                      published cases A-S through the H1 evaluator (gate G1)
  g3 <workers>            screen recall (gate G3, Amendment 1)
  g4                      planted non-chaotic systems (gate G4, Amendment 1)
  calib <workers> <n>     expected hits per published case (n draws on each support)
  search <tag> <workers> <n_total> <chunk> [seed0]   random search over D, appends results/h2_<tag>.jsonl
Each part writes results/h2_<part>.json (or .jsonl) and prints one line per item.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import permutations
from math import comb

import numpy as np

CASES = list("ABCDEFGHIJKLMNOPQRS")
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


# ----- worker functions (top level so they can be pickled) -----
def w_calib(args):
    case, seed, n = args
    from hyperchaos import search3d as s

    rng = np.random.default_rng(seed)
    sup = s.support_of(s.case_coefficients(case))
    cs = s.sample_on_support(rng, sup, n)
    idx, lam, ka = s.screen(cs)
    n2 = 0
    conf = 0
    ex = []
    for i in idx:
        ok, r = s.is_chaotic(cs[i], 5000.0)
        if ok:
            n2 += 1
            ok3, _ = s.confirm(cs[i], 20000.0)
            if ok3:
                conf += 1
                if len(ex) < 3:
                    ex.append(cs[i].tolist())
    return dict(case=case, n=n, keep_a=int(ka.sum()), keep_b=int(len(idx)), stage2=n2, confirmed=conf, examples=ex)


def w_g3(args):
    case, seed, n = args
    from hyperchaos import search3d as s

    rng = np.random.default_rng(seed)
    sup = s.support_of(s.case_coefficients(case))
    cs = s.sample_on_support(rng, sup, n)
    idx, lam, ka = s.screen(cs)
    kept = set(idx.tolist())
    pos = 0
    pos_kept = 0
    for i in range(n):
        ok, _ = s.is_chaotic(cs[i], 5000.0)
        if ok:
            pos += 1
            pos_kept += i in kept
    return dict(case=case, n=n, evaluator_chaotic=pos, stage1_kept=pos_kept, stage1_survivors=len(kept))


def w_search(args):
    seed, n = args
    from hyperchaos import search3d as s

    rng = np.random.default_rng(seed)
    t0 = time.time()
    cnt = dict(draws=0, keep_a=0, keep_b=0, stage2=0, confirmed=0)
    hits = []
    for _ in range(0, n, 20000):
        m = min(20000, n - cnt["draws"])
        cs = s.sample(rng, m)
        idx, lam, ka = s.screen(cs)
        cnt["draws"] += m
        cnt["keep_a"] += int(ka.sum())
        cnt["keep_b"] += int(len(idx))
        for i in idx:
            ok, r = s.is_chaotic(cs[i], 5000.0)
            if not ok:
                continue
            cnt["stage2"] += 1
            ok3, r3 = s.confirm(cs[i], 20000.0)
            if ok3:
                cnt["confirmed"] += 1
                sup = s.support_of(cs[i])
                hits.append(dict(c=cs[i].tolist(), k=len(sup), canon=list(s.canonical_support(sup)),
                                 exps=r3.exponents.tolist(), se=r3.se.tolist()))
    cnt["secs"] = time.time() - t0
    return dict(seed=seed, counts=cnt, hits=hits)


# ----- parts -----
def part_g1():
    from hyperchaos import search3d as s
    from hyperchaos.classify import classify
    from hyperchaos.spectrum import spectrum_p

    starts = [s.START, np.array([0.1, -0.05, 0.07]), np.array([-0.05, 0.08, 0.02])]
    rows = []
    for case in CASES:
        c = s.case_coefficients(case)
        pub = np.sort(np.array(s.PUBLISHED_EXPONENTS[case]))[::-1]
        for k, x0 in enumerate(starts):
            r = spectrum_p(s.flow, c, x0, dt=0.01, T=100000.0)
            cl = classify(r, rule="v2")
            diff = (r.exponents - pub).tolist()
            tol = [max(0.01, 0.05 * abs(p)) for p in pub]
            match = all(abs(d) <= t for d, t in zip(diff, tol))
            rows.append(dict(case=case, start=k, label=cl.label, conservative=cl.conservative, exps=r.exponents.tolist(),
                             se=r.se.tolist(), published=pub.tolist(), diff=diff, match=match))
            print(case, k, cl.label, "cons" if cl.conservative else "", np.round(r.exponents, 4).tolist(), "match" if match else "MISMATCH", flush=True)
            json.dump(rows, open(os.path.join(RES, "h2_g1.json"), "w"), indent=1)
    return rows


def part_g3(workers):
    ex = ProcessPoolExecutor(max_workers=workers)
    tasks = [(c, 100 + i, 2000) for i, c in enumerate("FBCSKQ")]
    out = []
    for r in ex.map(w_g3, tasks):
        out.append(r)
        print(r, flush=True)
    pos = sum(r["evaluator_chaotic"] for r in out)
    kept = sum(r["stage1_kept"] for r in out)
    print("pooled evaluator-chaotic", pos, "kept by stage 1", kept, "recall", kept / max(pos, 1))
    json.dump(out, open(os.path.join(RES, "h2_g3.json"), "w"), indent=1)


def part_g4():
    from hyperchaos import search3d as s

    def lin(rows):
        c = np.zeros(30)
        for e, m, v in rows:
            c[10 * e + s.NAMES.index(m)] = v
        return c

    ross = lambda cc: lin([(0, "y", -1), (0, "z", -1), (1, "x", 1), (1, "y", 0.2), (2, "1", 0.2), (2, "xz", 1), (2, "z", -cc)])  # noqa: E731
    planted = {
        "fixed sink": lin([(0, "x", -1), (1, "y", -1), (2, "z", -1)]),
        "spiral sink": lin([(0, "x", -1), (0, "y", 1), (1, "x", -1), (1, "y", -1), (2, "z", -1)]),
        "centre": lin([(0, "y", 1), (1, "x", -1), (2, "z", -1)]),
        "divergent": lin([(0, "1", 1), (0, "xx", 1), (1, "y", -1), (2, "z", -1)]),
        "Sprott A": s.case_coefficients("A"),
        "Rossler c=2.5": ross(2.5),
        "Rossler c=3": ross(3.0),
        "Rossler c=3.5": ross(3.5),
        "Rossler c=4": ross(4.0),
        "Lorenz rho=14": lin([(0, "x", -10), (0, "y", 10), (1, "x", 14), (1, "y", -1), (1, "xz", -1), (2, "xy", 1), (2, "z", -2.7)]),
    }
    rows = []
    for name, c in planted.items():
        ok_truth, r = s.is_chaotic(c, 20000.0)
        idx, lam, ka = s.screen(c[None, :])
        passed1 = len(idx) == 1
        ok2, _ = s.is_chaotic(c, 5000.0) if passed1 else (False, None)
        conf, _ = s.confirm(c, 20000.0) if passed1 and ok2 else (False, None)
        stage = "stage 1a" if not ka[0] else ("stage 1b" if not passed1 else ("stage 2" if not ok2 else ("confirmation" if not conf else "NOT REJECTED")))
        rows.append(dict(name=name, evaluator_chaotic_T20000=bool(ok_truth), stage1a=bool(ka[0]), stage1b=bool(passed1), stage2=bool(ok2),
                         confirmed=bool(conf), rejected_at=stage, lam_est=lam.tolist()))
        print(rows[-1], flush=True)
    json.dump(rows, open(os.path.join(RES, "h2_g4.json"), "w"), indent=1)


def part_calib(workers, n):
    from hyperchaos import search3d as s

    ex = ProcessPoolExecutor(max_workers=workers)
    tasks = [(c, 500 + i, int(n)) for i, c in enumerate(CASES)]
    rows = []
    for r in ex.map(w_calib, tasks):
        sup = s.support_of(s.case_coefficients(r["case"]))
        k = len(sup)
        cls = len(s.support_class(sup))
        p = 0.5 * cls / comb(30, k)
        f = r["confirmed"] / r["n"]
        r.update(k=k, class_size=cls, p_support=p, f_chaotic=f)
        rows.append(r)
        print(r["case"], "k", k, "class", cls, "p %.2e" % p, "f %.2e" % f, "keep_b", r["keep_b"], "stage2", r["stage2"], "conf", r["confirmed"], flush=True)
        json.dump(rows, open(os.path.join(RES, "h2_calib.json"), "w"), indent=1)


def part_search(tag, workers, n_total, chunk, seed0):
    path = os.path.join(RES, "h2_%s.jsonl" % tag)
    if os.path.exists(path):
        raise SystemExit("exists, delete first: " + path)
    ex = ProcessPoolExecutor(max_workers=workers)
    n_tasks = int(n_total // chunk)
    tasks = [(seed0 + i, int(chunk)) for i in range(n_tasks)]
    tot = dict(draws=0, keep_a=0, keep_b=0, stage2=0, confirmed=0)
    t0 = time.time()
    with open(path, "a") as fh:
        for r in ex.map(w_search, tasks):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            for k in tot:
                tot[k] += r["counts"][k]
            print("seed", r["seed"], r["counts"], "| total", tot, "| wall %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    part = sys.argv[1]
    if part == "g1":
        part_g1()
    elif part == "g3":
        part_g3(int(sys.argv[2]))
    elif part == "g4":
        part_g4()
    elif part == "calib":
        part_calib(int(sys.argv[2]), sys.argv[3])
    elif part == "search":
        part_search(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]) if len(sys.argv) > 6 else 1000)
