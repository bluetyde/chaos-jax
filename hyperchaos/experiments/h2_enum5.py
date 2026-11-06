"""Exhaustive 5-term pass and its random-search baseline (docs/PREDICTIONS-H2-ENUM5.md).
Usage: python -u experiments/h2_enum5.py baseline <workers> <n_seeds> <seed0>   -> results/h2_enum5_baseline.json (random k=5 draws, night-run pipeline, seconds per draw)
       python -u experiments/h2_enum5.py run <workers>                         -> results/h2_enum5/ chunk files, results/h2_enum5_summary.json
       python -u experiments/h2_enum5.py controls                              -> Sprott cases A-E through the same worker"""
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
COARSE = [round(-3.0 + 0.5 * i, 1) for i in range(13)]
PER = 40


def w_chunk(args):
    """Screen and confirm every representative of a list of 5-term supports (flip-reduced signs, invariant grid)."""
    sups, grid = args
    from hyperchaos import enumerate as en
    from hyperchaos import search3d as s

    n = ka = kb = bd = st2 = 0
    hits, cons = [], []
    for sup in sups:
        cs = en.reps(sup, grid, dedupe_flips=True)
        idx, lam, keep_a = s.screen(cs)
        n += len(cs)
        ka += int(keep_a.sum())
        kb += len(idx)
        if len(idx):
            idx = idx[s.bound_keep(cs[idx])]
        bd += len(idx)
        for j in idx:
            r2, k2 = s.evaluate(cs[j], s.START, 5000.0)
            if k2.label == "chaotic" and k2.conservative:
                cons.append(dict(c=cs[j].tolist(), exps=r2.exponents.tolist(), support=list(sup)))
                continue
            ok, rr = s.is_chaotic(cs[j], 5000.0)
            if not ok:
                continue
            st2 += 1
            ok3, r3 = s.confirm(cs[j], 20000.0)
            if ok3:
                hits.append(dict(c=cs[j].tolist(), exps=r3.exponents.tolist(), se=r3.se.tolist(), support=list(sup)))
    return dict(n=n, keep_a=ka, keep_b=kb, bound=bd, stage2=st2, hits=hits, cons=cons)


def merge(parts):
    out = dict(n=0, keep_a=0, keep_b=0, bound=0, stage2=0, hits=[], cons=[])
    for p in parts:
        for k in ("n", "keep_a", "keep_b", "bound", "stage2"):
            out[k] += p[k]
        out["hits"] += p["hits"]
        out["cons"] += p["cons"]
    return out


def b_phase1(seed):
    from hyperchaos import search3d as s

    cs = s.sample(np.random.default_rng(seed), 20000, ks=(5,))
    idx, lam, ka = s.screen(cs)
    if len(idx):
        idx = idx[s.bound_keep(cs[idx])]
    return dict(keep_a=int(ka.sum()), keep_b=int(len(idx)), cs=cs[idx])


def b_phase2(args):
    cs, batch = args
    from hyperchaos import search3d as s

    return s.f32_keep(cs, batch=batch) if len(cs) else np.zeros(0, bool)


def b_phase3(c):
    from hyperchaos import search3d as s

    ok, r = s.is_chaotic(c, 5000.0)
    if not ok:
        return None
    ok3, r3 = s.confirm(c, 20000.0)
    return dict(hit=ok3, c=c.tolist() if ok3 else None)


if __name__ == "__main__":
    mode = sys.argv[1]
    t0 = time.time()
    if mode == "baseline":
        workers, n_seeds, seed0 = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        with ProcessPoolExecutor(max_workers=workers) as ex:
            t = time.time()
            p1 = list(ex.map(b_phase1, range(seed0, seed0 + n_seeds)))
            t1 = time.time() - t
            pool = np.concatenate([p["cs"] for p in p1])
            t = time.time()
            keeps = list(ex.map(b_phase2, [(sl, 128) for sl in np.array_split(pool, workers)]))
            t2 = time.time() - t
            kept = pool[np.concatenate(keeps)]
            t = time.time()
            p3 = [r for r in ex.map(b_phase3, list(kept)) if r is not None]
            t3 = time.time() - t
        draws = 20000 * n_seeds
        out = dict(draws=draws, workers=workers, t_phase1=t1, t_phase2=t2, t_phase3=t3, wall=t1 + t2 + t3, sec_per_draw=(t1 + t2 + t3) / draws, stage2=len(p3), hits=sum(r["hit"] for r in p3))
        print(out)
        json.dump(out, open(os.path.join(RES, "h2_enum5_baseline.json"), "w"), indent=1)
    elif mode == "controls":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from hyperchaos import search3d as s

        res = {}
        with ProcessPoolExecutor(max_workers=5) as ex:
            sups = {k: tuple(sorted(s.support_of(s.case_coefficients(k)))) for k in "ABCDE"}
            futs = {k: ex.submit(w_chunk, ([sups[k]], COARSE)) for k in "ABCDE"}
            for k, f in futs.items():
                r = f.result()
                res[k] = dict(n=r["n"], stage2=r["stage2"], hits=len(r["hits"]), cons=len(r["cons"]))
                print(k, res[k], flush=True)
        json.dump(res, open(os.path.join(RES, "h2_enum5_controls.json"), "w"), indent=1)
    else:
        workers = int(sys.argv[2])
        from hyperchaos import enumerate as en

        d = os.path.join(RES, "h2_enum5")
        os.makedirs(d, exist_ok=True)
        classes = en.support_classes(5)
        chunks = [(classes[i : i + PER], COARSE) for i in range(0, len(classes), PER)]
        path = lambda i: os.path.join(d, "chunk_%03d.json" % i)  # noqa: E731
        todo = [i for i in range(len(chunks)) if not os.path.exists(path(i))]
        print("classes", len(classes), "chunks", len(chunks), "to run", len(todo), flush=True)
        stall = float(os.environ.get("H2_STALL_S", "1800"))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            it = iter(todo)
            pending = {}
            for _ in range(workers * 2):
                i = next(it, None)
                if i is not None:
                    pending[ex.submit(w_chunk, chunks[i])] = i
            last = time.time()
            done_n = 0
            while pending:
                fin, _ = wait(list(pending), timeout=60, return_when=FIRST_COMPLETED)
                if not fin and time.time() - last > stall:
                    print("STOP: no chunk finished for %.0f s; killing the worker processes" % stall, flush=True)
                    for p in list(ex._processes.values()):
                        p.terminate()
                    ex.shutdown(wait=False, cancel_futures=True)
                    sys.exit(2)
                for f in fin:
                    i = pending.pop(f)
                    json.dump(f.result(), open(path(i) + ".tmp", "w"))
                    os.replace(path(i) + ".tmp", path(i))
                    last = time.time()
                    done_n += 1
                    if done_n % 10 == 0:
                        print("%d/%d chunks this run, elapsed %.0f s" % (done_n, len(todo), time.time() - t0), flush=True)
                    j = next(it, None)
                    if j is not None:
                        pending[ex.submit(w_chunk, chunks[j])] = j
        out = merge([json.load(open(path(i))) for i in range(len(chunks))])
        out["wall_this_run_s"] = time.time() - t0
        json.dump(out, open(os.path.join(RES, "h2_enum5_summary.json"), "w"))
        print({k: v for k, v in out.items() if k not in ("hits", "cons")}, "hits", len(out["hits"]), "cons", len(out["cons"]), flush=True)
    print("wall %.0f s" % (time.time() - t0))
